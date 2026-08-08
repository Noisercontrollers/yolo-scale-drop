# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Difficulty-adaptive model routing: easy images -> small p34, hard images -> full model.

Collects per-image detections from both models on VOC val, computes per-image AP50, and
evaluates routing policies (oracle upper bound, p34-confidence cascade, detection-count,
image entropy) reporting mean AP50 + effective FPS + speedup.

Usage:
  python tools/difficulty_routing.py --full runs/detect/100-MuSGD/weights/best.pt \
      --small runs/detect/train/weights/best.pt --data VOC/VOC.yaml --device 0
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from feature_contribution import build_train_batches  # noqa  (unused; keep for loader helpers if needed)

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


def image_ap50(preds, gts, iou_thr=0.5):
    """Per-image AP@0.5. preds: (N,6) [conf, cls, x1,y1,x2,y2]; gts: (M,5) [cls, x1,y1,x2,y2]."""
    if gts.shape[0] == 0:
        return 1.0 if preds.shape[0] == 0 else 0.0
    if preds.shape[0] == 0:
        return 0.0
    order = preds[:, 0].argsort(descending=True)
    tp = torch.zeros(preds.shape[0], device=preds.device)
    fp = torch.zeros(preds.shape[0], device=preds.device)
    matched = torch.zeros(gts.shape[0], dtype=torch.bool, device=preds.device)
    ious = box_iou(preds[:, 2:], gts[:, 1:])  # (N, M)
    for i in order.tolist():
        cls_ok = (gts[:, 0] == preds[i, 1]).to(preds.device)
        cand = (ious[i] >= iou_thr) & cls_ok & ~matched
        if cand.any():
            j = cand.nonzero()[0].item()
            tp[i] = 1.0
            matched[j] = True
        else:
            fp[i] = 1.0
    # AP = area under PR curve (exact trapezoid)
    tp_c = tp.cumsum(0)
    fp_c = fp.cumsum(0)
    rec = tp_c / max(gts.shape[0], 1)
    prec = tp_c / (tp_c + fp_c).clamp(min=1e-9)
    # append terminal points
    rec = torch.cat([torch.tensor([0.0], device=preds.device), rec, torch.tensor([1.0], device=preds.device)])
    prec = torch.cat([torch.tensor([1.0], device=preds.device), prec, torch.tensor([0.0], device=preds.device)])
    ap = 0.0
    for r in torch.linspace(0, 1, 101, device=preds.device):
        m = prec[rec >= r]
        ap += m.max() if m.numel() else 0.0
    return ap / 101


def detections_from_out(y, conf=0.0):
    """y: (B, max_det, 6) end2end format [x1,y1,x2,y2,conf,cls] -> list per image of (N,6) [conf,cls,x1,y1,x2,y2]."""
    out = []
    for i in range(y.shape[0]):
        d = y[i]
        d = d[d[:, 4] > conf]
        if d.numel() == 0:
            out.append(torch.zeros((0, 6), device=d.device))
        else:
            out.append(torch.stack([d[:, 4], d[:, 5], d[:, 0], d[:, 1], d[:, 2], d[:, 3]], dim=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--small", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05, help="min det conf for AP computation")
    ap.add_argument("--project", default="runs/routing")
    ap.add_argument("--name", default="voc")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    torch.backends.cudnn.benchmark = True

    full = YOLO(args.full)
    full.to(device)
    full.model.fuse()
    small = YOLO(args.small)
    small.to(device)
    small.model.fuse()

    # build val loader with GT (reuse build_yolo_dataset val path)
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG, overrides=dict(data=args.data, batch=args.batch, imgsz=args.imgsz, mode="val", task="detect"))
    cfg.task = "detect"
    data_dict = check_det_dataset(args.data)
    root = Path(data_dict.get("path", "."))
    val_paths = data_dict.get("val") or []
    if isinstance(val_paths, str):
        val_paths = [val_paths]
    img_path = [str(root / x) for x in val_paths]
    ds = build_yolo_dataset(cfg, img_path, args.batch, data_dict, mode="val", rect=False, stride=32)
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)

    n_imgs = 0
    ap_full_all, ap_small_all, ap_oracle_all = [], [], []
    entropy_all = []
    p34_maxconf_all, p34_nobj_all = [], []
    t_full = t_small = 0.0

    # warmup
    with torch.no_grad():
        w = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
        for _ in range(5):
            full.model(w)
            small.model(w)
    torch.cuda.synchronize()

    with torch.no_grad():
        for batch in loader:
            img = (batch["img"].float() / 255).to(device)
            gts = batch
            B = img.shape[0]
            # per-image GT
            cls = gts["cls"].reshape(-1)
            boxes = gts["bboxes"].reshape(-1, 4)  # normalized xywh
            bidx = gts["batch_idx"].reshape(-1)
            # time both models
            t0 = time.perf_counter()
            yf = full.model(img)[0]
            torch.cuda.synchronize()
            t_full += (time.perf_counter() - t0) / B
            t0 = time.perf_counter()
            ys = small.model(img)[0]
            torch.cuda.synchronize()
            t_small += (time.perf_counter() - t0) / B

            df = detections_from_out(yf, args.conf)
            ds_ = detections_from_out(ys, args.conf)
            for b in range(B):
                # GT for this image
                m = bidx == b
                gt_cls = cls[m].to(device).float()
                gt_xywh = boxes[m].to(device)
                if gt_xywh.numel():
                    cx, cy, w, h = gt_xywh[:, 0], gt_xywh[:, 1], gt_xywh[:, 2], gt_xywh[:, 3]
                    x1 = (cx - w / 2) * args.imgsz
                    y1 = (cy - h / 2) * args.imgsz
                    x2 = (cx + w / 2) * args.imgsz
                    y2 = (cy + h / 2) * args.imgsz
                    gt = torch.stack([gt_cls, x1, y1, x2, y2], dim=1)
                else:
                    gt = torch.zeros((0, 5), device=device)
                a_f = image_ap50(df[b], gt)
                a_s = image_ap50(ds_[b], gt)
                ap_full_all.append(a_f)
                ap_small_all.append(a_s)
                ap_oracle_all.append(max(a_f, a_s))
                # proxies
                p = ds_[b]
                entropy_all.append(img[b].float().std().item())
                p34_maxconf_all.append(p[:, 0].max().item() if p.numel() else 0.0)
                p34_nobj_all.append(float(p.shape[0]))
                n_imgs += 1

    apf = sum(ap_full_all) / n_imgs
    aps = sum(ap_small_all) / n_imgs
    apo = sum(ap_oracle_all) / n_imgs
    fps_full = 1.0 / t_full
    fps_small = 1.0 / t_small
    print(f"images={n_imgs}")
    print(f"full : per-img AP50={apf:.4f}  per-img t={t_full*1000:.2f}ms -> {fps_full:.0f} FPS")
    print(f"p34  : per-img AP50={aps:.4f}  per-img t={t_small*1000:.2f}ms -> {fps_small:.0f} FPS")
    print(f"oracle routed : AP50={apo:.4f}")

    # policies
    def eval_policy(route_mask, name):
        hard = sum(1 for r in route_mask if not r) / n_imgs
        ap = sum((ap_small_all[i] if route_mask[i] else ap_full_all[i]) for i in range(n_imgs)) / n_imgs
        t = t_small + hard * t_full  # cascade: p34 always runs, full on hard
        fps = 1.0 / t
        print(f"{name:<34} p34_fraction={1 - hard:<6.2f} AP50={ap:.4f} FPS={fps:.0f} speedup={fps / fps_full:.2f}x")
        return name, 1 - hard, ap, fps, fps / fps_full

    rows = []
    rows.append(("full-only", 0.0, apf, fps_full, 1.0))
    rows.append(("p34-only", 1.0, aps, fps_small, fps_small / fps_full))
    p_hard = sum(1 for i in range(n_imgs) if ap_full_all[i] > ap_small_all[i]) / n_imgs
    fps_oracle = 1.0 / (t_small + p_hard * t_full)
    rows.append(("oracle-best", 1 - p_hard, apo, fps_oracle, fps_oracle / fps_full))

    # cascade by p34 max confidence thresholds
    for th in (0.3, 0.5, 0.7, 0.9):
        mask = [p34_maxconf_all[i] >= th for i in range(n_imgs)]
        rows.append(eval_policy(mask, f"cascade p34-conf>={th}"))
    # cascade by p34 detection count
    for c in (1, 3, 5):
        mask = [p34_nobj_all[i] <= c for i in range(n_imgs)]
        rows.append(eval_policy(mask, f"cascade p34-nobj<={c}"))
    # entropy
    import statistics
    med = statistics.median(entropy_all)
    for k in (0.7, 1.0, 1.3):
        th = med * k
        mask = [entropy_all[i] <= th for i in range(n_imgs)]
        rows.append(eval_policy(mask, f"entropy<={th:.2f}"))

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "routing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "p34_fraction", "AP50", "FPS", "speedup"])
        w.writerows(rows)
    print("saved ->", out_dir / "routing.csv")


if __name__ == "__main__":
    main()
