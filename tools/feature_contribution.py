# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Offline per-scale (P3/P4/P5) contribution analysis for YOLO26 - "Invalid Seeing Detection".

For each neck scale i feeding the Detect head we measure:
    E_i      = mean(|F_i|)                           feature energy
    G_i      = mean(|F_i * dL/dF_i|)                 gradient contribution (normalized over scales)
    C_loss_i = (L(Y^-i) - L(Y)) / L(Y)               counterfactual contribution on the detection loss
    dAP_i    = mAP(F) - mAP(F_-i)                    detection-gain contribution (val, strongest)

Usage:
    python tools/feature_contribution.py --model runs/detect/100-MuSGD/weights/best.pt ^
        --data VOC/VOC.yaml --imgsz 640 --device 0 --batch 32 --fraction 1.0
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG


def get_scale_indices(model):
    """Return the head layer indices (P3/P4/P5) that feed the Detect head."""
    seq = model.model.model  # nn.Sequential
    detect = seq[-1]
    f = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
    return [int(x) for x in f]


def zero_hook_factory():
    def hook(_m, _i, o):
        return torch.zeros_like(o if isinstance(o, torch.Tensor) else o[0])

    return hook


def collect(model, batch, scale_idx):
    """Training-mode forward + backward; returns per-scale energy, gradient contribution and baseline loss."""
    seq = model.model.model
    feats = {}

    def make_hook(i):
        def hook(_m, _i, out):
            out = out if isinstance(out, torch.Tensor) else out[0]
            out.retain_grad()
            feats[i] = out

        return hook

    handles = [seq[i].register_forward_hook(make_hook(i)) for i in scale_idx]
    try:
        loss, _ = model.model.loss(batch)
        loss = loss.sum()  # E2ELoss returns a per-component vector
        loss.backward()
    finally:
        for h in handles:
            h.remove()

    E, G = {}, {}
    for i in scale_idx:
        f = feats[i]
        E[i] = float(f.detach().abs().mean())
        grad = f.grad if f.grad is not None else torch.zeros_like(f)
        G[i] = float((f.detach().abs() * grad.abs()).mean())
    return E, G, float(loss.detach())


def counterfactual_loss(model, batch, scale_idx):
    """Unmasked loss, then zero each scale's feature map and measure relative detection-loss change."""
    seq = model.model.model
    base = float(model.model.loss(batch)[0].detach().sum())
    out = {}
    for i in scale_idx:
        hook = seq[i].register_forward_hook(zero_hook_factory())
        try:
            loss_i = float(model.model.loss(batch)[0].detach().sum())
        finally:
            hook.remove()
        out[i] = (loss_i - base) / max(base, 1e-8)
    return base, out


class _MaskBoxes(nn.Module):
    """Suppress one scale in the Detect head: zero box predictions."""

    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x):
        bs = x.shape[0]
        n = x.shape[2] * x.shape[3]
        return torch.zeros(bs, 4 * self.reg_max, n, device=x.device)


class _MaskScores(nn.Module):
    """Suppress one scale in the Detect head: -inf class scores."""

    def __init__(self, nc):
        super().__init__()
        self.nc = nc

    def forward(self, x):
        bs = x.shape[0]
        n = x.shape[2] * x.shape[3]
        return torch.full((bs, self.nc, n), float("-inf"), device=x.device)





def run_ap_ablation(model, args, scale_idx):
    """Full validation with each scale masked; returns {scale: (map, map50)} plus baseline."""
    seq = model.model.model
    detect = seq[-1]
    reg_max = detect.reg_max
    nc = detect.nc
    results = {}

    def val_with(mask_idx=None):
        handles = []
        saved = saved_o2o = None
        if mask_idx is None:
            pass
        elif args.mask_mode == "feat":
            handles.append(seq[mask_idx].register_forward_hook(zero_hook_factory()))
        else:  # head: validator fuses the model (cv2/cv3 -> None), so mask the one2one heads that inference uses
            pos = scale_idx.index(mask_idx)  # map layer index (16/19/22) to scale position (0/1/2)
            saved = [(detect.cv2[i], detect.cv3[i]) for i in range(detect.nl)] if detect.cv2 is not None else None
            saved_o2o = None
            if hasattr(detect, "one2one_cv2") and detect.one2one_cv2 is not None:
                saved_o2o = [(detect.one2one_cv2[i], detect.one2one_cv3[i]) for i in range(detect.nl)]
            mask_b, mask_s = _MaskBoxes(reg_max), _MaskScores(nc)
            if detect.cv2 is not None:
                detect.cv2 = nn.ModuleList([mask_b if i == pos else detect.cv2[i] for i in range(detect.nl)])
                detect.cv3 = nn.ModuleList([mask_s if i == pos else detect.cv3[i] for i in range(detect.nl)])
            if saved_o2o is not None:
                detect.one2one_cv2 = nn.ModuleList([mask_b if i == pos else detect.one2one_cv2[i] for i in range(detect.nl)])
                detect.one2one_cv3 = nn.ModuleList([mask_s if i == pos else detect.one2one_cv3[i] for i in range(detect.nl)])
        try:
            metrics = model.val(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=False,
                split="val",
                fraction=args.fraction,
                project=args.project,
                name=args.name,
                exist_ok=True,
                verbose=False,
            )
        finally:
            if args.mask_mode == "feat":
                for h in handles:
                    h.remove()
            elif saved is not None or saved_o2o is not None:
                if saved is not None:
                    detect.cv2 = nn.ModuleList([s[0] for s in saved])
                    detect.cv3 = nn.ModuleList([s[1] for s in saved])
                if saved_o2o is not None:
                    detect.one2one_cv2 = nn.ModuleList([s[0] for s in saved_o2o])
                    detect.one2one_cv3 = nn.ModuleList([s[1] for s in saved_o2o])
        return float(metrics.box.map), float(metrics.box.map50)

    base = val_with(None)
    for i in scale_idx:
        results[i] = val_with(i)
    return base, results


def build_train_batches(args):
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(data=args.data, batch=args.batch, imgsz=args.imgsz, mode="train", task="detect"))
    cfg.task = "detect"
    gs = 32
    data_dict = check_det_dataset(args.data)
    root = Path(data_dict.get("path", "."))
    train_paths = data_dict.get("train") or []
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    img_path = [str(root / x) for x in train_paths] if train_paths else str(root)
    ds = build_yolo_dataset(cfg, img_path, args.batch, data_dict, mode="train", rect=False, stride=gs)
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=True, rank=-1)
    return loader




def match_detections(d1, d2, iou_thr=0.5):
    """Return True if two end2end detection tensors (Nx6 conf-cls-xyxy) are equivalent after thresholding."""
    if d1.numel() == 0 and d2.numel() == 0:
        return True
    if d1.numel() == 0 or d2.numel() == 0:
        return False
    keep = lambda d: d[d[:, 0] >= 0.0]  # already thresholded by caller
    d1, d2 = keep(d1), keep(d2)
    if d1.shape[0] != d2.shape[0]:
        return False
    if d1.shape[0] == 0:
        return True
    # match by class + IoU
    from ultralytics.utils.metrics import box_iou
    iou = box_iou(d1[:, 2:], d2[:, 2:])
    matched = torch.zeros(d2.shape[0], dtype=torch.bool, device=d1.device)
    cls1, cls2 = d1[:, 1].long(), d2[:, 1].long()
    for i in range(d1.shape[0]):
        cand = (iou[i] >= iou_thr) & (cls2 == cls1[i]) & ~matched
        if not cand.any():
            return False
        matched[cand.argmax()] = True
    return True


def per_image_skippability(model, args, scale_idx, n_images, conf, iou_thr):
    """For N val images, check whether masking P4/P5 changes the image-level detections."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg(DEFAULT_CFG, overrides=dict(data=args.data, batch=1, imgsz=args.imgsz, mode="val", task="detect"))
    cfg.task = "detect"
    data_dict = check_det_dataset(args.data)
    root = Path(data_dict.get("path", "."))
    val_paths = data_dict.get("val") or []
    if isinstance(val_paths, str):
        val_paths = [val_paths]
    img_path = [str(root / x) for x in val_paths]
    ds = build_yolo_dataset(cfg, img_path, 1, data_dict, mode="val", rect=False, stride=32)
    loader = build_dataloader(ds, batch=1, workers=0, shuffle=False, rank=-1)

    seq = model.model.model
    detect = seq[-1]
    names = {16: "P3/8-small", 19: "P4/16-medium", 22: "P5/32-large"}
    model.model.eval()
    total = 0
    skip = {i: 0 for i in scale_idx}
    with torch.no_grad():
        for batch in loader:
            if total >= n_images:
                break
            img = (batch["img"].float() / 255).to(args.device)
            y, _ = model.model(img)  # full
            d_full = y[0]  # end2end detections
            for pos, i in enumerate(scale_idx):
                saved = None
                if hasattr(detect, "one2one_cv2") and detect.one2one_cv2 is not None:
                    saved = (detect.one2one_cv2[pos], detect.one2one_cv3[pos])
                    detect.one2one_cv2[pos] = _MaskBoxes(detect.reg_max)
                    detect.one2one_cv3[pos] = _MaskScores(detect.nc)
                else:
                    continue
                try:
                    y2, _ = model.model(img)
                finally:
                    detect.one2one_cv2[pos] = saved[0]
                    detect.one2one_cv3[pos] = saved[1]
                d_masked = y2[0]
                if match_detections(d_full[d_full[:, 0] >= conf], d_masked[d_masked[:, 0] >= conf], iou_thr):
                    skip[i] += 1
            total += 1
    print(f"\n===== Per-image skippability ({total} val images, conf={conf}, iou={iou_thr}, head-level mask) =====")
    for i in scale_idx:
        print(f"{names.get(i, i)}: {skip[i]}/{total} = {skip[i] / max(total, 1):.1%} images unchanged when masked")
    return skip, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--train-batches", type=int, default=4, help="batches used for energy/gradient/counterfactual stats")
    ap.add_argument("--project", default="runs/contribution")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--per-image", type=int, default=0, help="analyze N val images for per-image skippability")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--mask-mode", default="head", choices=["feat", "head"], help="feat=zero feature map; head=suppress scale head outputs (fairer)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    model = YOLO(args.model)
    model.to(device)
    model.model.requires_grad_(True)  # analysis needs grads; saved checkpoints may have grads off
    model.model.args = get_cfg(DEFAULT_CFG, overrides=dict(data=args.data, imgsz=args.imgsz, batch=args.batch))  # hyperparams for criterion
    scale_idx = get_scale_indices(model)
    names = {16: "P3/8-small", 19: "P4/16-medium", 22: "P5/32-large"}
    labels = [names.get(i, str(i)) for i in scale_idx]
    print(f"Detect head fed by layers: {[(i, labels[k]) for k, i in enumerate(scale_idx)]}")

    # ---- Training-mode stats: energy / gradient / counterfactual loss ----
    loader = build_train_batches(args)
    model.model.train()
    E_sum = {i: 0.0 for i in scale_idx}
    G_sum = {i: 0.0 for i in scale_idx}
    Cf_sum = {i: 0.0 for i in scale_idx}
    n = 0
    for batch in loader:
        if n >= args.train_batches:
            break
        batch["img"] = batch["img"].float() / 255
        batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        E, G, loss_base = collect(model, batch, scale_idx)
        _, cf = counterfactual_loss(model, batch, scale_idx)
        for i in scale_idx:
            E_sum[i] += E[i]
            G_sum[i] += G[i]
            Cf_sum[i] += cf[i]
        n += 1
    print(f"processed {n} train batches")

    g_total = sum(G_sum.values())
    print("\n===== Training-mode stats (avg over batches) =====")
    print(f"{'scale':<14}{'E_i':>12}{'G_i(rel)':>12}{'C_loss':>12}")
    for k, i in enumerate(scale_idx):
        print(f"{labels[k]:<14}{E_sum[i] / n:>12.4f}{G_sum[i] / max(g_total, 1e-9):>12.4f}{Cf_sum[i] / n:>12.4f}")

    # ---- Validation-mode AP ablation ----
    print("\n===== AP ablation (mask scale -> zero) =====")
    base, abl = run_ap_ablation(model, args, scale_idx)
    print(f"{'scale':<14}{'mAP50-95':>12}{'mAP50':>10}{'dAP':>10}")
    print(f"{'baseline':<14}{base[0]:>12.4f}{base[1]:>10.4f}{'-':>10}")
    dap = {}
    for k, i in enumerate(scale_idx):
        d = base[0] - abl[i][0]
        dap[i] = d
        print(f"{labels[k]:<14}{abl[i][0]:>12.4f}{abl[i][1]:>10.4f}{d:>10.4f}")

    # ---- Save CSV ----
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "contribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale", "label", "E_i", "G_i_rel", "C_loss", "mAP50-95_full", "mAP50-95_masked", "dAP"])
        for k, i in enumerate(scale_idx):
            w.writerow([i, labels[k], E_sum[i] / n, G_sum[i] / max(g_total, 1e-9), Cf_sum[i] / n, base[0], abl[i][0], dap[i]])
    if args.per_image:
        per_image_skippability(model, args, scale_idx, args.per_image, args.conf, args.iou)
    print(f"\nsaved -> {out_dir / 'contribution.csv'}")


if __name__ == "__main__":
    main()

