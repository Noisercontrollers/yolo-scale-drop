# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Build the YOLO26n P3-drop variant, transfer weights from a trained 3-scale checkpoint, and measure mAP/FLOPs/FPS.

Usage:
    python tools/drop_p3.py --src runs/detect/100-MuSGD/weights/best.pt \
        --yaml ultralytics/cfg/models/26/yolo26n-p34.yaml --data VOC/VOC.yaml --device 0
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

sys.path.insert(0, str(Path(__file__).parent))
from feature_contribution import _MaskBoxes, _MaskScores  # noqa


def transfer(src: YOLO, dst: YOLO):
    """Transfer shared weights from src (3-scale) to dst (P3-drop). mapping: dst_key -> src_key."""
    s = src.model.state_dict()
    d = dst.model.state_dict()
    mapping = {}
    for i in range(14):  # layers 0..13 identical
        for k in list(d):
            if k.startswith(f"model.{i}."):
                mapping[k] = k
    # P5 chain: src 20->14, 22->16 (Concat has no params)
    for src_i, dst_i in ((20, 14), (22, 16)):
        for k in list(s):
            if k.startswith(f"model.{src_i}."):
                mapping[k.replace(f"model.{src_i}.", f"model.{dst_i}.")] = k
    # Detect head: keep P4 (src idx1) and P5 (src idx2) -> dst idx0, idx1
    for k in list(s):
        parts = k.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "23" and parts[2] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
            idx = int(parts[3])
            if idx in (1, 2):
                new_idx = idx - 1
                mapping[f"model.17.{parts[2]}.{new_idx}." + ".".join(parts[4:])] = k

    out = {}
    skipped = []
    for k in d:
        if k in mapping:
            sk = mapping[k]
            if s[sk].shape == d[k].shape:
                out[k] = s[sk]
            else:
                skipped.append((k, tuple(s[sk].shape), tuple(d[k].shape)))
        elif k not in s or s[k].shape != d[k].shape:
            skipped.append((k, tuple(s[k].shape), tuple(d[k].shape)) if k in s else (k, None, tuple(d[k].shape)))
        else:
            out[k] = s[k]
    miss, unexp = dst.model.load_state_dict(out, strict=False)
    print(f"transferred {len(out)}/{len(d)} tensors; skipped={len(skipped)} (shape mismatch, head init) unexpected={len(unexp)}")
    if skipped:
        print("skipped examples:", skipped[:3])
    if miss:
        print("missing:", miss[:10])
    return dst


def bench_fps(model, device, batch=32, imgsz=640, warmup=10, iters=60):
    """Benchmark fused eval-mode inference FPS on GPU."""
    m = model.model.to(device).eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return batch / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--yaml", default="ultralytics/cfg/models/26/yolo26n-p34.yaml")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs/drop_p3")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- baseline ----
    base = YOLO(args.src)
    base.to(device)

    # ---- P3-drop variant with transferred weights ----
    p34 = YOLO(args.yaml)
    p34.to(device)
    transfer(base, p34)
    p34.save(out_dir / "yolo26n-p34-voc.pt")
    print("saved ->", out_dir / "yolo26n-p34-voc.pt")

    # ---- metrics ----
    rows = []

    def measure(name, model, mask_p3_head=False):
        detect = model.model.model[-1]
        saved = None
        if mask_p3_head:
            saved = (detect.one2one_cv2[0], detect.one2one_cv3[0])
            detect.one2one_cv2[0] = _MaskBoxes(detect.reg_max)
            detect.one2one_cv3[0] = _MaskScores(detect.nc)
        try:
            res = model.val(
                data=args.data, imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers,
                plots=False, split="val", fraction=1.0, project=out_dir, name=name.replace("(", "").replace(")", "").replace("/", "_").replace("-", "_"), exist_ok=True, verbose=False,
            )
        finally:
            if saved:
                detect.one2one_cv2[0] = saved[0]
                detect.one2one_cv3[0] = saved[1]
        n_p = sum(p.numel() for p in model.model.parameters())
        if mask_p3_head:  # restore mask for flops/fps measurement (head suppressed)
            detect.one2one_cv2[0] = _MaskBoxes(detect.reg_max)
            detect.one2one_cv3[0] = _MaskScores(detect.nc)
            n_p = sum(p.numel() for p in model.model.parameters())
        flops = get_flops(model.model, args.imgsz)
        fps = bench_fps(model, device, batch=args.batch, imgsz=args.imgsz)
        if saved:
            detect.one2one_cv2[0] = saved[0]
            detect.one2one_cv3[0] = saved[1]
        row = (name, float(res.box.map), float(res.box.map50), flops, n_p / 1e6, fps)
        rows.append(row)
        print(f"{name:<26} mAP50-95={row[1]:.4f} mAP50={row[2]:.4f} GFLOPs={row[3]:.3f} Params(M)={row[4]:.3f} FPS={row[5]:.1f}")

    measure("baseline(yolo26n)", base)
    measure("P3-drop head-only", base, mask_p3_head=True)
    measure("P3-drop full(p34)", p34)

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mAP50-95", "mAP50", "GFLOPs", "Params(M)", "FPS"])
        w.writerows(rows)
    print("saved ->", out_dir / "results.csv")


if __name__ == "__main__":
    main()
