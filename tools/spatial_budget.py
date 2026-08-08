# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Spatial Token Budget for YOLO26 Detect head.

Masks (suppresses) grid cells in the Detect head below a per-image budget and measures mAP.
Gate modes:
  energy : keep cells with highest per-cell feature energy  mean(|F_i|)   (train-free)
  score  : keep cells with highest model head scores (pre-NMS cls max)   (self-oracle upper bound)

Usage:
  python tools/spatial_budget.py --model runs/detect/100-MuSGD/weights/best.pt \
      --data VOC/VOC.yaml --device 0 --budgets 0.05,0.1,0.25,0.5,0.75,1.0
"""
import argparse
import csv
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def apply_cell_mask(x: dict, gate: str, budget: float):
    """Zero boxes and set -inf scores for pruned cells; returns modified dict in place."""
    feats = x["feats"]
    scores = x["scores"]  # (B, nc, N)
    boxes = x["boxes"]  # (B, 4*reg_max, N)
    bs = scores.shape[0]
    masks = []
    offset = 0
    for fi in feats:
        B, C, H, W = fi.shape
        n = H * W
        if gate == "energy":
            cell = fi.detach().abs().mean(dim=1).view(B, -1)  # (B, n)
        elif gate == "score":
            s = scores[:, :, offset:offset + n]  # (B, nc, n)
            cell = s.max(dim=1).values  # (B, n)
        else:
            raise ValueError(gate)
        k = max(int(n * budget), 1)
        m = torch.zeros_like(cell, dtype=torch.bool)
        topk = torch.topk(cell, k, dim=1).indices
        m.scatter_(1, topk, True)
        masks.append(m)
        offset += n
    mask = torch.cat(masks, dim=1).unsqueeze(1)  # (B, 1, N)
    scores.masked_fill_(~mask, float("-inf"))
    boxes *= mask
    return x


def patch_head(model, gate: str, budget: float):
    detect = model.model.model[-1]
    orig = detect._inference

    def inference(x):
        apply_cell_mask(x, gate, budget)
        return orig(x)

    detect._inference = inference
    return detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50,0.75,1.00")
    ap.add_argument("--project", default="runs/spatial_budget")
    ap.add_argument("--name", default="voc")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    torch.backends.cudnn.benchmark = True
    budgets = [float(b) for b in args.budgets.split(",")]

    model = YOLO(args.model)
    model.to(device)
    detect = model.model.model[-1]
    orig_inference = detect._inference  # bound class method, survives fusion
    HEAD_GFLOPS = 0.902  # measured Detect-head FLOPs for yolo26n at 640

    rows = []
    for gate in ("score", "energy"):
        for budget in budgets:
            def inference(x, g=gate, b=budget, orig=orig_inference):
                apply_cell_mask(x, g, b)
                return orig(x)

            detect._inference = inference
            try:
                res = model.val(
                    data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
                    workers=args.workers, plots=False, split="val", fraction=1.0,
                    project=args.project, name=args.name, exist_ok=True, verbose=False,
                )
            finally:
                detect._inference = orig_inference
            map50_95 = float(res.box.map)
            map50 = float(res.box.map50)
            rows.append((gate, budget, map50_95, map50))
            print(f"gate={gate:<7} budget={budget:<6.2f} mAP50-95={map50_95:.4f} mAP50={map50:.4f}")

    # total FLOPs (fused) for reference
    model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
              plots=False, split="val", fraction=0.02, project=args.project, name=args.name, exist_ok=True, verbose=False)
    total_flops = get_flops(model.model, args.imgsz)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "budget_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gate", "budget", "mAP50-95", "mAP50", "head_flops_saved_G", "total_flops_G"])
        for gate, budget, m, m50 in rows:
            w.writerow([gate, budget, m, m50, round(HEAD_GFLOPS * (1 - budget), 3), round(total_flops, 3)])
    print("saved ->", out_dir / "budget_curve.csv")


if __name__ == "__main__":
    main()
