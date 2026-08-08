# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Real block-skip spatial token budget for the YOLO26 Detect head.

Keeps top-k blocks per scale (by the distilled gate), runs the head convs only on kept
blocks (with a 1-pixel halo so results match the full convolution exactly), scatters back,
then measures REAL mAP + FPS at each budget.

Usage:
  python tools/spatial_budget_skip.py --model runs/detect/100-MuSGD/weights/best.pt \
      --gate runs/spatial_budget/voc_gate_score/gate.pt --data VOC/VOC.yaml \
      --device 0 --budgets 0.05,0.10,0.25,0.50,1.00
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


class BudgetGate(nn.Module):
    """Per-scale 1x1 conv gate (must match spatial_budget_gate.BudgetGate)."""

    def __init__(self, ch):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in ch)

    def forward(self, feats):
        return [self.convs[i](f) for i, f in enumerate(feats)]


def block_size(H):
    """Larger blocks cut halo/scatter overhead: P3(80)->16, P4(40)->8, P5(20)->4."""
    if H % 16 == 0:
        return (16, 16)
    if H % 8 == 0:
        return (8, 8)
    return (4, 4)


def budget_forward_head(x, box_head, cls_head, gate, budget, reg_max, nc):
    """Compute head boxes/scores only on top-budget blocks; return full-grid tensors."""
    bs = x[0].shape[0]
    boxes_parts, scores_parts = [], []
    for i, xi in enumerate(x):
        B, C, H, W = xi.shape
        n = H * W
        bh, bw = block_size(H)
        nb_h, nb_w = H // bh, W // bw
        nb = nb_h * nb_w
        # block importance (avg gate score per block)
        s = torch.sigmoid(gate.convs[i](xi.detach()))  # (B,1,H,W)
        s_blocks = F.avg_pool2d(s, (bh, bw)).view(B, -1)  # (B, nb)
        k = min(max(int(nb * budget), 1), nb)
        topk = torch.topk(s_blocks, k, dim=1).indices  # (B, k)
        # halo-pad then unfold into (bh+2, bw+2) patches
        xi_pad = F.pad(xi.detach(), (1, 1, 1, 1), mode="constant", value=0)  # (B,C,H+2,W+2)
        patches = F.unfold(xi_pad, kernel_size=(bh + 2, bw + 2), stride=(bh, bw))  # (B, C*(bh+2)*(bw+2), nb)
        kept = torch.gather(patches, 2, topk.unsqueeze(1).expand(B, patches.shape[1], k))  # (B, C*ph*pw, k)
        block_imgs = kept.view(B, C, bh + 2, bw + 2, k).permute(0, 4, 1, 2, 3).reshape(B * k, C, bh + 2, bw + 2)
        b_out = box_head[i](block_imgs)  # (B*k, 4*reg_max, bh+2, bw+2)
        c_out = cls_head[i](block_imgs)  # (B*k, nc, bh+2, bw+2)
        b_out = b_out[..., 1:-1, 1:-1].reshape(B, k, 4 * reg_max, bh * bw).permute(0, 2, 1, 3).reshape(B, 4 * reg_max, k * bh * bw)
        c_out = c_out[..., 1:-1, 1:-1].reshape(B, k, nc, bh * bw).permute(0, 2, 1, 3).reshape(B, nc, k * bh * bw)
        # scatter back
        rows = torch.arange(nb_h, device=xi.device).view(-1, 1).repeat(1, nb_w).view(-1)
        cols = torch.arange(nb_w, device=xi.device).view(1, -1).repeat(nb_h, 1).view(-1)
        inner = torch.arange(bh * bw, device=xi.device)
        dr, dc = inner // bw, inner % bw
        block_cell = (rows.view(-1, 1) * bh * W + cols.view(-1, 1) * bw + (dr * W + dc).view(1, -1))  # (nb, bh*bw) cell indices
        cell_idx = block_cell[topk].view(B, -1)  # (B, k*bh*bw)
        full_boxes = torch.zeros(B, 4 * reg_max, n, device=xi.device)
        full_scores = torch.full((B, nc, n), float("-inf"), device=xi.device)
        full_boxes.scatter_(2, cell_idx.unsqueeze(1).expand(B, 4 * reg_max, k * bh * bw), b_out)
        full_scores.scatter_(2, cell_idx.unsqueeze(1).expand(B, nc, k * bh * bw), c_out)
        boxes_parts.append(full_boxes)
        scores_parts.append(full_scores)
    return dict(boxes=torch.cat(boxes_parts, dim=-1), scores=torch.cat(scores_parts, dim=-1), feats=x)


def make_patch(detect, gate, budget):
    orig = detect.forward_head

    def forward_head(x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:  # fused one2many path
            return dict()
        return budget_forward_head(x, box_head, cls_head, gate, budget, detect.reg_max, detect.nc)

    return forward_head, orig


def bench_fps(model, device, batch=32, imgsz=640, warmup=20, iters=80):
    mm = model.model.to(device).eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            mm(x)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(iters):
                mm(x)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / iters)
    return batch / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--gate", default="runs/spatial_budget/voc_gate_score/gate.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50,1.00")
    ap.add_argument("--project", default="runs/spatial_budget")
    ap.add_argument("--name", default="voc_skip")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    torch.backends.cudnn.benchmark = True

    model = YOLO(args.model)
    model.to(device)
    model.model.fuse()  # keep only one2one head path
    detect = model.model.model[-1]

    # gate channels: input channels of each one2one box head (fused model)
    ch = [m[0].conv.in_channels for m in detect.one2one_cv2]
    gate = BudgetGate(ch).to(device)
    gate.load_state_dict(torch.load(args.gate, map_location=device))
    gate.eval()

    # baseline
    fps0 = bench_fps(model, device, args.batch, args.imgsz)
    flops0 = get_flops(model.model, args.imgsz)
    res0 = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers,
                     plots=False, split="val", fraction=1.0, project=args.project, name=args.name, exist_ok=True, verbose=False)
    map0 = float(res0.box.map)
    print(f"baseline mAP50-95={map0:.4f} GFLOPs={flops0:.3f} FPS={fps0:.1f}", flush=True)

    rows = [("baseline", 1.0, map0, flops0, fps0)]
    for budget in [float(b) for b in args.budgets.split(",")]:
        patch, orig = make_patch(detect, gate, budget)
        detect.forward_head = patch
        try:
            res = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers,
                            plots=False, split="val", fraction=1.0, project=args.project, name=args.name, exist_ok=True, verbose=False)
            flops = get_flops(model.model, args.imgsz)
            fps = bench_fps(model, device, args.batch, args.imgsz)
        finally:
            detect.forward_head = orig
        rows.append((f"budget={budget:.2f}", budget, float(res.box.map), flops, fps))
        print(f"budget={budget:<6.2f} mAP50-95={rows[-1][2]:.4f} GFLOPs={flops:.3f} FPS={fps:.1f}", flush=True)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "skip_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "budget", "mAP50-95", "GFLOPs", "FPS"])
        w.writerows(rows)
    print("saved ->", out_dir / "skip_results.csv")


if __name__ == "__main__":
    main()
