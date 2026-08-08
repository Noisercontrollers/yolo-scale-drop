# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Train a learned spatial-budget gate for the YOLO26 Detect head and evaluate the budget curve.

The gate is a per-scale 1x1 conv predicting per-cell importance, supervised by GT-box cells
(pos-weighted BCE), while backbone/neck/head are frozen. At inference the top-budget cells
are kept and the rest suppressed in the head.

Usage:
  python tools/spatial_budget_gate.py --model runs/detect/100-MuSGD/weights/best.pt \
      --data VOC/VOC.yaml --device 0 --epochs 5 --budgets 0.05,0.10,0.25,0.50,0.75
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from feature_contribution import build_train_batches  # noqa

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


class BudgetGate(nn.Module):
    """Per-scale 1x1 conv gate: feature map -> per-cell importance logits."""

    def __init__(self, ch):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in ch)

    def forward(self, feats):
        return [self.convs[i](f) for i, f in enumerate(feats)]


def gt_cell_labels(batch, grids, device, bs):
    """Return per-scale positive masks (B,1,H,W) for cells whose center lies inside a GT box."""
    labels = []
    cls, bboxes, batch_idx = batch["cls"], batch["bboxes"], batch["batch_idx"]
    for (H, W) in grids:
        pos = torch.zeros(bs, 1, H, W, device=device, dtype=torch.bool)
        if len(bboxes):
            cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            for j in range(bs):
                m = batch_idx == j
                if not m.any():
                    continue
                cxc, cyc = (cx[m] * W), (cy[m] * H)  # center in grid coords
                hw, hh = w[m] * W / 2, h[m] * H / 2
                x0, x1 = (cxc - hw).floor().clamp(0, W - 1).long(), (cxc + hw).ceil().clamp(0, W).long()
                y0, y1 = (cyc - hh).floor().clamp(0, H - 1).long(), (cyc + hh).ceil().clamp(0, H).long()
                for k in range(len(x0)):
                    pos[j, 0, y0[k]:y1[k], x0[k]:x1[k]] = True
        labels.append(pos)
    return labels


def train_gate(model, loader, epochs, device, args):
    feats = {}

    def capture(i):
        def hook(_m, _i, out):
            o = out if isinstance(out, torch.Tensor) else out[0]
            feats[i] = o

        return hook

    seq = model.model.model
    scale_idx = [int(f) for f in seq[-1].f]
    handles = [seq[i].register_forward_hook(capture(i)) for i in scale_idx]

    def current_feats():
        return [feats[i] for i in scale_idx]

    model.model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        model.model((batch["img"].float() / 255).to(device))
    ch = [f.shape[1] for f in current_feats()]

    gate = BudgetGate(ch).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    gate.train()
    step = 0
    try:
        for epoch in range(epochs):
            total = 0.0
            n = 0
            for batch in loader:
                img = (batch["img"].float() / 255).to(device)
                with torch.no_grad():
                    out = model.model(img)  # (y, preds); feats captured by hooks
                fl = current_feats()
                grids = [(f.shape[2], f.shape[3]) for f in fl]
                bs = img.shape[0]
                loss = torch.tensor(0.0, device=device)
                for i, f in enumerate(fl):
                    logits = gate.convs[i](f)  # (B,1,H,W)
                    if args.label == "gt":
                        labels = gt_cell_labels(batch, grids, device, bs)
                        pos = labels[i].float()
                        n_pos = pos.sum()
                        n_neg = pos.numel() - n_pos
                        pw = (n_neg / max(n_pos, 1)).clamp(max=50.0)
                        bce = nn.functional.binary_cross_entropy_with_logits(
                            logits, pos, pos_weight=torch.tensor([pw], device=device)
                        )
                    else:  # score distillation: target = max-class sigmoid of teacher one2one scores
                        scores = out[1]["one2one"]["scores"]  # (B, nc, N)
                        H, W = grids[i]
                        n_c = H * W
                        offset = sum(h * w for h, w in grids[:i])
                        t = scores[:, :, offset:offset + n_c].max(dim=1).values.sigmoid().view(bs, 1, H, W)
                        w = 1 + 30 * t  # importance-weighted soft BCE
                        bce = nn.functional.binary_cross_entropy_with_logits(logits, t, weight=w, reduction="mean")
                    loss = loss + bce
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
                n += 1
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}", flush=True)
            print(f"epoch {epoch + 1}/{epochs} avg_loss {total / max(n, 1):.4f}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return gate


def apply_gate_mask(x, gate_fn, budget):
    feats = x["feats"]
    scores = x["scores"]
    boxes = x["boxes"]
    bs = scores.shape[0]
    logits = gate_fn(feats)
    masks = []
    offset = 0
    for fi, lo in zip(feats, logits):
        B, C, H, W = fi.shape
        n = H * W
        cell = torch.sigmoid(lo.detach()).view(B, -1)
        k = max(int(n * budget), 1)
        m = torch.zeros_like(cell, dtype=torch.bool)
        m.scatter_(1, torch.topk(cell, k, dim=1).indices, True)
        masks.append(m)
        offset += n
    mask = torch.cat(masks, dim=1).unsqueeze(1)
    scores.masked_fill_(~mask, float("-inf"))
    boxes *= mask
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label", default="score", choices=["gt", "score"], help="gt=GT-cell BCE; score=distill teacher head scores")
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50,0.75")
    ap.add_argument("--project", default="runs/spatial_budget")
    ap.add_argument("--name", default="voc_gate")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    torch.backends.cudnn.benchmark = True

    model = YOLO(args.model)
    model.to(device)
    loader = build_train_batches(args)
    gate = train_gate(model, loader, args.epochs, device, args)

    detect = model.model.model[-1]
    orig_inference = detect._inference
    rows = []
    for budget in [float(b) for b in args.budgets.split(",")]:
        def inference(x, b=budget):
            apply_gate_mask(x, lambda fs: gate(fs), b)
            return orig_inference(x)

        detect._inference = inference
        try:
            res = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device,
                            workers=args.workers, plots=False, split="val", fraction=1.0,
                            project=args.project, name=args.name, exist_ok=True, verbose=False)
        finally:
            detect._inference = orig_inference
        rows.append((budget, float(res.box.map), float(res.box.map50)))
        print(f"gate=learned budget={budget:<6.2f} mAP50-95={rows[-1][1]:.4f} mAP50={rows[-1][2]:.4f}", flush=True)

    out_dir = Path(args.project) / f"{args.name}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gate.state_dict(), out_dir / "gate.pt")
    with open(out_dir / "budget_curve_gate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gate", "budget", "mAP50-95", "mAP50"])
        for b, m, m50 in rows:
            w.writerow(["learned", b, m, m50])
    print("saved ->", out_dir / "gate.pt", out_dir / "budget_curve_gate.csv")


if __name__ == "__main__":
    main()
