# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Simulate Dynamic Backbone Exit: measure mAP when only early-stage features feed the Detect head.

exit=P3     -> mask P4+P5 heads  (exit after backbone Block2, stride 8)
exit=P3P4   -> mask P5 head      (exit after backbone Block3, stride 16)
exit=full   -> no mask           (final head, stride 32)

Usage: python tools/exit_feasibility.py [--device 0] [--batch 32]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from feature_contribution import _MaskBoxes, _MaskScores, get_scale_indices  # noqa

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--data", default="VOC/VOC.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", default="runs/exit_feasibility")
    ap.add_argument("--name", default="voc")
    args = ap.parse_args()

    device = args.device if args.device.startswith(("cuda", "cpu")) else f"cuda:{args.device}"
    args.device = device
    torch.backends.cudnn.benchmark = True

    model = YOLO(args.model)
    model.to(device)
    seq = model.model.model
    detect = seq[-1]
    scale_idx = get_scale_indices(model)  # [16, 19, 22] -> pos 0,1,2

    configs = {
        "full": (),
        "exit_P3_only": (1, 2),   # keep only P3
        "exit_P3P4": (2,),        # keep P3+P4, mask P5
    }
    print(f"{'config':<16}{'mAP50-95':>12}{'mAP50':>10}")
    for name, mask_pos in configs.items():
        saved_o2o = [(detect.one2one_cv2[i], detect.one2one_cv3[i]) for i in range(detect.nl)]
        if mask_pos:
            detect.one2one_cv2 = torch.nn.ModuleList(
                [_MaskBoxes(detect.reg_max) if i in mask_pos else detect.one2one_cv2[i] for i in range(detect.nl)]
            )
            detect.one2one_cv3 = torch.nn.ModuleList(
                [_MaskScores(detect.nc) if i in mask_pos else detect.one2one_cv3[i] for i in range(detect.nl)]
            )
        try:
            res = model.val(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=False,
                split="val",
                fraction=1.0,
                project=args.project,
                name=args.name,
                exist_ok=True,
                verbose=False,
            )
        finally:
            detect.one2one_cv2 = torch.nn.ModuleList([s[0] for s in saved_o2o])
            detect.one2one_cv3 = torch.nn.ModuleList([s[1] for s in saved_o2o])
        print(f"{name:<16}{float(res.box.map):>12.4f}{float(res.box.map50):>10.4f}")


if __name__ == "__main__":
    main()
