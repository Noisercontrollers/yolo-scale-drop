# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Difficulty-adaptive routed inference: run the small model first, fall back to the full
model only when the small model is not confident (max confidence < threshold).

This is the deployable form of "difficulty-adaptive model routing" (cascade / draft-verify):
  - easy images  -> small (fast, p34, +28% FPS)
  - hard images  -> small + full fallback (accurate)

CLI:
  python tools/route_predict.py --source ultralytics/assets/bus.jpg --show
  python tools/route_predict.py --source VOC/VOCdevkit/images/test2007 --save

Python:
  from route_predict import RoutedDetector
  det = RoutedDetector(threshold=0.3, device="0")
  results = det.predict("bus.jpg")       # list[Results], same API as YOLO.predict
  det.stats()                            # routed-image statistics
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO


class RoutedDetector:
    """Runs the small model first and falls back to the full model on low-confidence images."""

    def __init__(self, full="runs/detect/100-MuSGD/weights/best.pt",
                 small="runs/detect/train/weights/best.pt", threshold=0.3,
                 device=None, **default_kwargs):
        self.full = YOLO(full)
        self.small = YOLO(small)
        self.threshold = threshold
        self.default_kwargs = default_kwargs
        self.n_small = 0
        self.n_full = 0
        if isinstance(device, str) and not device.startswith(("cuda", "cpu")):
            device = f"cuda:{device}"
        if device is not None:
            self.full.to(device)
            self.small.to(device)
        self.full.model.fuse()
        self.small.model.fuse()
        self.full.model.eval()
        self.small.model.eval()

    @staticmethod
    def _max_conf(result):
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0
        return float(result.boxes.conf.max().item())

    def predict(self, source, verbose=False, **kwargs):
        """Run routed inference on a source (path / list / ndarray / tensor); returns list[Results]."""
        kw = {**self.default_kwargs, **kwargs}
        small_results = self.small.predict(source, verbose=False, **kw)
        paths = [r.path for r in small_results]
        fallback = [i for i, r in enumerate(small_results) if self._max_conf(r) < self.threshold]
        self.n_small += len(small_results) - len(fallback)
        self.n_full += len(fallback)
        if verbose:
            print(f"routed: {len(small_results) - len(fallback)} small + {len(fallback)} full "
                  f"(fallback {len(fallback) / max(len(small_results), 1):.1%})")
        if not fallback:
            return small_results
        fb_src = self._fallback_source(source, paths, fallback)
        fb_results = self.full.predict(fb_src, verbose=False, **kw)
        merged = dict(zip(fallback, fb_results))
        return [merged.get(i, small_results[i]) for i in range(len(small_results))]

    @staticmethod
    def _fallback_source(source, paths, fallback):
        """Build the fallback sub-source from the original source and result paths."""
        if isinstance(source, (list, tuple)):
            return [source[i] for i in fallback]
        if isinstance(source, (torch.Tensor, np.ndarray)):
            return source[fallback] if len(fallback) > 1 else source[fallback[0]].unsqueeze(0) if isinstance(source, torch.Tensor) else source[fallback[0]][None]
        # str path: single file or directory; Results.path is reliable for these
        return [paths[i] for i in fallback]

    def stats(self):
        """Return (n_small, n_full) routed-image counts."""
        return self.n_small, self.n_full

    def save(self, results, out_dir="runs/route_predict", prefix="routed"):
        """Save results to out_dir as images."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, r in enumerate(results):
            fn = out / f"{prefix}_{Path(r.path).stem}_{i}.jpg"
            r.save(filename=str(fn))
            saved.append(str(fn))
        return saved


def main():
    ap = argparse.ArgumentParser(description="Routed inference: small model first, full fallback.")
    ap.add_argument("--source", required=True, help="image path / directory / list")
    ap.add_argument("--full", default="runs/detect/100-MuSGD/weights/best.pt")
    ap.add_argument("--small", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--threshold", type=float, default=0.3, help="small-model max-conf fallback threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--show", action="store_true", help="display results (GUI)")
    ap.add_argument("--save", action="store_true", help="save result images to runs/route_predict")
    args = ap.parse_args()

    det = RoutedDetector(full=args.full, small=args.small, threshold=args.threshold, device=args.device)
    results = det.predict(args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=True)
    print(f"got {len(results)} result(s)")
    for i, r in enumerate(results[:5]):
        nb = 0 if r.boxes is None else len(r.boxes)
        mc = 0.0 if nb == 0 else float(r.boxes.conf.max())
        print(f"  [{i}] {Path(r.path).name}: {nb} boxes, max_conf={mc:.3f}")
    n_s, n_f = det.stats()
    total = n_s + n_f
    if total:
        print(f"route stats: small={n_s} full={n_f} fallback_rate={n_f / total:.1%}")
    if args.save:
        files = det.save(results)
        print("saved:", files[:5], "...")
    if args.show:
        for r in results:
            r.show()


if __name__ == "__main__":
    main()
