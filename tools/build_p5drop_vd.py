# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Build P5-drop (2-scale P3/P4) VisDrone model, transfer weights from the 3-scale VisDrone model, evaluate."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def transfer(src: YOLO, dst: YOLO):
    s = src.model.state_dict()
    d = dst.model.state_dict()
    mapping = {}
    for i in range(20):  # layers 0..19 identical
        for k in d:
            if k.startswith(f"model.{i}."):
                mapping[k] = k
    # Detect: src 23 -> dst 20, keep head idx0(P3)->0, idx1(P4)->1
    for k in s:
        parts = k.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "23" and parts[2] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
            idx = int(parts[3])
            if idx in (0, 1):
                mapping[f"model.20.{parts[2]}.{idx}." + ".".join(parts[4:])] = k
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
            skipped.append((k, None, tuple(d[k].shape)))
        else:
            out[k] = s[k]
    miss, unexp = dst.model.load_state_dict(out, strict=False)
    print(f"transferred {len(out)}/{len(d)}; skipped={len(skipped)} unexpected={len(unexp)}")
    if skipped:
        print("skipped:", skipped[:5])
    if miss:
        print("missing:", miss[:5])
    return dst


def bench_fps(mm, device, batch=32, imgsz=640, warmup=15, iters=60):
    mm = mm.eval()
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
    src = YOLO("runs/detect/train-3/weights/best.pt")
    src.to("cuda:0")
    dst = YOLO("ultralytics/cfg/models/26/yolo26n-p5drop-visdrone.yaml")
    dst.to("cuda:0")
    transfer(src, dst)
    dst.save("runs/visdrone/yolo26n-p5drop-vd.pt")
    print("saved runs/visdrone/yolo26n-p5drop-vd.pt")

    # eval both
    torch.backends.cudnn.benchmark = True
    for name, m in (("3-scale baseline", src), ("P5-drop", dst)):
        res = m.val(data="VisDrone/VisDrone.yaml", imgsz=640, batch=32, device="cuda:0", workers=0, plots=False,
                    split="val", fraction=1.0, project="runs/visdrone", name=("b3" if name.startswith("3") else "p5drop"), exist_ok=True, verbose=False)
        flops = get_flops(m.model, 640)
        n_p = sum(p.numel() for p in m.model.parameters())
        fps = bench_fps(m.model, "cuda:0")
        print(f"{name:<18} mAP50-95={float(res.box.map):.4f} mAP50={float(res.box.map50):.4f} GFLOPs={flops:.3f} Params(M)={n_p/1e6:.3f} FPS={fps:.0f}")


if __name__ == "__main__":
    main()
