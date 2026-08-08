# -*- coding: utf-8 -*-
from pathlib import Path
"""Extra baselines: VisDrone resolution scaling (vs P5-drop), for revision."""
import sys, time, csv, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from ultralytics import YOLO

REPO = str(Path(__file__).resolve().parent.parent)
DEVICE = "cuda:0"
OUT = os.path.join(REPO, "runs", "revision")
os.makedirs(OUT, exist_ok=True)

def bench_fps(model, batch, imgsz=640, fp16=False, warmup=12, iters=50):
    m = model.eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=DEVICE)
    if fp16:
        m = m.half(); x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        best_t = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(iters):
                m(x)
            torch.cuda.synchronize()
            best_t = min(best_t, (time.perf_counter() - t0) / iters)
    return batch / best_t, best_t * 1000

def main():
    vd_yaml = os.path.join(REPO, "VisDrone", "VisDrone.yaml")
    full = YOLO(os.path.join(REPO, "runs", "detect", "train-3", "weights", "best.pt"))
    full.to(DEVICE)
    rows = []
    for imgsz in (480, 512, 640):
        met = full.val(data=vd_yaml, imgsz=imgsz, batch=32, device=DEVICE, workers=0, plots=False, verbose=False)
        fps, ms = bench_fps(full, 32, imgsz=imgsz)
        fps1, ms1 = bench_fps(full, 1, imgsz=imgsz)
        print(f"[vd-res] imgsz={imgsz} mAP50-95={met.box.map:.4f} fps32={fps:.1f} fps1={fps1:.1f}", flush=True)
        rows.append(dict(model="vd_full", imgsz=imgsz, mAP50_95=round(float(met.box.map), 4), fps32=round(fps, 1), fps1=round(fps1, 1)))
    p5 = YOLO(os.path.join(REPO, "runs", "visdrone", "yolo26n-p5drop-vd.pt"))
    p5.to(DEVICE)
    met = p5.val(data=vd_yaml, imgsz=640, batch=32, device=DEVICE, workers=0, plots=False, verbose=False)
    fps, ms = bench_fps(p5, 32)
    fps1, ms1 = bench_fps(p5, 1)
    print(f"[vd-p5drop] imgsz=640 mAP50-95={met.box.map:.4f} fps32={fps:.1f} fps1={fps1:.1f}", flush=True)
    rows.append(dict(model="vd_p5drop", imgsz=640, mAP50_95=round(float(met.box.map), 4), fps32=round(fps, 1), fps1=round(fps1, 1)))
    with open(os.path.join(OUT, "visdrone_resolution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "imgsz", "mAP50_95", "fps32", "fps1"])
        w.writeheader(); w.writerows(rows)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
