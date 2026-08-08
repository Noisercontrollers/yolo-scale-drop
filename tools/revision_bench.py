# -*- coding: utf-8 -*-
from pathlib import Path
"""Revision benchmarks: unified FPS/latency, resolution baseline, eval repeatability, YOLO11n COCO scale contribution."""
import sys, time, csv, os, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from ultralytics import YOLO

REPO = str(Path(__file__).resolve().parent.parent)
DEVICE = "cuda:0"
OUT = os.path.join(REPO, "runs", "revision")
os.makedirs(OUT, exist_ok=True)
LOG = lambda *a: print(*a, flush=True)

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
# ---- 1. unified FPS / latency ----
    models = [
        ("voc_full",   os.path.join(REPO, "runs", "detect", "100-MuSGD", "weights", "best.pt")),
        ("voc_p3drop", os.path.join(REPO, "runs", "drop_p3", "yolo26n-p34-voc.pt")),
        ("vd_full",    os.path.join(REPO, "runs", "detect", "train-3", "weights", "best.pt")),
        ("vd_p5drop",  os.path.join(REPO, "runs", "visdrone", "yolo26n-p5drop-vd.pt")),
    ]
    rows = []
    for name, path in models:
        LOG(f"[{name}] loading")
        m = YOLO(path)
        m.to(DEVICE)
        for fp16 in (False, True):
            for batch in (32, 1):
                try:
                    fps, ms = bench_fps(m, batch, fp16=fp16)
                    LOG(f"[{name}] fp16={fp16} batch={batch} fps={fps:.1f} ms={ms:.2f}")
                    rows.append(dict(model=name, fp16=fp16, batch=batch, imgsz=640, fps=round(fps, 1), ms=round(ms, 2)))
                except Exception as e:
                    LOG(f"[{name}] fp16={fp16} batch={batch} FAILED {e}")
    with open(os.path.join(OUT, "fps_latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fp16", "batch", "imgsz", "fps", "ms"])
        w.writeheader(); w.writerows(rows)
    LOG("fps_latency done")

    # ---- 2. resolution baseline (VOC full) ----
    m = YOLO(os.path.join(REPO, "runs", "detect", "100-MuSGD", "weights", "best.pt"))
    m.to(DEVICE)
    res = []
    for imgsz in (480, 512, 640):
        met = m.val(data=os.path.join(REPO, "VOC", "VOC.yaml"), imgsz=imgsz, batch=32, device=DEVICE, plots=False, verbose=False, workers=0)
        fps, _ = bench_fps(m, 32, imgsz=imgsz)
        LOG(f"[res] imgsz={imgsz} mAP50-95={met.box.map:.4f} fps={fps:.1f}")
        res.append(dict(imgsz=imgsz, mAP50_95=round(float(met.box.map), 4), fps=round(fps, 1)))
    with open(os.path.join(OUT, "resolution_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imgsz", "mAP50_95", "fps"])
        w.writeheader(); w.writerows(res)
    LOG("resolution done")

    # ---- 3. eval repeatability (VOC full, 3 runs) ----
    rep = []
    for k in range(3):
        met = m.val(data=os.path.join(REPO, "VOC", "VOC.yaml"), imgsz=640, batch=32, device=DEVICE, plots=False, verbose=False, workers=0)
        LOG(f"[rep] run{k+1} mAP50-95={met.box.map:.4f}")
        rep.append(dict(run=k + 1, mAP50_95=round(float(met.box.map), 4)))
    with open(os.path.join(OUT, "eval_repeatability.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "mAP50_95"])
        w.writeheader(); w.writerows(rep)
    LOG("repeatability done")

    # ---- 4. YOLO11n COCO scale contribution (2nd architecture) ----
    coco_yaml = os.path.join(OUT, "coco_local.yaml")
    builtin = os.path.join(REPO, "ultralytics", "cfg", "datasets", "coco.yaml")
    with open(builtin, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("path: coco", "path: " + os.path.join(REPO, "coco").replace("\\", "/"))
    with open(coco_yaml, "w", encoding="utf-8") as f:
        f.write(content)
    LOG("coco_local.yaml written")
    r = subprocess.run([
        sys.executable, os.path.join(REPO, "tools", "feature_contribution.py"),
        "--model", "yolo11n.pt", "--data", coco_yaml, "--fraction", "1.0", "--batch", "32",
        "--mask-mode", "head", "--project", "runs/revision", "--name", "yolo11n_coco",
        "--train-batches", "4", "--workers", "0",
    ], cwd=REPO, capture_output=True, text=True)
    with open(os.path.join(OUT, "yolo11n_coco_stdout.log"), "w", encoding="utf-8") as f:
        f.write(r.stdout)
    with open(os.path.join(OUT, "yolo11n_coco_stderr.log"), "w", encoding="utf-8") as f:
        f.write(r.stderr)
    LOG("yolo11n contribution exit:", r.returncode)
    LOG("ALL DONE")


if __name__ == "__main__":
    main()
