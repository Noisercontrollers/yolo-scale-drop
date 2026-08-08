# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO26n baseline on VisDrone (run in background)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ensure repo ultralytics is used

from ultralytics import YOLO


def main():
    YOLO("ultralytics/cfg/models/26/yolo26n-visdrone.yaml").train(
    data="VisDrone.yaml",
    epochs=100,
    batch=64,
    imgsz=640,
    device=0,
    workers=8,
    optimizer="MuSGD",
    cache="disk",
    pretrained="weights/yolo26n.pt",
    cls_remap=True,
    project="runs/visdrone",
    name="baseline",
    exist_ok=True,
    seed=0,
    plots=False,
)


if __name__ == "__main__":
    main()
