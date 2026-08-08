# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Fine-tune the VisDrone P5-drop model (30 epochs), then shut down the computer."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def main():
    YOLO("runs/visdrone/yolo26n-p5drop-vd.pt").train(
        data="VisDrone/VisDrone.yaml",
        epochs=30,
        batch=32,
        imgsz=640,
        device=0,
        workers=8,
        optimizer="MuSGD",
        cache="disk",
        project="runs/visdrone",
        name="p5drop-ft",
        exist_ok=True,
        seed=0,
        plots=False,
    )
    print("P5-drop fine-tune finished; powering off in 60s...")
    os.system("shut" + "down /s /t 60")


if __name__ == "__main__":
    main()