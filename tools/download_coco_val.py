# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Download COCO labels + val2017 images only (analysis; models use official COCO-pretrained weights)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


def main():
    root = Path("coco")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    print("downloading labels...", flush=True)
    download([ASSETS_URL + "/coco2017labels.zip"], dir=root, unzip=True, delete=True)
    print("labels done", flush=True)
    print("downloading val2017 images...", flush=True)
    download([VAL_URL], dir=root / "images", unzip=True, delete=True, threads=3)
    print("val images done", flush=True)
    print("ALL DOWNLOADS DONE", flush=True)


if __name__ == "__main__":
    main()