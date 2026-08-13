# YOLO Scale Drop

**When Is a Scale Redundant? Data-Dependent Scale Redundancy and Zero-Training Pruning for Real-Time Object Detection** 

An empirical study of when a feature-pyramid scale in a YOLO detector is redundant, why it depends on the data, and how it can be removed with **zero training**.

## Key findings

1. **Scale redundancy is dataset-dependent.** On YOLO26n, removing the small-object scale P3 costs 0.010 / 0.032 / 0.101 mAP on VOC / COCO / VisDrone, while removing the large-object scale P5 costs 0.417 / 0.176 / ~0. The per-scale gradient contribution tracks the small-object fraction, and the P3/P5 ranking reverses between 3.8% and 24.8% small objects (a three-point trend, not a law; crossover near 12.6%).
2. **FLOPs ≠ FPS on GPU.** Cell-level redundancy (~95% of grid cells) is only exploitable as a *mask-level oracle*: real block skipping neither speeds up GPU inference nor preserves accuracy without re-training.
3. **Zero-training scale drop.** Where a scale is truly redundant, dropping it needs no training: on VisDrone, P5 removal cuts FLOPs / parameters by 10.4% / 28.6% at identical mAP and reduces single-image latency by 6%. On dense small-object data it beats input-resolution scaling (0.173 vs 0.145 mAP).

Also reports negative results: backbone early exit is infeasible, and fine-tuning a dropped scale can hurt.

## Repository layout

```
yolo-scale-drop/
├── paper/                 # (private) PRL submission files — kept in private repo yolo-scale-drop-paper
├── tools/                 # experiment scripts (contribution analysis, spatial-budget gate/skip, scale-drop, ...)
├── configs/               # custom YOLO YAMLs (p34 = P3/P4-only variant, p5drop-visdrone, visdrone, coco_local)
├── results/               # key measured CSVs (FPS/latency, resolution baselines, YOLO11n cross-check, ...)
├── yolo26_train.py        # train entry (see ultralytics YOLO26)
├── yolo26_predict.py      # inference entry
└── LICENSE                # AGPL-3.0 (derived from Ultralytics)
```

## Datasets

- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (test2007)
- [MS COCO](https://cocodataset.org/) (val2017)
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)

## Requirements

- Python >= 3.8, PyTorch >= 1.8
- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) (AGPL-3.0); scripts import the `ultralytics` package, so install the repo locally or `pip install ultralytics`.
- CUDA GPU for latency/FPS measurements; all reported numbers are FP32 on an NVIDIA RTX 5070 (CUDA 13.2, no TensorRT).

## Reproduce

1. Clone Ultralytics YOLO26 and place this repo alongside it (or install ultralytics).
2. Download the datasets and point the YAML configs in `configs/` at your local paths.
3. Train with the standard YOLO26n recipe, then run, e.g.:

```bash
# per-scale contribution analysis (gradient share + mAP cost of scale removal)
python tools/feature_contribution.py --yaml configs/yolo26n-p34.yaml --data VOC/VOC.yaml --device 0

# spatial-budget oracle / energy / self-distilled gate (mask-level)
python tools/spatial_budget_gate.py --data VOC/VOC.yaml --device 0 --epochs 5 --budgets 0.05,0.10,0.25,0.50,0.75

# zero-training P5-drop on VisDrone
python tools/build_p5drop_vd.py
python tools/drop_p3.py --yaml configs/yolo26n-p34.yaml --data VOC/VOC.yaml --device 0
```

Note: scripts were developed against a local Ultralytics checkout; adjust `REPO` paths / dataset YAML paths to your environment.

## Citation

```bibtex
@article{yan2026scale,
  title={When Is a Scale Redundant? Data-Dependent Scale Redundancy and Zero-Training Pruning for Real-Time Object Detection},
  author={Yan, Yuanchao and Yan, Xin and Liu, Jianlong},
  journal={Pattern Recognition Letters},
  year={2026}
}
```

## License

AGPL-3.0. This project is derived from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
