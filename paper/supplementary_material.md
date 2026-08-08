# Supplementary Material
## When Is a Scale Redundant? Dataset-Dependent Spatial Redundancy and Token Budgeting for Real-Time Object Detection

### A. Full spatial-budget curves (mAP50-95) ? VOC test2007

| Budget | Oracle | Energy gate | GT gate | Distilled gate |
|--------|--------|-------------|---------|----------------|
| 5%  | 0.5895 | 0.1706 | 0.3747 | 0.5789 |
| 10% | 0.5896 | 0.2975 | 0.4400 | 0.5852 |
| 25% | 0.5896 | 0.4851 | 0.5252 | 0.5882 |
| 50% | 0.5896 | 0.5695 | 0.5718 | 0.5891 |
| 75% | 0.5896 | 0.5865 | 0.5858 | 0.5894 |
| 100%| 0.5896 | 0.5896 | ?      | 0.5896 |

### B. Full spatial-budget curves (mAP50-95) ? VisDrone val

| Budget | Energy gate | Distilled gate |
|--------|-------------|----------------|
| 5%  | 0.0745 | 0.1675 |
| 10% | 0.1093 | 0.1719 |
| 25% | 0.1479 | 0.1724 |
| 50% | 0.1677 | 0.1725 |
| 75% | 0.1717 | 0.1725 |
| 100%| 0.1725 | 0.1725 |

### C. Scale contribution on COCO for YOLO26n / 26s / 26m

Gradient contribution $G_i$ (normalized) and mAP50-95 cost $\Delta$AP of removing each scale (head-level fair mask).

| Model | $G_{P3}$ | $G_{P4}$ | $G_{P5}$ | $\Delta$AP P3 | $\Delta$AP P4 | $\Delta$AP P5 |
|-------|----------|----------|----------|--------------|--------------|--------------|
| YOLO26n | 28.3% | 41.3% | 30.3% | 0.032 | 0.110 | 0.176 |
| YOLO26s | 33.2% | 40.2% | 26.6% | 0.045 | 0.156 | 0.216 |
| YOLO26m | 31.6% | 30.8% | 37.6% | 0.078 | 0.179 | 0.217 |

In all three model sizes the small-object scale P3 is the least important and the large-object scale P5 is the most important, so the scale-redundancy ranking reported in the paper is not a nano-specific artifact.

### D. Backbone early-exit feasibility (VOC test2007, mAP50-95)

| Exit point | mAP50-95 | Relative to full |
|------------|----------|------------------|
| P3 only  | 0.022 | -97% |
| P3 + P4  | 0.159 | -73% |
| Full (P3+P4+P5) | 0.590 | ? |

Early exiting at P3 or P3+P4 destroys accuracy because deep features are the detection representation itself; this motivates the scale-drop (not exit) formulation in the paper.

### E. Full head-level block-skipping latency breakdown (VOC test2007, batch 32, RTX 5070 FP32)

| Config | mAP50-95 | GFLOPs | FPS |
|--------|----------|--------|-----|
| Full | 0.5898 | 5.21 | 995 |
| Block-skip 5%  | 0.381 | 4.79 | 991 |
| Block-skip 10% | 0.421 | 4.83 | 986 |
| Block-skip 25% | 0.478 | 4.94 | 962 |
| Block-skip 50% | 0.513 | 5.12 | 911 |
| Block-skip 100% (halo only) | 0.520 | 5.50 | 805 |

Head-level skipping never accelerates inference on GPU: halo padding inflates compute and gather/scatter overhead cancels the FLOPs savings.

### F. Training configuration (referred to in Sec. 4.1)

- **Model**: YOLO26n (end-to-end, reg_max=1), initialized from COCO-pretrained weights.
- **Training**: MuSGD optimizer, batch 32, image size 640, AMP, 100 epochs (VOC and VisDrone); COCO results use the official pretrained weights evaluated on val2017 (5,000 images).
- **Self-distillation gate**: per-scale 1x1 conv, trained with Adam (lr = 1e-3, batch 64) for 5 epochs with the backbone, neck, and head frozen; loss is the importance-weighted BCE of Eq. (1).
- **Throughput**: single-GPU, FP32, batch 32, NVIDIA GeForce RTX 5070 (12 GB), CUDA 13.2, no TensorRT.

### G. Unified latency / throughput (RTX 5070, FP32, no TensorRT)

| Model | mAP50-95 | GFLOPs | Params(M) | b1 latency (ms) | b1 FPS | b32 FPS |
|-------|----------|--------|-----------|-----------------|--------|---------|
| VOC full @640 | 0.5899 | 5.21 | 2.38 | 9.42 | 106 | 603 |
| VOC P3-drop @640 | 0.5806 | 4.93 | 2.36 | 8.43 | 119 | 769 |
| VisDrone full @640 | 0.1725 | 5.20 | 2.38 | 10.02 | 100 | 634 |
| VisDrone P5-drop @640 | 0.1725 | 4.66 | 1.70 | 9.40 | 106 | 670 |

FP16 did not change batch-1 latency materially (e.g., VOC full: 103.7 vs 106.2 FPS).

### H. Input-resolution scaling vs. scale drop

| Model / input | mAP50-95 | b32 FPS | b1 FPS |
|---------------|----------|---------|--------|
| VOC full @640 | 0.5899 | 603 | 106 |
| VOC full @512 | 0.5759 | 1033 | ? |
| VOC full @480 | 0.5710 | 1114 | ? |
| VisDrone full @640 | 0.1725 | 634 | 100 |
| VisDrone full @512 | 0.1450 | 910 | 95 |
| VisDrone full @480 | 0.1378 | 1009 | 94 |
| **VisDrone P5-drop @640** | **0.1725** | **670** | **113** |

Resolution scaling is cheap on VOC (small objects rare) but destroys 2.8?3.5 pp of mAP on VisDrone (85.9% small objects); P5-drop keeps mAP identical at 640.

### I. Cross-architecture check: YOLO11n on COCO val

Head-level scale-removal cost (mAP50-95). YOLO26n: 0.032/0.110/0.176; YOLO11n: 0.017/0.052/0.186 for P3/P4/P5. The ranking P3 < P4 < P5 holds for both architectures.

### J. Residuals of the three-point ln-s fits

| Dataset | G_P3 fit | G_P3 obs | G_P5 fit | G_P5 obs |
|---------|----------|----------|----------|----------|
| VOC | 18.7% | 22% | 41.6% | 39% |
| COCO | 36.3% | 28% | 23.3% | 30% |
| VisDrone | 48.0% | 53% | 11.1% | 7% |

Residuals are 2.6?8.3 pp; the relation is a trend, not a point-accurate predictor.

### K. Block-skipping latency (batch-32, RTX 5070)

No budget improved throughput: 5%/10%/25%/50% budgets gave ?baseline or slower b32 FPS (halo + gather/scatter overhead), and 100%-budget (halo only) was slower than baseline. Combined with the mAP collapse in Table 3, block-skipping is not a viable GPU speed-up without re-training the head on pruned features.
