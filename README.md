<div align="center">

# YOLOv8 from Scratch — Object Detection in PyTorch

### A full, hand-built reimplementation of YOLOv8 (backbone → PANet neck → decoupled anchor-free head), trained on a 20-class PASCAL VOC dataset

**Sami El Akkad** · Computer-Vision Internship Project · **Neusoft**, Summer 2024

[![Framework](https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/YOLOv8-s%20variant-5E81AC)](#architecture)
[![Task](https://img.shields.io/badge/task-object%20detection-A3BE8C)](#)
[![Input](https://img.shields.io/badge/input-640×640-B48EAD)](#)
[![Docs](https://img.shields.io/badge/code%20comments-français-2E86AB)](#)

*Not a wrapper around the Ultralytics package — the network, the loss, the label
assignment and the training loop are all implemented by hand in PyTorch.
The code is documented in French.*

</div>

---

## What this is

A complete YOLOv8 object detector built **from the ground up** during my computer-vision
internship at Neusoft (summer 2024). Every moving part is written by hand so the whole
detection stack is transparent and inspectable:

- the **CSPDarknet backbone** (Conv·C2f·SPPF) producing 3 feature maps,
- the **PANet neck** fusing those scales top-down then bottom-up,
- the **decoupled, anchor-free head** with a **DFL** box-regression branch,
- the full **loss** — dynamic label assignment (TaskAlignedAssigner) + CIoU + DFL + BCE,
- the **two-phase freeze/unfreeze training loop**, EMA, cosine LR, Mosaic/MixUp augmentation,
- inference: DFL decode → NMS → drawing, plus FPS benchmark, heatmap view and ONNX export.

Trained on a **20-class PASCAL-VOC-format dataset (~21k images)** at 640×640, using the
**YOLOv8-s** variant (~11M parameters).

## Architecture

![YOLOv8 architecture](figures/yolov8_architecture.png)

The network keeps the three classic stages but each is implemented explicitly in
[`nets/backbone.py`](nets/backbone.py) and [`nets/yolo.py`](nets/yolo.py):

| Stage | What it does | Key idea |
|---|---|---|
| **Backbone — CSPDarknet** | 3×640×640 → feature maps at strides 8/16/32 (P3/P4/P5) | `C2f` blocks split-and-stack feature maps for richer gradient flow; `SPPF` stacks max-pools to widen the receptive field cheaply |
| **Neck — PANet** | fuses P3/P4/P5 top-down (upsample) then bottom-up (downsample) | small objects keep high-res detail, large objects keep deep semantics |
| **Head — decoupled, anchor-free** | two branches per scale: `cv2` box + `cv3` class | **DFL** predicts a 16-bin distribution per coordinate, then takes its expectation → continuous box edges without anchors |

### The C2f block (heart of the backbone)

![C2f block](figures/c2f_block.png)

## Training

![Training pipeline](figures/training_pipeline.png)

The recipe in [`train.py`](train.py) reflects how you actually train a detector on a
single GPU with limited memory:

- **Two phases.** *Freeze* the pretrained backbone first (fast, stable, cheap on VRAM),
  then *unfreeze* and fine-tune the whole network.
- **Pretrained start, always.** The backbone is initialised from `yolov8_s.pth` — features
  like edges and textures transfer; training from random weights on a dataset this size
  barely converges.
- **Loss** = `7.5·CIoU + 0.5·BCE(cls) + 1.5·DFL`, with positives chosen by the
  **TaskAlignedAssigner** (`align = score^α · IoU^β`), which couples classification
  confidence and localisation quality when assigning ground-truth boxes to anchors.
- **Stabilisers**: EMA weights for evaluation/saving, cosine LR with warmup, LR scaled to
  the batch size, gradient clipping at `max_norm=10`, Mosaic + MixUp augmentation.

### Training evidence

A short training run (freeze phase) shows the loss collapsing as expected from the
high-variance random head toward a converging detector:

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 55.81 | 3.56 |
| 2 | 4.16 | 3.61 |
| 3 | 4.15 | 3.63 |

*(Logs in [`logs/`](logs). The internship runs were short proof-of-pipeline trainings, not
a full 100-epoch schedule.)*

## Repository layout

```
.
├── config.py              # single source of truth for all paths (relative, portable)
├── train.py               # two-phase freeze/unfreeze training loop
├── predict.py             # inference driver: image / video / fps / dir / heatmap / onnx
├── yolo.py                # YOLO inference class (preprocess → decode → NMS → draw)
├── get_map.py             # VOC mAP@0.5 / COCO mAP@0.5:0.95 evaluation
├── voc_annotation.py      # builds train/val splits + 2007_train/val.txt from VOC XML
├── summary.py             # model FLOPs / params
├── nets/
│   ├── backbone.py        # CSPDarknet: Conv, Bottleneck, C2f, SPPF
│   ├── yolo.py            # YoloBody: PANet neck + decoupled DFL head + fuse()
│   └── yolo_training.py   # TaskAlignedAssigner, CIoU/DFL/BCE loss, EMA, LR schedule
├── utils/
│   ├── dataloader.py      # dataset + Mosaic / MixUp / letterbox augmentation
│   ├── utils_bbox.py      # anchors, DFL decode, NMS
│   ├── utils_fit.py       # one train+val epoch
│   ├── callbacks.py       # loss curves + mAP callback
│   ├── utils.py           # helpers (letterbox, classes, seeding, ...)
│   └── utils_map.py       # mAP computation (adapted from Cartucho/mAP + pycocotools)
└── figures/               # architecture diagrams (generated by make_diagrams.py)
```

*(The `VOCdevkit/` dataset, `model_data/*.pth` weights and `logs/` checkpoints are
intentionally **not** committed — see [`.gitignore`](.gitignore).)*

## How to run

```bash
pip install -r requirements.txt

# 1) build the splits + annotation files from a VOC-format dataset
python voc_annotation.py

# 2) train (edit hyper-params at the top of train.py)
python train.py

# 3) detect — set `mode` in predict.py: predict / video / fps / dir_predict / heatmap / export_onnx
python predict.py

# 4) evaluate
python get_map.py
```

A VOC-format dataset is expected under `VOCdevkit/VOC2007/{JPEGImages, Annotations, ImageSets}`,
and the YOLOv8-s pretrained weights under `model_data/`.

## Tech stack

`Python` · `PyTorch` · `NumPy` · `OpenCV` · `Pillow` · `matplotlib` · `tqdm` · VOC-format data ·
TensorBoard logging · ONNX export.

## Notes (honest)

- This is **internship work**: the goal was to implement and understand the full YOLOv8 stack,
  not to chase a leaderboard number. Training runs were short.
- Cleanup done while open-sourcing it: paths made **relative/portable**, duplicate and
  typo-named files removed, the dead code stripped, comments translated to **French**, the
  truncated `get_map.py` evaluation body completed, and a quote-corruption bug in
  `utils_map.py` (which had silently broken mAP evaluation) fixed.

## Author

**Sami El Akkad** — built during the computer-vision internship at **Neusoft**, summer 2024.

📧 [sam25@mails.tsinghua.edu.cn](mailto:sam25@mails.tsinghua.edu.cn) · 🔗 [LinkedIn](https://www.linkedin.com/in/samielakkad)

---

<div align="center">
Architecture diagrams and documentation are my own work · code comments in French.
</div>
