[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**YOLO-SCEMA** (Spatial-Channel Enhanced Multiscale Attention) is a lightweight and efficient object detection model that integrates advanced attention mechanisms into the YOLOv8 architecture. This repository contains the official implementation of our paper:

**"Efficient Multiscale Attention with Spatial–Channel Reconstruction for Lightweight Object Detection"**

📄 **Paper**: [arXiv link](https://arxiv.org/abs/2501.xxxxx) | 👥 **Authors**: Mohammed MAIZA, Chahira CHERIF, Samira CHOURAQUI, Abdelmalik TALEB-AHMED

## ✨ Key Features

- **🚀 High Performance**: Achieves state-of-the-art accuracy with significantly fewer parameters
- **⚡ Lightweight Design**: Reduces parameters by 37.02% and computation by 8.72% compared to YOLOv8n
- **🔍 Advanced Attention**: Integrates Spatial-Channel Enhanced Multiscale Attention (SCEMA) module
- **🌙 Low-Light Robust**: Excellent performance on low-light datasets (ExDark)
- **🛰️ Dense Scene Capable**: Handles complex scenes with occlusion and scale variation
- **🔄 Multi-Scale Fusion**: Effective fusion of features at different scales
- **📱 Deployment Ready**: Export to TorchScript, ONNX, TensorRT, CoreML, TFLite

## 📊 Performance Comparison

| Model | mAP@50 (ExDark) | Params (M) | GFLOPs | Improvement |
|-------|----------------|------------|--------|-------------|
| YOLOv8n | 69.07% | 3.01 | 8.1 | Baseline |
| **YOLO-SCEMA-n** | **76.44%** | **1.90** | **7.4** | **+7.49% mAP** |
| YOLOv12n | 75.94% | 2.56 | 6.3 | +6.87% mAP |

*Results on ExDark dataset for low-light object detection*

## 🏗️ Model Architecture

YOLO-SCEMA introduces a novel **Spatial-Channel Enhanced Multiscale Attention (SCEMA)** module with parallel dual-branch architecture:

### Left Branch: Feature Refinement Module
- **Spatial Reconstruction Unit (SRU)**: Reduces spatial redundancy
- **Channel Reconstruction Unit (CRU)**: Minimizes channel redundancy
- **CBAM Attention**: Sequential channel and spatial attention

### Right Branch: Cross Spatial Learning Module
- **Feature Grouping**: Divides features into semantic groups
- **Multi-scale Processing**: Parallel 1×1 and 3×3 convolutions
- **Cross Spatial Learning**: Integrates information across spatial dimensions
