# cnn-vgg16-densenet-custom-init
This repository presents PyTorch-based from-scratch implementations of classical convolutional neural network (CNN) architectures — **Custom CNN**, **VGG16**, and **DenseNet** — with a focus on architectural fidelity, modularity, and manual weight initialization. The project is intended for research, educational, and benchmarking purposes in the context of deep learning systems design.

---

## Abstract

We reimplement key deep convolutional architectures using PyTorch without relying on pretrained models or external wrappers. Our goal is to highlight the architectural intuition behind deep CNNs and the impact of initialization schemes on model behavior. Each network is built to be modular and extensible for future experimentation in areas such as weight initialization strategies, training dynamics, and transfer learning.

---

## Objectives

- To understand and replicate foundational CNN architectures from first principles.
- To explore the structural design of deep networks such as VGG16 and DenseNet.
- To implement and analyze manual weight initialization techniques.
- To provide a clean and extensible codebase for experimentation and academic use.

---

## Architectures Implemented

### 1. Custom CNN
- 2 Convolutional layers followed by 2 Fully Connected layers
- Designed for 28×28 grayscale image input (e.g., MNIST)
- Serves as a baseline for deeper architectures

### 2. VGG16 (Visual Geometry Group, 2014)
- 13 Convolutional layers + 3 Fully Connected layers
- Structured into 5 convolutional blocks
- Includes Adaptive Average Pooling and Dropout
- Normal distribution weight initialization with post-rescaling

### 3. DenseNet (Densely Connected Networks)
- Uses Dense Blocks and Transition Layers
- Features direct connections between all layers in a block
- Contains BatchNorm, ReLU, Conv, and AvgPool units
- Implements global average pooling before classification

---

## Methodology

All models are implemented using low-level PyTorch modules (`nn.Conv2d`, `nn.Linear`, etc.) without using prebuilt model classes. Weight initialization is manually applied using a normal distribution (`N(0, 1)`) followed by min-max rescaling to [0, 1] for controlled experimentation. Biases are initialized to a small constant (`0.1`) for stability.

---
