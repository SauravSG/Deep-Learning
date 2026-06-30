# 🧠 PyTorch Deep Learning Sandbox & Computer Vision Framework

Welcome to my deep learning proving ground! This repository tracks my tactical evolution from slicing raw tensors to training complex multi-class Convolutional Neural Networks (CNNs), integrating MLOps pipelines, and refactoring chaotic notebook code into modular, production-grade software architecture. 

No high-level wrappers. No training wheels. Just pure PyTorch mechanics, explicit data engineering, and mathematical optimization.

---

## 🗺️ Repository Structure & Learning Roadmap

As displayed in the repository blueprint (`image_710c84.png`), this project maps a structured journey from machine learning basics to enterprise-ready deep learning pipelines:

```text
├── 🛠️ FUNDAMENTALS & WORKFLOWS
│   ├── CH_00_PyTorch_Fundamentals.ipynb           # Tensor mathematics, matrix multiplication, and GPU routing
│   ├── CH_01_PyTorch_Workflow.ipynb               # Linear regression modeling, data splitting, and model serialization
│   ├── CH_02_Neural_Network_Classification.ipynb  # Non-linear decision boundaries, binary & multi-class classification setups
│   └── Hands_on_Regression_in_DL.ipynb           # Advanced regression patterns, loss optimization, and feature analysis
│
├── 🖼️ CUSTOM DATA & COMPUTER VISION
│   ├── CH_04_PyTorch_Custom_Datasets.ipynb        # Subclassing torch.utils.data.Dataset & image preprocessing/transforms
│   ├── CIFAR_10.ipynb                            # Multi-class CNN classifiers trained on the benchmark CIFAR-10 dataset
│   └── Hands_on_FashionMNIST_v2.ipynb            # Dynamic multi-class clothes classification pipeline on FashionMNIST
│
└── 🚀 MLOPS & PRODUCTION REFACTORING
    ├── CIFAR_10_with_WandB_Implementation.ipynb  # Cloud experiment tracking, metric logging, and hyperparameter management via WandB
    └── CH_05_pytorch_going_modular_cell_mode.ipynb # Re-architecting notebook cells into clean, production-grade .py scripts
