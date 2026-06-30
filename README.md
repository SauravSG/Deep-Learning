# 🧠 PyTorch Deep Learning Sandbox & Computer Vision Framework

Welcome to my deep learning and computer vision repository! This project tracks my technical progression from foundational tensor math to advanced multi-class Convolutional Neural Networks (CNNs), MLOps experiment tracking, and production-grade software modularity. 

Every notebook in this repository is self-contained, heavily documented, and optimized for device-agnostic execution (`CPU`, `GPU`, or `CUDA`).

---

## 🗺️ Repository Structure & Learning Roadmap

As displayed in the repository blueprint (`image_710c84.png`), the project is structurally split into fundamental building blocks, advanced dataset tasks, and experiment tracking:

```text
├── 🛠️ FUNDAMENTALS & WORKFLOWS
│   ├── CH_00_PyTorch_Fundamentals.ipynb           # Tensor creation, slicing, matrix math, and GPU memory routing
│   ├── CH_01_PyTorch_Workflow.ipynb               # Linear regression modeling, data splitting, train/test loops, and serialization
│   ├── CH_02_Neural_Network_Classification.ipynb  # Non-linear decision boundaries, binary/multi-class classification setups
│   └── Hands_on_Regression_in_DL.ipynb           # Deep Learning regression patterns, feature scaling, and optimization
│
├── 🖼️ CUSTOM DATA & COMPUTER VISION
│   ├── CH_04_PyTorch_Custom_Datasets.ipynb        # Subclassing torch.utils.data.Dataset & image preprocessing/transforms
│   ├── CIFAR_10.ipynb                            # Deep multi-class CNN classifiers trained on the benchmark CIFAR-10 dataset
│   └── Hands_on_FashionMNIST_v2.ipynb            # Dynamic multi-class clothes classification pipeline on FashionMNIST
│
└── 🚀 MLOPS & PRODUCTION REFACTORING
    ├── CIFAR_10_with_WandB_Implementation.ipynb  # Cloud experiment tracking, metric logging, and hyperparameter management via WandB
    └── CH_05_pytorch_going_modular_cell_mode.ipynb # Re-architecting notebook cells into clean, production-grade .py scripts
