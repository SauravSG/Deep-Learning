# End-to-End Deep Learning Pipeline & Modular Computer Vision Framework

A high-performance, modular computer vision pipeline built entirely from scratch using **PyTorch**. This framework handles the complete deep learning lifecycle: structured data ingestion, device-agnostic tensor execution, custom neural network architecture design, and low-overhead evaluation loops. 

Rather than relying on high-level wrappers, this repository implements core PyTorch mechanics to maximize execution efficiency and protect against common pipeline failures.

---

## 🛠️ Tech Stack & Dependencies

*   **Frameworks:** PyTorch (`torch.nn`, `torch.utils.data`, `torch.autograd`)
*   **Computer Vision:** Torchvision (`torchvision.transforms`), PIL (Pillow)
*   **Analysis & Visualization:** NumPy, Scikit-Learn, Matplotlib

---

## 🏗️ Project Architecture & Modules

The framework is decoupled into modular scripts to ensure reproducibility, scale, and clean maintenance, abandoning monolithic notebook architectures.
