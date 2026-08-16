# PixelAI: Semiconductor Image Restoration Engine

This repository contains the training and evaluation code for restoring severely degraded semiconductor wafer scans, removing impulse noise, Gaussian noise, and speckle artifacts to reveal underlying micro-defects.

## 📁 Repository Contents

This repository is structured to meet the evaluation requirements:
* **`evaluate.py`**: Standalone evaluation script for running inference on new test images.
* **`train.py`**: The training script used to reproduce the model training process from scratch.
* **`architecture.py`**: The activation-free neural network backbone (SemiconNAFNet).
* **`checkpoints/best_model.pt`**: The final trained model weights. *(Note: If weights are missing due to GitHub size limits, see the download link in the Model Weights section below).*
* **`restored_output/`**: Folder containing the final restored images generated from the test set.
* **`requirements.txt`**: Complete list of Python dependencies required for reproducibility.

---

## 🛠️ Setup & Installation

Reviewers can run this pipeline by cloning the repository and installing the frozen environment dependencies. Python 3.8+ is recommended.

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)<YOUR-GITHUB-USERNAME>/PixelAI.git
cd PixelAI
