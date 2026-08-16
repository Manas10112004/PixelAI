# PixelAI: Semiconductor Image Restoration Engine

This repository contains the training and evaluation code for restoring severely degraded semiconductor wafer scans. The model strips away heavy salt-and-pepper (impulse) noise, Gaussian noise, and speckle artifacts to reveal underlying micro-defects for hardware analysis.

## 📁 Repository Contents

This repository is structured to meet all evaluation requirements:
* **`evaluate.py`**: Standalone evaluation script for running inference on new test images.
* **`train.py`**: The training script used to reproduce the model training process from scratch.
* **`architecture.py`**: The activation-free neural network backbone (SemiconNAFNet).
* **`checkpoints/best_model.pt`**: The final trained model weights.
* **`restored_output/`**: Folder containing the final restored images generated from the test set.
* **`requirements.txt`**: Complete list of Python dependencies required for reproducibility.

---

## 🛠️ Setup & Installation

Reviewers can seamlessly run this pipeline by cloning the repository and installing the frozen environment dependencies. Python 3.8+ is recommended.

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)<YOUR-GITHUB-USERNAME>/PixelAI.git
cd PixelAI

```

**2. Install dependencies:**
Install the exact environment packages required for reproducibility:

```bash
pip install -r requirements.txt

```

---

## 🚀 Evaluation (Running Inference)

The standalone evaluation script accepts an input directory of degraded images and an output directory where the restored images will be saved. It automatically loads the trained model weights and processes the files without any manual edits required.

**Run the following command:**

```bash
python evaluate.py --input_dir ./path/to/test/images --output_dir ./path/to/save/outputs

```

* `--input_dir`: Path to the folder containing the noisy test images (e.g., `.png` or `.npy`).
* `--output_dir`: Path to the folder where the restored outputs should be written.

---

## 🧠 Training the Model

To reproduce the training process from scratch, run the training script. Ensure your training datasets are located in the directories specified within `dataset.py` or `data_ingestion.py` before running.

```bash
python train.py

```

The script will automatically utilize mixed-precision training (`torch.amp.autocast`) and save the best model weights to the `./checkpoints/` directory based on validation loss.

---

## 📦 Trained Model Weights

The final trained model weights (`best_model.pt`) are loaded automatically by the evaluation script.

*(If you are downloading this repository via a zip file or if Git LFS was not used, please download the weights from the link below and place them in the `checkpoints` directory before running inference):*

**[Download Model Weights Here] (https://drive.google.com/file/d/1joDtipMvIZ3c_RU8M6xbC8SKwy-SUzn7/view?usp=sharing)**

```

```
