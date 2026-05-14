# Embedded Machine Learning – Semester Project
### 3-Class Speech Command Classification (Option A)
**Classes:** `tree`, `three`, `two`

## Project Overview
Full embedded ML pipeline for speech command classification using the
Google Speech Commands v0.02 dataset. Covers MFCC preprocessing,
classical ML (Random Forest, SVM), deep learning (Standard CNN,
Mini-SqueezeNet, UltraLight CNN), TFLite conversion, PTQ and QAT.

---

## Run Order
1. `01_data_preprecess.ipynb` — Download data, extract MFCCs, save splits
2. `02_ML.ipynb` — Train and evaluate Random Forest & SVM
3. `03_DL_CNN.ipynb` — Train CNNs, export TFLite, apply PTQ & QAT
4. `04_Confusion_Matrix_Visual.ipynb` — Confusion matrices & metric plots

## Repository Structure

    speech_commands_project/
    ├── 01_data_preprecess.ipynb
    ├── 02_ML.ipynb
    ├── 03_DL_CNN.ipynb
    ├── 04_Confusion_Matrix_Visual.ipynb
    ├── README.md
    ├── data/
    │   ├── raw/           # Raw .wav files (excluded via .gitignore)
    │   ├── splits/        # train.txt / val.txt / test.txt
    │   └── processed/     # NumPy arrays (excluded via .gitignore)
    ├── results/           # .tflite models, CSVs, plots
    └── logs/              # Training histories (excluded via .gitignore)

> **Note:** Developed in Google Colab. All paths use Google Drive under
> `MyDrive/speech_commands_project/`. Adjust `base_path` if running elsewhere.

---

## Dependencies

All notebooks run in **Google Colab** (Python 3.10+):

| Package | Purpose |
|---|---|
| `tensorflow` | Model building, TFLite conversion |
| `tf_keras` | Keras API (legacy mode for QAT compatibility) |
| `tensorflow-model-optimization` | Quantization-Aware Training (QAT) |
| `librosa` | Audio loading, MFCC feature extraction |
| `numpy < 2.0.0` | Numerical arrays |
| `scikit-learn` | Random Forest, SVM, metrics |
| `pandas` | Results logging and CSV output |
| `matplotlib`, `seaborn` | Visualizations and confusion matrices |
| `objsize` | In-memory model size measurement |
| `psutil` | Process memory monitoring |

Non-default packages are installed inside the notebooks automatically.

---

## Data
Downloaded automatically via notebook 01 from:
http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

## Notes
- Fixed random seed (42) used throughout for reproducibility
- Inference benchmarks measured on single-thread CPU (num_threads=1)
