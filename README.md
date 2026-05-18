# Embedded Machine Learning – Semester Project
### 3-Class Speech Command Classification (Option A)
**Classes:** `tree`, `three`, `two`

---

## Project Overview

This project implements a full embedded machine learning pipeline for speech command classification using the [Google Speech Commands v0.02 dataset](http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz). It covers data preprocessing, classical ML, deep learning, TFLite conversion, and quantization (PTQ + QAT).

The three classes — *tree*, *three*, and *two* — were deliberately chosen for their phonetic similarity, making the classification task non-trivial and well-suited for evaluating model robustness.

---

## Repository Structure

```
speech_commands_project/
├── 01_data_preprecess.ipynb       # Dataset download, splitting, MFCC extraction & augmentation
├── 02_ML.ipynb                    # Classical ML: Random Forest & SVM
├── 03_DL_CNN.ipynb                # Deep Learning: CNN training, TFLite export, PTQ & QAT
├── 04_Confusion_Matrix_Visual.ipynb  # Confusion matrices & cross-metric visualizations
├── README.md
│
├── data/
│   ├── raw/           # Raw .wav files per class (tree/, three/, two/)
│   ├── splits/        # train.txt / val.txt / test.txt file lists
│   └── processed/     # Saved NumPy arrays (X_train.npy, y_train.npy, ...)
│
├── results/           # .tflite models, benchmark CSVs, plots
└── logs/              # Per-epoch training histories, process logs
```

> **Note:** This project was developed in Google Colab. All paths reference Google Drive under `MyDrive/speech_commands_project/`. Adjust `base_path` in each notebook if running in a different environment.

---

## Dependencies

All notebooks are designed to run in **Google Colab** (Python 3.10+). The following packages are used:

| Package | Purpose |
|---|---|
| `tensorflow` | Model building, TFLite conversion |
| `tf_keras` | Keras API (legacy mode for QAT compatibility) |
| `tensorflow-model-optimization` | Quantization-Aware Training (QAT) |
| `librosa` | Audio loading, MFCC feature extraction |
| `numpy` `< 2.0.0` | Numerical arrays |
| `scikit-learn` | Random Forest, SVM, metrics |
| `pandas` | Results logging and CSV output |
| `matplotlib` `seaborn` | Visualizations and confusion matrices |
| `objsize` | In-memory model size measurement |
| `psutil` | Process memory monitoring |

Install non-default packages (handled inside the notebooks):
```python
!pip install "numpy<2.0.0"
!pip install -q tf_keras
!pip install -q tensorflow-model-optimization
!pip install objsize
```

---

## Execution Steps

Run the notebooks **in order**. Each notebook saves its outputs to Google Drive so the next notebook can load them.

#### Step 1 — Data Preprocessing (`01_data_preprecess.ipynb`)

1. Mount Google Drive and create the project folder structure.
2. Download the Google Speech Commands v0.02 dataset via `tf.keras.utils.get_file`.
3. Copy the three class folders (`tree/`, `three/`, `two/`) to `data/raw/`.
4. Split files into train (80%) / val (10%) / test (10%) using a fixed seed (`42`) and save as `.txt` lists in `data/splits/`.
5. Extract **40-coefficient MFCCs** from every audio file (padded/clipped to 1 second, 16 kHz). Training samples receive data augmentation (time shifting ±100 ms, Gaussian noise, volume scaling ±20%).
6. Save feature arrays as NumPy `.npy` files in `data/processed/`.

#### Step 2 — Classical Machine Learning (`02_ML.ipynb`)

1. Load and **flatten** the 40×32 MFCC arrays to 1,280-length vectors.
2. Train **Random Forest** (100 estimators) and **SVM** (RBF kernel).
3. Evaluate both models: accuracy, classification report, confusion matrix heatmap.
4. Measure engineering metrics: inference latency, CPU throughput, model size, RAM footprint (objsize), and parameter count.

#### Step 3 — Deep Learning & Quantization (`03_DL_CNN.ipynb`)

Three CNN architectures are trained and benchmarked:

| Model | Architecture | Learning Rate |
|---|---|---|
| **Model A** | Standard 2D CNN (Conv→Pool→Conv→Pool→Dense) | 0.0001 |
| **Model B** | Mini-SqueezeNet with Fire Modules (1×1 squeeze + expand) | 0.001 |
| **Model C** | UltraLight MobileNet-style (Depthwise Separable Conv + GAP) | 0.001 |

For each model, three TFLite variants are produced and benchmarked:
- **Baseline** — float32, no optimization
- **PTQ** — Post-Training Quantization (int8, 200 representative samples)
- **QAT** — Quantization-Aware Training (8 fine-tune epochs at lr=1e-5)

Benchmark metrics per model: accuracy, latency (ms), CPU runtime (ms/sample), flash size (KB), memory RAM (KB), efficiency score.


#### Step 4 — Visualizations (`04_Confusion_Matrix_Visual.ipynb`)

1. Loads the final comparison table and regenerates cross-metric bar charts (accuracy, RAM, flash size, efficiency score) grouped by model and optimization type.
2. Runs fresh TFLite inference for **Model B PTQ** and **Model B QAT** and plots their confusion matrices.
3. Saves classification reports (`.txt`) and confusion matrices (`.csv` and `.png`) to `results/` and `logs/`.

---

#### Key Results

- **Recommended deployment model:** **Model B(Mini-SqueezeNet) QAT** — within 91.57% accuracy | 13.21 KB flash | 0.112 ms latency. Suitable for real-time, resource-constrained deployment.

---

### Reproducibility Notes

- A fixed random seed (`42`) is used throughout all notebooks for shuffling, model initialization, and numpy operations.
- The dataset split is deterministic: the same `train.txt` / `val.txt` / `test.txt` files are reused in all subsequent notebooks.
- All intermediate results are saved to Google Drive so individual notebooks can be re-run independently without repeating upstream steps.
- Training was performed on Google Colab GPU; inference benchmarks are measured on CPU (single thread, `num_threads=1`) to reflect embedded deployment conditions.
