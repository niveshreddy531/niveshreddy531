# 🔬 AI-Driven Healthcare Diagnostics — Breast Cancer Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A Convolutional Neural Network (CNN) pipeline for early-stage breast cancer detection using medical imaging datasets — designed to support clinical decision-making through accurate, interpretable predictions.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection significantly improves survival rates. This project builds an end-to-end CNN-based classification model that distinguishes between malignant and benign tumour images, with full performance visualisation for clinical interpretability.

**Key outcomes:**
- Binary classification: Malignant vs. Benign
- End-to-end ML pipeline: data ingestion → preprocessing → training → evaluation
- Visual reporting of model performance for non-technical stakeholders

---

## 🗂️ Dataset

- **Source:** [Breast Cancer Histopathological Image Dataset (BreakHis)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) / [Kaggle Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Format:** Medical images (`.png` / `.jpg`) or tabular features
- **Classes:** Malignant, Benign
- **Split:** 80% train / 10% validation / 10% test

---

## 🏗️ Architecture

```
Input Layer
    ↓
Conv2D (32 filters, 3×3) + ReLU + MaxPooling
    ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPooling
    ↓
Conv2D (128 filters, 3×3) + ReLU + MaxPooling
    ↓
Flatten → Dense (256) + Dropout (0.5)
    ↓
Output: Dense (1) + Sigmoid
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## 🔄 Pipeline

```
1. Data Loading & EDA
        ↓
2. Preprocessing
   - Resize images to 224×224
   - Normalise pixel values (0–1)
   - Data augmentation (flip, rotate, zoom)
        ↓
3. Model Training
   - CNN architecture (custom)
   - Optimizer: Adam | Loss: Binary Crossentropy
   - Early stopping + learning rate scheduling
        ↓
4. Evaluation
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - Loss & accuracy curves
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~94% |
| Validation Accuracy | ~89% |
| Precision | ~90% |
| Recall | ~88% |

> Performance visualisations (confusion matrix, loss curves) are available in the `/outputs` folder.

---

## 📁 Project Structure

```
breast-cancer-detection/
│
├── data/
│   ├── raw/                  # Original images
│   └── processed/            # Resized & normalised images
│
├── notebooks/
│   └── breast_cancer_cnn.ipynb
│
├── src/
│   ├── preprocess.py         # Data loading & augmentation
│   ├── model.py              # CNN architecture
│   └── evaluate.py           # Metrics & visualisation
│
├── outputs/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/breast-cancer-detection.git
cd breast-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/breast_cancer_cnn.ipynb
```

---

## 📦 Requirements

```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 💡 Key Learnings

- Applied data augmentation to address class imbalance in medical datasets
- Used dropout regularisation to reduce overfitting on a small training set
- Communicated model performance through visual reports (confusion matrix, ROC curve) accessible to non-technical stakeholders
- Gained hands-on experience with the full ML pipeline in a healthcare context

---

## 👤 Author

**Annadi Nivesh Reddy**
MSc Computer Science — Middlesex University, London
[LinkedIn](https://www.linkedin.com) | niveshreddy756@gmail.com
