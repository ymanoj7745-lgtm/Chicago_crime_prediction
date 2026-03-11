# 🏙️ Chicago Crime ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=for-the-badge)

**A full end-to-end machine learning pipeline on Chicago crime data — from raw data ingestion to model explainability — across three distinct prediction tasks.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tasks](#-ml-tasks)
- [Results](#-results)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Notebooks](#-notebooks)
- [Outputs](#-outputs)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

This project applies supervised machine learning to the [Chicago Crime Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) to solve three real-world prediction problems:

- **Will a crime result in an arrest?** *(Binary Classification)*
- **What type of crime is it?** *(Multi-class Classification)*
- **How many crimes will occur in a given area?** *(Regression)*

The pipeline covers every stage of the ML lifecycle: data loading, feature engineering, preprocessing, model training & tuning, evaluation, and feature importance analysis — across 6 structured Jupyter notebooks.

---

## 🎯 ML Tasks

| # | Task | Type | Target |
|---|------|------|--------|
| 1 | **Arrest Prediction** | Binary Classification | Will the crime lead to an arrest? |
| 2 | **Crime Type Prediction** | Multi-class Classification | What category of crime occurred? |
| 3 | **Crime Count per Area** | Regression | How many crimes will occur in a community area? |

---

## 📊 Results

### Task 1 — Arrest Prediction
| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting (HGB) |
| Test AUC | **0.8852** |
| Test F1 (weighted) | **0.7918** |
| Test Accuracy | **0.7957** |

### Task 2 — Crime Type Prediction
| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting (HGB) |
| Classes | 10 crime types |
| Test F1 (weighted) | See scorecard |
| Test Accuracy | See scorecard |

### Task 3 — Crime Count per Area
| Metric | Score |
|--------|-------|
| Best Model | Gradient Boosting (HGB) |
| Test RMSE | See scorecard |
| Test R² | See scorecard |

> Full scorecard available in `ml_data/05_scorecard.csv` after running the pipeline.

---

## 🏗️ Pipeline Architecture

```
Raw Data
   │
   ▼
┌─────────────────────────────┐
│  01 · Data Loading &        │  Sampling, EDA, train/val/test split
│       Sampling              │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  02 · Feature Engineering   │  Temporal, spatial & interaction features
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  03 · Preprocessing         │  Scaling, encoding, task-specific splits
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  04 · Model Training        │  Cross-validation, hyperparameter tuning
│                             │  Logistic Regression, Ridge, HGB, RF, ...
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  05 · Evaluation &          │  Scorecard, ROC curves, confusion matrices
│       Comparison            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  06 · Feature Importance    │  Tree/coef importance, permutation importance,
│       & Final Report        │  SHAP, learning curves, error analysis
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
chicago-crime-ml/
│
├── 📓 01_data_loading_sampling.ipynb
├── 📓 02_feature_engineering.ipynb
├── 📓 03_preprocessing.ipynb
├── 📓 04_model_training.ipynb
├── 📓 05_evaluation_comparison.ipynb
├── 📓 06_feature_importance_report.ipynb
│
├── ml_data/
│   ├── task1_arrest_prediction.csv
│   ├── task2_crime_type_prediction.csv
│   ├── task3_crime_count_per_area.csv
│   ├── task1_engineered.csv
│   ├── task2_engineered.csv
│   ├── task3_engineered.csv
│   ├── taskN_preprocessed.npz        # Train/val/test arrays
│   ├── taskN_scaler.pkl              # Fitted scalers
│   ├── taskN_meta.json               # Feature names & metadata
│   ├── task2_label_map.json          # Crime type label mapping
│   ├── 05_scorecard.csv              # All model comparison results
│   └── 06_final_report.json         # Pipeline summary report
│
├── ml_models/
│   ├── task1_best_model.pkl
│   ├── task2_best_model.pkl
│   ├── task3_best_model.pkl
│   └── training_summary.json
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/chicago-crime-ml.git
cd chicago-crime-ml

# 2. Create and activate environment
conda create -n chicago_crime_env python=3.12
conda activate chicago_crime_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Requirements

```
scikit-learn>=0.24
pandas
numpy
matplotlib
seaborn
joblib
shap
packaging
jupyter
```

> **Note:** The pipeline was developed on scikit-learn 0.24 (booksenv). If running on sklearn 1.3+, the notebooks include automatic compatibility shims for loading saved models.

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_loading_sampling` | Load raw Chicago crime data, explore distributions, stratified sampling, create three task-specific datasets |
| `02_feature_engineering` | Extract temporal features (hour, day, month, season), spatial features (community area, district), crime interaction features |
| `03_preprocessing` | Scale numerical features, encode categoricals, create train/validation/test splits, save `.npz` arrays |
| `04_model_training` | Train multiple model families per task with cross-validation, select best model, save to `ml_models/` |
| `05_evaluation_comparison` | Scorecard across all models, ROC curves, confusion matrices, regression diagnostics |
| `06_feature_importance_report` | Feature importance (tree/coef/permutation), SHAP analysis, learning curves, error analysis, final report |

---

## 📤 Outputs

After running all notebooks end-to-end, the following are generated:

| Output | Description |
|--------|-------------|
| `ml_data/05_scorecard.csv` | All model metrics across all tasks |
| `ml_data/06_final_report.json` | JSON summary of best models and top features |
| `ml_models/task*_best_model.pkl` | Serialized best models per task |
| `ml_data/06_tree_importance.png` | Feature importance bar charts |
| `ml_data/06_permutation_importance.png` | Permutation importance charts |
| `ml_data/06_learning_curves.png` | Bias-variance diagnosis plots |
| `ml_data/06_task1_error_analysis.png` | Error breakdown for Task 1 |
| `ml_data/06_task3_error_analysis.png` | Residual analysis for Task 3 |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Model training, evaluation, preprocessing |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` / `seaborn` | Visualisation |
| `joblib` | Model serialisation |
| `shap` | Model explainability |
| `packaging` | sklearn version compatibility |

---

## ⚠️ sklearn Compatibility Note

The best models were saved with an older scikit-learn version. Notebook `06` includes automatic compatibility shims that detect your sklearn version and apply the appropriate patches — so the pipeline runs correctly on any sklearn from `0.21` through `1.8+`.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ☕ and too many kernel restarts
</div>
