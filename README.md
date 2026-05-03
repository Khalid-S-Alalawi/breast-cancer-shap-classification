# Comparative Explainable AI for Breast Cancer Classification

This repository contains reproducible Python code for the manuscript:

**Comparative Explainable AI for Breast Cancer Classification: Cross-Model SHAP Agreement Analysis Using XGBoost and Logistic Regression**

The study evaluates Logistic Regression, Support Vector Machines (SVM), and XGBoost on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset using a unified preprocessing pipeline. It also applies SHAP explanations to XGBoost and Logistic Regression and quantifies cross-model feature-importance agreement using Spearman rank correlation.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| XGBoost | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9967 |

Additional findings:

- No statistically significant difference between any model pair using paired t-test and Wilcoxon signed-rank test.
- Cross-model SHAP agreement: Spearman r = 0.578, p = 0.0008.
- Seven of the top ten SHAP-ranked features overlapped between XGBoost and Logistic Regression.

## Repository Structure

```text
breast-cancer-shap-classification/
├── breast_cancer_analysis.py
├── shap_agreement_analysis.py
├── requirements.txt
├── README.md
└── LICENSE
```

The dataset file `wdbc.data` should be downloaded separately from the UCI Machine Learning Repository and placed in the repository root directory before running the scripts.

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

- Source: UCI Machine Learning Repository
- DOI: https://doi.org/10.24432/C5DW2B
- Instances: 569
- Features: 30 numerical features derived from fine needle aspirate cell nucleus images
- Target: Benign (0) or Malignant (1)

Dataset citation:

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## Installation

```bash
git clone https://github.com/Khalid-S-Alalawi/breast-cancer-shap-classification.git
cd breast-cancer-shap-classification
pip install -r requirements.txt
```

## Running the Analysis

Download `wdbc.data` from the UCI repository and place it in the same directory as the Python scripts.

Run the main experiment:

```bash
python breast_cancer_analysis.py
```

Run the cross-model SHAP agreement analysis:

```bash
python shap_agreement_analysis.py
```

## Output Files

| File | Description |
|---|---|
| `roc_curves.png` | ROC curves for Logistic Regression, SVM, and XGBoost |
| `confusion_matrices.png` | Confusion matrices for all three models |
| `feature_importance.png` | XGBoost top-10 feature importance plot |
| `shap_summary.png` | SHAP summary plot for XGBoost |
| `shap_waterfall.png` | SHAP waterfall plot for one XGBoost prediction |
| `shap_summary_lr.png` | SHAP summary plot for Logistic Regression |
| `shap_agreement_bars.png` | Side-by-side mean absolute SHAP bar charts |
| `shap_agreement_scatter.png` | Feature-rank agreement scatter plot |
| `shap_agreement_heatmap.png` | Normalised mean absolute SHAP heatmap |
| `shap_agreement_dotplot.png` | Feature-rank agreement dot plot |

## Methodology Summary

1. Data preprocessing with `StandardScaler` inside scikit-learn pipelines.
2. Hyperparameter tuning with grid search and 5-fold stratified cross-validation.
3. Model evaluation using an 80/20 stratified test split and 10-fold stratified cross-validation.
4. Pairwise statistical testing using paired t-test and Wilcoxon signed-rank test.
5. SHAP explainability using TreeExplainer for XGBoost and LinearExplainer for Logistic Regression.
6. Cross-model SHAP agreement using Spearman rank correlation across all 30 features.

## Dependencies

Python 3.8 or higher is recommended. Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap scipy
```


