"""Main experimental pipeline for WDBC breast cancer classification.

This script reproduces the preprocessing, model tuning, evaluation,
statistical testing, feature importance, and SHAP analyses reported in the paper.
Place `wdbc.data` in the same directory before running.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "wdbc.data"
np.random.seed(RANDOM_STATE)

COLUMN_NAMES = [
    "id", "diagnosis",
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]


def load_wdbc(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    return df.drop(columns=["diagnosis"]), df["diagnosis"]


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
        ]),
    }


def tune_models(models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    param_grids = {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "SVM": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
        "XGBoost": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.01, 0.1],
        },
    }

    best_models: dict[str, Pipeline] = {}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== Hyperparameter Tuning ===")
    for name, model in models.items():
        grid = GridSearchCV(model, param_grids[name], cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"\n{name}")
        print("Best Params:", grid.best_params_)
        print("Best CV ROC-AUC:", round(grid.best_score_, 4))

    return best_models


def cross_validate(best_models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: dict[str, np.ndarray] = {}

    print("\n=== 10-Fold Cross-Validation ===")
    for name, model in best_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = scores
        print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    return cv_scores


def statistical_tests(cv_scores: dict[str, np.ndarray]) -> None:
    print("\n=== Statistical Significance Testing ===")
    for name_a, name_b in combinations(cv_scores.keys(), 2):
        scores_a = cv_scores[name_a]
        scores_b = cv_scores[name_b]
        t_stat, t_p = stats.ttest_rel(scores_a, scores_b)
        w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
        print(f"\n{name_a} vs {name_b}:")
        print(f"  Paired t-test : t={t_stat:.4f}, p={t_p:.4f}")
        print(f"  Wilcoxon test : W={w_stat:.4f}, p={w_p:.4f}")
        print(f"  Result        : {'SIGNIFICANT' if t_p < 0.05 else 'not significant'} (alpha=0.05)")


def evaluate_models(
    best_models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    results = []
    plt.figure(figsize=(10, 6))

    for name, model in best_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision (Malignant)": round(precision_score(y_test, y_pred, pos_label=1), 4),
            "Recall (Malignant)": round(recall_score(y_test, y_pred, pos_label=1), 4),
            "F1-Score (Malignant)": round(f1_score(y_test, y_pred, pos_label=1), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("\n=== FINAL TEST SET RESULTS ===")
    print(results_df.to_string(index=False))
    return results_df


def plot_confusion_matrices(best_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, best_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix\n{name}")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()


def xgboost_feature_importance(best_models: dict[str, Pipeline], feature_names: pd.Index) -> None:
    xgb_model = best_models["XGBoost"].named_steps["model"]
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()


def shap_analysis(best_models: dict[str, Pipeline], X_test: pd.DataFrame) -> None:
    print("\n=== SHAP Analysis (XGBoost) ===")
    xgb_pipeline = best_models["XGBoost"]
    xgb_model = xgb_pipeline.named_steps["model"]
    xgb_scaler = xgb_pipeline.named_steps["scaler"]
    X_test_scaled_xgb = pd.DataFrame(xgb_scaler.transform(X_test), columns=X_test.columns)

    explainer_xgb = shap.Explainer(xgb_model, X_test_scaled_xgb)
    shap_values_xgb = explainer_xgb(X_test_scaled_xgb)

    shap.summary_plot(shap_values_xgb, X_test_scaled_xgb, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    shap.plots.waterfall(shap_values_xgb[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== SHAP Analysis (Logistic Regression) ===")
    lr_pipeline = best_models["Logistic Regression"]
    lr_model = lr_pipeline.named_steps["model"]
    lr_scaler = lr_pipeline.named_steps["scaler"]
    X_test_scaled_lr = pd.DataFrame(lr_scaler.transform(X_test), columns=X_test.columns)

    explainer_lr = shap.LinearExplainer(lr_model, X_test_scaled_lr)
    shap_values_lr = explainer_lr(X_test_scaled_lr)

    shap.summary_plot(shap_values_lr, X_test_scaled_lr, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_lr.png", dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    X, y = load_wdbc()
    print("Dataset shape:", X.shape)
    print("Class distribution (0=Benign, 1=Malignant):", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    best_models = tune_models(build_models(), X_train, y_train)
    cv_scores = cross_validate(best_models, X_train, y_train)
    statistical_tests(cv_scores)
    evaluate_models(best_models, X_train, X_test, y_train, y_test)
    plot_confusion_matrices(best_models, X_test, y_test)
    xgboost_feature_importance(best_models, X.columns)
    shap_analysis(best_models, X_test)
    print("\n=== Analysis complete. All plots saved. ===")


if __name__ == "__main__":
    main()
