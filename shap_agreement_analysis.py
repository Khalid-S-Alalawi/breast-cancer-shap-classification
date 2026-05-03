"""Cross-model SHAP agreement analysis for XGBoost and Logistic Regression.

This script reproduces the Spearman rank correlation, rank comparison table,
and agreement figures reported in the paper. Place `wdbc.data` in the same
folder before running.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def build_models() -> tuple[Pipeline, Pipeline]:
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1, max_iter=5000, random_state=RANDOM_STATE)),
    ])

    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])
    return lr_pipeline, xgb_pipeline


def compute_mean_abs_shap(model_pipeline: Pipeline, X_test: pd.DataFrame, model_type: str) -> pd.Series:
    scaler = model_pipeline.named_steps["scaler"]
    model = model_pipeline.named_steps["model"]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if model_type == "xgboost":
        explainer = shap.Explainer(model, X_test_scaled)
    elif model_type == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_test_scaled)
    else:
        raise ValueError("model_type must be 'xgboost' or 'logistic_regression'.")

    shap_values = explainer(X_test_scaled)
    return pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_test.columns)


def create_agreement_table(mean_shap_xgb: pd.Series, mean_shap_lr: pd.Series) -> pd.DataFrame:
    xgb_rank = mean_shap_xgb.rank(ascending=False).astype(int)
    lr_rank = mean_shap_lr.rank(ascending=False).astype(int)

    return pd.DataFrame({
        "XGBoost Mean |SHAP|": mean_shap_xgb,
        "LR Mean |SHAP|": mean_shap_lr,
        "XGBoost Rank": xgb_rank,
        "LR Rank": lr_rank,
        "Rank Difference (|XGB - LR|)": (xgb_rank - lr_rank).abs().astype(int),
    }).sort_values("XGBoost Mean |SHAP|", ascending=False)


def report_agreement(shap_agreement: pd.DataFrame) -> tuple[float, float]:
    spearman_corr, spearman_p = stats.spearmanr(
        shap_agreement["XGBoost Mean |SHAP|"],
        shap_agreement["LR Mean |SHAP|"],
    )

    print("\n" + "=" * 60)
    print("CROSS-MODEL SHAP AGREEMENT — SPEARMAN RANK CORRELATION")
    print("=" * 60)
    print(f"  Spearman r  = {spearman_corr:.4f}")
    print(f"  p-value     = {spearman_p:.6f}")

    if spearman_p < 0.05:
        if spearman_corr >= 0.7:
            strength = "STRONG"
        elif spearman_corr >= 0.4:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        print(f"  Result      : SIGNIFICANT {strength.lower()} agreement (p < 0.05)")
    else:
        print("  Result      : No significant agreement (p >= 0.05)")
    print("=" * 60)

    print("\nFull Feature Rank Comparison Table (sorted by XGBoost importance):")
    print(shap_agreement.to_string())

    print("\nTop 10 Features by XGBoost SHAP Rank vs LR Rank:")
    print(shap_agreement[["XGBoost Rank", "LR Rank", "Rank Difference (|XGB - LR|)"]].head(10).to_string())

    xgb_top10 = set(shap_agreement[shap_agreement["XGBoost Rank"] <= 10].index)
    lr_top10 = set(shap_agreement[shap_agreement["LR Rank"] <= 10].index)
    agreed_top10 = xgb_top10 & lr_top10

    print(f"\nFeatures in top 10 for BOTH models ({len(agreed_top10)} features):")
    for feature in sorted(agreed_top10):
        print(
            f"  {feature:35s}  "
            f"XGB rank={int(shap_agreement.loc[feature, 'XGBoost Rank']):2d}  "
            f"LR rank={int(shap_agreement.loc[feature, 'LR Rank']):2d}"
        )

    return spearman_corr, spearman_p


def plot_bar_comparison(shap_agreement: pd.DataFrame, spearman_corr: float, spearman_p: float) -> None:
    top_features = shap_agreement.head(15).index
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    xgb_vals = shap_agreement.loc[top_features, "XGBoost Mean |SHAP|"].sort_values()
    axes[0].barh(range(len(xgb_vals)), xgb_vals.values, color="#2E75B6", alpha=0.88)
    axes[0].set_yticks(range(len(xgb_vals)))
    axes[0].set_yticklabels(xgb_vals.index, fontsize=10)
    axes[0].set_xlabel("Mean |SHAP Value|", fontsize=11)
    axes[0].set_title("XGBoost — Mean |SHAP|", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    lr_vals = shap_agreement.loc[top_features, "LR Mean |SHAP|"].sort_values()
    axes[1].barh(range(len(lr_vals)), lr_vals.values, color="#ED7D31", alpha=0.88)
    axes[1].set_yticks(range(len(lr_vals)))
    axes[1].set_yticklabels(lr_vals.index, fontsize=10)
    axes[1].set_xlabel("Mean |SHAP Value|", fontsize=11)
    axes[1].set_title("Logistic Regression — Mean |SHAP|", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Cross-Model SHAP Feature Importance Comparison\n"
        f"Spearman r = {spearman_corr:.3f}  (p = {spearman_p:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("shap_agreement_bars.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_rank_scatter(shap_agreement: pd.DataFrame, spearman_corr: float, spearman_p: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    xgb_ranks = shap_agreement["XGBoost Rank"]
    lr_ranks = shap_agreement["LR Rank"]
    rank_diff = shap_agreement["Rank Difference (|XGB - LR|)"]

    scatter = ax.scatter(
        xgb_ranks,
        lr_ranks,
        c=rank_diff,
        cmap="RdYlGn_r",
        s=90,
        alpha=0.85,
        zorder=3,
        vmin=0,
        vmax=20,
    )
    plt.colorbar(scatter, ax=ax, label="Rank Difference |XGB rank − LR rank|")

    for feature in shap_agreement.head(10).index:
        ax.annotate(feature, (xgb_ranks[feature], lr_ranks[feature]), textcoords="offset points", xytext=(6, 3), fontsize=7.5)

    ax.plot([1, 30], [1, 30], "k--", alpha=0.35, label="Perfect agreement")
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)
    ax.set_xlabel("XGBoost Feature Rank  (1 = most important)", fontsize=11)
    ax.set_ylabel("Logistic Regression Feature Rank  (1 = most important)", fontsize=11)
    ax.set_title(
        f"SHAP Feature Rank Agreement: XGBoost vs Logistic Regression\n"
        f"Spearman r = {spearman_corr:.3f}  (p = {spearman_p:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("shap_agreement_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_heatmap(shap_agreement: pd.DataFrame, spearman_corr: float, spearman_p: float) -> None:
    top_features = shap_agreement.head(15).index
    heatmap_data = shap_agreement.loc[top_features, ["XGBoost Mean |SHAP|", "LR Mean |SHAP|"]]
    heatmap_data.columns = ["XGBoost", "Logistic Regression"]
    heatmap_norm = heatmap_data.div(heatmap_data.max())

    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(
        heatmap_norm,
        annot=heatmap_data.round(4),
        fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Mean |SHAP|  (0 = min, 1 = max within model)"},
    )
    ax.set_title(
        f"Top 15 Features: Normalised Mean |SHAP| per Model\n"
        f"Spearman r = {spearman_corr:.3f}  (p = {spearman_p:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig("shap_agreement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_rank_dotplot(shap_agreement: pd.DataFrame) -> None:
    plot_df = shap_agreement.copy()
    plot_df["Avg Rank"] = (plot_df["XGBoost Rank"] + plot_df["LR Rank"]) / 2
    plot_df = plot_df.sort_values("Avg Rank").head(20)

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = range(len(plot_df))

    ax.scatter(plot_df["XGBoost Rank"], y_pos, s=100, color="#2E75B6", label="XGBoost rank", zorder=3)
    ax.scatter(plot_df["LR Rank"], y_pos, s=100, color="#ED7D31", label="LR rank", zorder=3, marker="D")

    for i, feature in enumerate(plot_df.index):
        ax.plot(
            [plot_df.loc[feature, "XGBoost Rank"], plot_df.loc[feature, "LR Rank"]],
            [i, i],
            color="grey",
            linewidth=1.2,
            alpha=0.6,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_xlabel("Feature Rank  (1 = most important)", fontsize=11)
    ax.set_title(
        "Feature Rank Agreement: XGBoost vs Logistic Regression\n"
        "Top 20 Features by Average Rank  —  Shorter lines = stronger agreement",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig("shap_agreement_dotplot.png", dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    X, y = load_wdbc()
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    lr_pipeline, xgb_pipeline = build_models()
    lr_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    print("Models trained.")

    mean_shap_xgb = compute_mean_abs_shap(xgb_pipeline, X_test, "xgboost")
    mean_shap_lr = compute_mean_abs_shap(lr_pipeline, X_test, "logistic_regression")
    print("SHAP values computed.")

    shap_agreement = create_agreement_table(mean_shap_xgb, mean_shap_lr)
    spearman_corr, spearman_p = report_agreement(shap_agreement)

    plot_bar_comparison(shap_agreement, spearman_corr, spearman_p)
    plot_rank_scatter(shap_agreement, spearman_corr, spearman_p)
    plot_heatmap(shap_agreement, spearman_corr, spearman_p)
    plot_rank_dotplot(shap_agreement)

    print("\n=== Cross-Model SHAP Agreement Analysis Complete ===")


if __name__ == "__main__":
    main()
