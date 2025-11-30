import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st


# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Clinical Uncertainty Dashboard (HEART)",
    layout="centered",
)


# -------------------------------
# Data loading
# -------------------------------
DATA_DIR = Path("data")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load HEART prediction summary data (mean + CI + label)."""
    path = DATA_DIR / "sample_predictions_heart.csv"
    return pd.read_csv(path)


@st.cache_data
def load_shap() -> pd.DataFrame:
    """Load SHAP summaries for HEART patients."""
    path = DATA_DIR / "sample_shap_summaries.csv"
    return pd.read_csv(path)


# -------------------------------
# Simulation utilities
# -------------------------------
def simulate_bootstrap_preds(
    risk_mean: float,
    ci_low: float,
    ci_high: float,
    n: int = 200,
) -> np.ndarray:
    """
    Simulate bootstrap predictions roughly consistent with the CI.
    Used for dotplot-style uncertainty views.
    """
    sigma = (ci_high - ci_low) / (2 * 1.96 + 1e-8)
    if sigma <= 0:
        sigma = 0.01
    samples = np.random.normal(loc=risk_mean, scale=sigma, size=n)
    samples = np.clip(samples, 0.0, 1.0)
    return samples


# -------------------------------
# Matplotlib helpers
# -------------------------------
def make_risk_bar(ax, risk_mean, ci_low, ci_high, label: str) -> None:
    """Top panel: risk spectrum bar with CI band and mean marker."""
    bar_y = 0.3
    bar_h = 0.25

    # low / medium / high background
    ax.add_patch(
        Rectangle(
            (0.0, bar_y),
            0.33,
            bar_h,
            edgecolor="none",
            facecolor="#d0f0d0",
        )
    )
    ax.add_patch(
        Rectangle(
            (0.33, bar_y),
            0.33,
            bar_h,
            edgecolor="none",
            facecolor="#fff3b0",
        )
    )
    ax.add_patch(
        Rectangle(
            (0.66, bar_y),
            0.34,
            bar_h,
            edgecolor="none",
            facecolor="#f4b2b0",
        )
    )

    # CI band
    band_width = ci_high - ci_low
    ax.add_patch(
        Rectangle(
            (ci_low, bar_y),
            band_width,
            bar_h,
            edgecolor="none",
            facecolor="gray",
            alpha=0.5,
        )
    )

    # mean marker
    ax.axvline(
        risk_mean,
        ymin=bar_y,
        ymax=bar_y + bar_h,
        color="black",
        linewidth=1.5,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Predicted risk")
    ax.get_yaxis().set_visible(False)

    ax.text(0.165, bar_y - 0.05, "Low", ha="center", va="center", fontsize=8)
    ax.text(0.495, bar_y - 0.05, "Medium", ha="center", va="center", fontsize=8)
    ax.text(0.83, bar_y - 0.05, "High", ha="center", va="center", fontsize=8)

    ax.set_title(f"Risk summary – {label}", fontsize=9)


def make_prediction_dotplot(ax, samples, risk_mean, ci_low, ci_high) -> None:
    """Middle panel: dotplot of prediction samples."""
    samples = np.sort(samples)
    n_bins = 20
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        bin_low = bins[i]
        bin_high = bins[i + 1]
        in_bin = samples[(samples >= bin_low) & (samples < bin_high)]
        k = len(in_bin)
        if k == 0:
            continue
        xs = in_bin
        ys = np.linspace(-0.4, 0.4, k)
        ax.scatter(xs, ys, s=8, alpha=0.8)

    ax.axvline(risk_mean, color="black", linewidth=1)
    ax.axvline(ci_low, color="gray", linestyle="--", linewidth=1)
    ax.axvline(ci_high, color="gray", linestyle="--", linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Predicted risk")
    ax.set_title("Prediction uncertainty (bootstrap samples)", fontsize=9)


def make_shap_bar(ax, shap_subset: pd.DataFrame) -> None:
    """Bottom panel: SHAP feature contributions."""
    df = shap_subset.copy()
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=True)

    features = df["feature"].tolist()
    shap_values = df["shap_value"].tolist()
    y_pos = np.arange(len(features))

    ax.barh(y_pos, shap_values, color="#8888ff")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel("SHAP value")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Feature contributions", fontsize=9)


def render_single_dashboard(pred_row: pd.Series, shap_df: pd.DataFrame):
    """
    Create the full 3-row dashboard figure (Prototype 4)
    for a single patient.
    """
    risk_mean = float(pred_row["risk_mean"])
    ci_low = float(pred_row["ci_low"])
    ci_high = float(pred_row["ci_high"])
    label = str(pred_row["label"])
    pid = int(pred_row["patient_id"])

    shap_subset = shap_df[shap_df["patient_id"] == pid]
    boot_preds = simulate_bootstrap_preds(risk_mean, ci_low, ci_high, n=200)

    fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [0.9, 1.1, 1.1]},
    )

    make_risk_bar(ax_top, risk_mean, ci_low, ci_high, label)
    make_prediction_dotplot(ax_mid, boot_preds, risk_mean, ci_low, ci_high)
    make_shap_bar(ax_bottom, shap_subset)

    fig.suptitle(
        f"HEART – {label} | risk={risk_mean:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_small_interval(ax, risk_mean, ci_low, ci_high, label: str) -> None:
    """Tiny interval plot for comparison view (Prototype 5)."""
    ax.hlines(y=0, xmin=ci_low, xmax=ci_high, color="black", linewidth=2)
    ax.plot(risk_mean, 0, marker="o", color="black", markersize=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1.0"], fontsize=7)
    ax.set_title(label, fontsize=9)
    ax.axvline(0.5, color="lightgray", linestyle="--", linewidth=1)


def plot_small_shap(ax, shap_subset: pd.DataFrame, max_features: int = 3) -> None:
    """Tiny SHAP bar chart for comparison view (top-k features)."""
    df = shap_subset.copy()
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=True).tail(max_features)

    features = df["feature"].tolist()
    shap_vals = df["shap_value"].tolist()
    y_pos = np.arange(len(features))

    ax.barh(y_pos, shap_vals, color="#8888ff")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=7)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("SHAP", fontsize=7)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(7)


def render_comparison_view(preds: pd.DataFrame, shap_df: pd.DataFrame):
    """
    Render Prototype 5: comparison view across the three archetype patients.
    """
    # Sort so columns appear LOW -> MID -> HIGH
    preds_sorted = preds.sort_values("risk_mean")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(preds_sorted),
        figsize=(9, 4.5),
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )

    top_axes = axes[0]
    bottom_axes = axes[1]

    for col_idx, (_, row) in enumerate(preds_sorted.iterrows()):
        risk_mean = float(row["risk_mean"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        label = str(row["label"])
        pid = int(row["patient_id"])

        shap_subset = shap_df[shap_df["patient_id"] == pid]

        # Top row: CI interval
        plot_small_interval(top_axes[col_idx], risk_mean, ci_low, ci_high, label)

        # Bottom row: SHAP
        plot_small_shap(bottom_axes[col_idx], shap_subset, max_features=3)

        # Semantic labels under each column
        semantic_label = {
            "LOW_tight": "Confident low",
            "MID_wide": "Uncertain mid",
            "HIGH_tight": "Confident high",
        }.get(label, "")

        bottom_axes[col_idx].text(
            0.5,
            -0.35,
            semantic_label,
            transform=bottom_axes[col_idx].transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

    top_axes[0].set_ylabel("Uncertainty", fontsize=8)
    bottom_axes[0].set_ylabel("Features", fontsize=8)

    fig.suptitle(
        "HEART – Comparative view across archetype patients",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Clinical Uncertainty + Explanation Dashboard (HEART)")

st.markdown(
    """
This app demonstrates a **clinical AI design space** for heart disease risk:

- A **single-patient dashboard** that combines risk, uncertainty, and explanations.
- A **comparison view** across three archetype patients:
  - **LOW_tight** – confident low risk  
  - **MID_wide** – uncertain mid risk  
  - **HIGH_tight** – confident high risk  
"""
)

preds = load_predictions()
shap_df = load_shap()

# Sidebar selection for single-patient view
labels = preds["label"].tolist()
default_index = labels.index("MID_wide") if "MID_wide" in labels else 0

selected_label = st.sidebar.selectbox(
    "Select patient archetype (for single-patient view)",
    options=labels,
    index=default_index,
    format_func=lambda x: {
        "LOW_tight": "LOW_tight – Confident low",
        "MID_wide": "MID_wide – Uncertain mid",
        "HIGH_tight": "HIGH_tight – Confident high",
    }.get(x, x),
)

selected_row = preds[preds["label"] == selected_label].iloc[0]
risk_mean = float(selected_row["risk_mean"])
ci_low = float(selected_row["ci_low"])
ci_high = float(selected_row["ci_high"])

# Tabs: single-patient vs comparison
tab_single, tab_compare = st.tabs(["Single patient dashboard", "Comparison view"])

with tab_single:
    st.subheader("Risk summary (numeric)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk", f"{risk_mean:.3f}")
    c2.metric("CI low", f"{ci_low:.3f}")
    c3.metric("CI high", f"{ci_high:.3f}")

    fig_single = render_single_dashboard(selected_row, shap_df)
    st.pyplot(fig_single)

with tab_compare:
    st.subheader("Compare archetype patients")
    st.markdown(
        "Side-by-side CI ranges and top SHAP features for "
        "**LOW_tight**, **MID_wide**, and **HIGH_tight**."
    )
    fig_compare = render_comparison_view(preds, shap_df)
    st.pyplot(fig_compare)

# --- About this prototype section ---
st.markdown(
    """
    ---
    ### ℹ️ About this prototype

    This dashboard illustrates **three layers of uncertainty and explanation**:
    
    - **Risk Spectrum (top)** shows where the predicted risk falls across low/medium/high zones, along with a 95% CI band.  
    - **Prediction Uncertainty (middle)** visualizes bootstrap samples to show how spread out the model’s predictions are.  
    - **Feature Contributions (bottom)** highlights the SHAP values explaining which features most increase or decrease risk.

    Together, these views provide a compact, interpretable summary of a single patient’s model prediction.
    """
)
