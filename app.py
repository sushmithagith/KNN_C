import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="KNN Visual Dashboard", layout="wide")
st.title("KNN Classification — Visual Dashboard")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Model Parameters")
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
test_size   = st.sidebar.slider("Test Size (%)", 10, 50, 20)
scale_data  = st.sidebar.checkbox("Apply StandardScaler", value=True)

# ── Data & Model ──────────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=1000, n_features=4, n_redundant=2,
    n_classes=2, random_state=42
)

if scale_data:
    X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, output_dict=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#0f1117"
CARD = "#1c1f2b"
ACC  = "#4f8ef7"
GRN  = "#43d9a0"
RED  = "#f76f6f"
YEL  = "#f7c948"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor"  : CARD,
    "axes.edgecolor"  : "#2e3350",
    "axes.labelcolor" : "#c8cde0",
    "xtick.color"     : "#8b92b3",
    "ytick.color"     : "#8b92b3",
    "text.color"      : "#c8cde0",
    "grid.color"      : "#2e3350",
    "grid.linestyle"  : "--",
    "grid.alpha"      : 0.6,
})

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — Metric cards (accuracy, precision, recall, F1)
# ═════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
precision = report["weighted avg"]["precision"]
recall    = report["weighted avg"]["recall"]
f1        = report["weighted avg"]["f1-score"]

for col, label, val, colour in zip(
    [c1, c2, c3, c4],
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [accuracy, precision, recall, f1],
    [ACC, GRN, YEL, RED],
):
    col.markdown(
        f"""
        <div style="background:{CARD};border-left:4px solid {colour};
                    border-radius:8px;padding:16px 20px;text-align:center;">
            <p style="margin:0;font-size:13px;color:#8b92b3;">{label}</p>
            <p style="margin:4px 0 0;font-size:32px;font-weight:700;color:{colour};">
                {val:.2%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — Confusion Matrix  |  Per-class Bar Chart
# ═════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

# ── Confusion Matrix heatmap ──────────────────────────────────────────────────
with col_a:
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="#2e3350",
        annot_kws={"size": 18, "weight": "bold"},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticklabels(["Class 0", "Class 1"], fontsize=10)
    ax.set_yticklabels(["Class 0", "Class 1"], fontsize=10, rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Per-class Precision / Recall / F1 bars ────────────────────────────────────
with col_b:
    st.markdown("#### Per-Class Metrics")
    classes   = ["Class 0", "Class 1"]
    metrics   = ["precision", "recall", "f1-score"]
    m_colours = [ACC, GRN, YEL]
    x         = np.arange(len(classes))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, (metric, colour) in enumerate(zip(metrics, m_colours)):
        vals = [report[str(c)][metric] for c in [0, 1]]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=colour, alpha=0.85, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — K vs Accuracy curve  |  Prediction Confidence distribution
# ═════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2)

# ── K vs Accuracy ─────────────────────────────────────────────────────────────
with col_c:
    st.markdown("#### K vs Accuracy")
    k_range  = range(1, 21)
    k_scores = [
        KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                                            .score(X_test, y_test)
        for k in k_range
    ]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(k_range, k_scores, color=ACC, linewidth=2.5, marker="o",
            markersize=5, markerfacecolor=GRN)
    ax.axvline(n_neighbors, color=RED, linestyle="--", linewidth=1.5,
               label=f"Current k={n_neighbors}")
    ax.set_xlabel("k (Number of Neighbors)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_xticks(list(k_range))
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Prediction Confidence histogram ───────────────────────────────────────────
with col_d:
    st.markdown("#### Prediction Confidence Distribution")
    correct   = y_proba[y_pred == y_test]
    incorrect = y_proba[y_pred != y_test]
    fig, ax   = plt.subplots(figsize=(5, 3.5))
    ax.hist(correct,   bins=20, color=GRN, alpha=0.75, label="Correct",   edgecolor="none")
    ax.hist(incorrect, bins=20, color=RED, alpha=0.75, label="Incorrect", edgecolor="none")
    ax.set_xlabel("Predicted Probability (Class 1)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 4 — Cross-Validation scores  |  Feature scatter (PC1 vs PC2)
# ═════════════════════════════════════════════════════════════════════════════
col_e, col_f = st.columns(2)

# ── Cross-validation bar chart ────────────────────────────────────────────────
with col_e:
    st.markdown("#### 5-Fold Cross-Validation Scores")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    fig, ax   = plt.subplots(figsize=(5, 3.5))
    fold_colours = [GRN if s >= cv_scores.mean() else ACC for s in cv_scores]
    bars = ax.bar(
        [f"Fold {i+1}" for i in range(5)],
        cv_scores, color=fold_colours, alpha=0.85, edgecolor="none", width=0.5
    )
    ax.axhline(cv_scores.mean(), color=YEL, linestyle="--", linewidth=1.8,
               label=f"Mean {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    for bar, v in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(cv_scores.min() - 0.05, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Feature scatter coloured by prediction correctness ────────────────────────
with col_f:
    st.markdown("#### Feature Space: Correct vs Incorrect Predictions")
    correct_mask = (y_pred == y_test)
    fig, ax      = plt.subplots(figsize=(5, 3.5))
    ax.scatter(
        X_test[correct_mask, 0],  X_test[correct_mask, 1],
        c=GRN, alpha=0.6, s=18, label="Correct", edgecolors="none"
    )
    ax.scatter(
        X_test[~correct_mask, 0], X_test[~correct_mask, 1],
        c=RED, alpha=0.9, s=30, marker="x", label="Incorrect", linewidths=1.2
    )
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)