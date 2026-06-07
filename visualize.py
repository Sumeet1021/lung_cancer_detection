"""
visualize.py
============
All model-performance and prediction visualisations for the
Hybrid Lung Cancer Detection project.

Usage
-----
# After training
from visualize import (
    plot_accuracy, plot_loss, plot_confusion_matrix,
    plot_class_distribution, plot_prediction_probs, plot_roc_curve
)

plot_accuracy(history)
plot_loss(history)
plot_confusion_matrix(y_true, y_pred_classes)
plot_class_distribution("dataset/images")
plot_prediction_probs(probs, class_names)
plot_roc_curve(y_true_bin, y_probs, class_names)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Flask)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Output directory ───────────────────────────────────────────────
STATIC_DIR = os.path.join("static", "graphs")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────
PALETTE   = ["#1a56db", "#0ea5e9", "#6366f1", "#f43f5e", "#10b981"]
BG_COLOR  = "#f8fafc"
GRID_CLR  = "#e2e8f0"
TEXT_CLR  = "#0f172a"
FONT_MAIN = "DejaVu Sans"

def _base_style():
    """Apply a consistent medical-dashboard style to every figure."""
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    "#ffffff",
        "axes.edgecolor":    GRID_CLR,
        "axes.labelcolor":   TEXT_CLR,
        "axes.titlecolor":   TEXT_CLR,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlepad":     12,
        "axes.labelsize":    10,
        "axes.labelpad":     8,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.linewidth":    0.8,
        "xtick.color":       TEXT_CLR,
        "ytick.color":       TEXT_CLR,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.fontsize":   9,
        "font.family":       FONT_MAIN,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": BG_COLOR,
    })

def _save(fig, filename):
    path = os.path.join(STATIC_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# 1. Training Accuracy vs Validation Accuracy
# ══════════════════════════════════════════════════════════════════
def plot_accuracy(history, filename="accuracy.png"):
    """
    Parameters
    ----------
    history : keras History object  (returned by model.fit())
    """
    _base_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    epochs = range(1, len(history.history["accuracy"]) + 1)
    train_acc = history.history["accuracy"]
    val_acc   = history.history["val_accuracy"]

    ax.plot(epochs, train_acc, color=PALETTE[0], linewidth=2.2,
            marker="o", markersize=4, label="Training Accuracy")
    ax.plot(epochs, val_acc,   color=PALETTE[2], linewidth=2.2,
            marker="s", markersize=4, linestyle="--", label="Validation Accuracy")

    # Highlight best val epoch
    best_ep  = int(np.argmax(val_acc)) + 1
    best_val = max(val_acc)
    ax.axvline(best_ep, color=PALETTE[3], linewidth=1.2,
               linestyle=":", alpha=0.8, label=f"Best Val Epoch ({best_ep})")
    ax.annotate(f"  Best: {best_val:.4f}",
                xy=(best_ep, best_val),
                xytext=(best_ep + 0.5, best_val - 0.04),
                fontsize=8.5, color=PALETTE[3],
                arrowprops=dict(arrowstyle="->", color=PALETTE[3], lw=1.2))

    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# 2. Training Loss vs Validation Loss
# ══════════════════════════════════════════════════════════════════
def plot_loss(history, filename="loss.png"):
    _base_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    epochs    = range(1, len(history.history["loss"]) + 1)
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]

    ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=2.2,
            marker="o", markersize=4, label="Training Loss")
    ax.plot(epochs, val_loss,   color=PALETTE[3], linewidth=2.2,
            marker="s", markersize=4, linestyle="--", label="Validation Loss")

    best_ep  = int(np.argmin(val_loss)) + 1
    best_val = min(val_loss)
    ax.axvline(best_ep, color=PALETTE[2], linewidth=1.2,
               linestyle=":", alpha=0.8, label=f"Best Val Epoch ({best_ep})")
    ax.annotate(f"  Min: {best_val:.4f}",
                xy=(best_ep, best_val),
                xytext=(best_ep + 0.5, best_val + 0.03),
                fontsize=8.5, color=PALETTE[2],
                arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=1.2))

    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# 3. Confusion Matrix Heatmap
# ══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred,
                          class_names=("Benign", "Malignant", "Normal"),
                          filename="confusion_matrix.png"):
    """
    Parameters
    ----------
    y_true : array-like of int  (ground-truth class indices)
    y_pred : array-like of int  (predicted class indices)
    """
    _base_style()
    cm = confusion_matrix(y_true, y_pred)

    # Row-normalised (percentage)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = sns.light_palette(PALETTE[0], as_cmap=True)

    sns.heatmap(cm_norm, annot=False, fmt=".2%", cmap=cmap,
                linewidths=0.5, linecolor=GRID_CLR,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"shrink": 0.8})

    # Annotate each cell with count + percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct  = cm_norm[i, j]
            cnt  = cm[i, j]
            clr  = "white" if pct > 0.55 else TEXT_CLR
            ax.text(j + 0.5, i + 0.5,
                    f"{cnt}\n({pct:.1%})",
                    ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=clr)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# 4. Class Distribution Bar Graph
# ══════════════════════════════════════════════════════════════════
def plot_class_distribution(dataset_dir="dataset/images",
                            filename="class_distribution.png"):
    _base_style()

    if not os.path.isdir(dataset_dir):
        print(f"  [!] Dataset directory not found: {dataset_dir}")
        return None

    classes = []
    counts  = []
    for cls in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        n = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        classes.append(cls)
        counts.append(n)

    colors = PALETTE[:len(classes)]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(classes, counts, color=colors,
                  width=0.5, edgecolor="white", linewidth=1.5, zorder=3)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_CLR)

    total = sum(counts)
    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.1f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save(fig, filename)
    # Value labels on top of bars
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_CLR)

    # Percentage labels inside bars
    total = sum(counts)
    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.1f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# 5. Prediction Probability Bar Graph  (single image)
# ══════════════════════════════════════════════════════════════════
def plot_prediction_probs(probs,
                          class_names=("Benign", "Malignant", "Normal"),
                          filename="prediction_probs.png"):
    """
    Parameters
    ----------
    probs        : 1-D array/list  e.g. [0.02, 0.91, 0.07]
    class_names  : sequence of strings
    """
    _base_style()
    probs = np.array(probs).flatten()
    pred_idx = int(np.argmax(probs))

    bar_colors = [PALETTE[3] if i == pred_idx else PALETTE[1]
                  for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(class_names, probs * 100,
                   color=bar_colors, edgecolor="white",
                   linewidth=1.2, height=0.5, zorder=3)

    # Percentage labels at end of bars
    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.2f}%", va="center",
                fontsize=10, fontweight="bold", color=TEXT_CLR)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Probability per Class")
    ax.invert_yaxis()

    # Legend
    predicted_patch = mpatches.Patch(color=PALETTE[3], label="Predicted Class")
    other_patch     = mpatches.Patch(color=PALETTE[1], label="Other Classes")
    ax.legend(handles=[predicted_patch, other_patch], loc="lower right")

    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# 6. ROC Curve  (one-vs-rest, multi-class)
# ══════════════════════════════════════════════════════════════════
def plot_roc_curve(y_true, y_probs,
                   class_names=("Benign", "Malignant", "Normal"),
                   filename="roc_curve.png"):
    """
    Parameters
    ----------
    y_true  : 1-D int array  (class indices,  e.g. [0, 2, 1, ...])
    y_probs : 2-D float array  shape (n_samples, n_classes)
              e.g. model.predict(X_test)
    """
    _base_style()
    n_classes  = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    auc_scores = []
    for i, (cls, color) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color=GRID_CLR, linewidth=1.5,
            linestyle="--", label="Random Classifier")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true_bin.ravel(), y_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro,
            color=TEXT_CLR, linewidth=2, linestyle="-.",
            label=f"Micro-avg  (AUC = {auc_micro:.3f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — One-vs-Rest (Multi-class)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════
# Convenience: generate ALL graphs at once after training
# ══════════════════════════════════════════════════════════════════
def generate_all_training_graphs(history, y_true, y_pred_classes,
                                 y_probs, dataset_dir="dataset/images",
                                 class_names=("Benign", "Malignant", "Normal")):
    """
    Call once after training to produce every graph.

    Parameters
    ----------
    history         : Keras History object
    y_true          : ground-truth class indices for test set
    y_pred_classes  : predicted class indices for test set
    y_probs         : raw softmax output, shape (n, 3)
    dataset_dir     : path to dataset root
    class_names     : tuple/list of class labels
    """
    print("\n── Generating visualisations ───────────────────────────")
    plot_accuracy(history)
    plot_loss(history)
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    plot_class_distribution(dataset_dir)
    plot_roc_curve(y_true, y_probs, class_names)
    print("── All graphs saved to static/graphs/ ──────────────────\n")