"""
paper/figures/generate_paper_figures.py
========================================

Generates every figure used in `paper/research_paper.md` from a single,
centralized data source: `paper/data/experiment_results.json`.

Revision 2: consolidates the original 12-figure set (from the author's
original plotting script) into 5 main-paper figures + 1 appendix figure,
without discarding any of the underlying data -- every original figure
concept is still represented, just grouped into multi-panel figures for a
cleaner academic layout. See the mapping table printed at the end of this
script's output, and the research paper's "Figure Set" note, for the
explicit old-to-new mapping.

This script is standalone: it only depends on matplotlib (no PyTorch
import is required), so it can be re-run purely to regenerate the paper's
figures from the recorded numbers, without a working training environment.

Usage (from the project root):

    python paper/figures/generate_paper_figures.py

All figures are written to `paper/figures/`.

IMPORTANT: this script does not run any experiment and does not compute
any new numbers from models or data. It only visualizes the values already
recorded in `experiment_results.json`. If those values are wrong,
incomplete, or disputed, fix them in that one file -- do not hard-code
corrected numbers directly in this script. Figures built from Tier-C
(script-only, not independently verified) data are explicitly labeled as
such in their titles/captions; see the JSON's `_provenance_tiers` block.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = THIS_DIR.parent / "data" / "experiment_results.json"
OUTPUT_DIR = THIS_DIR

MODEL_LABELS = {
    "resnet18": "ResNet18",
    "mobilenet_v3_small": "MobileNetV3-Small",
}
COLORS = {"resnet18": "#4C72B0", "mobilenet_v3_small": "#DD8452"}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_results() -> dict:
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def _annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _save(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


# --------------------------------------------------------------------------
# Figure 1 (main): Baseline accuracy & macro F1 comparison (2-panel)
# Consolidates original figures #1 (accuracy comparison) and #2 (macro F1
# comparison).
# --------------------------------------------------------------------------


def fig1_baseline_accuracy_f1(results: dict) -> Path:
    baseline = results["baseline"]
    labels = [MODEL_LABELS[k] for k in baseline if not k.startswith("_")]
    keys = [k for k in baseline if not k.startswith("_")]
    colors = [COLORS[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    acc = [baseline[k]["accuracy"] * 100 for k in keys]
    bars = axes[0].bar(labels, acc, color=colors)
    axes[0].set_title("Baseline accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    _annotate_bars(axes[0], bars, "{:.2f}%")

    f1 = [baseline[k]["f1_macro"] * 100 for k in keys]
    bars = axes[1].bar(labels, f1, color=colors)
    axes[1].set_title("Baseline macro F1-score")
    axes[1].set_ylabel("Macro F1-score (%)")
    axes[1].set_ylim(0, 100)
    _annotate_bars(axes[1], bars, "{:.2f}%")

    fig.suptitle("Baseline Classification Performance (CIFAR-10 test subset, n=500)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, "fig1_baseline_accuracy_f1.png")


# --------------------------------------------------------------------------
# Figure 2 (main): Parameter count & serialized model size (2-panel)
# Consolidates original figures #3 (parameter count) and #4 (model size).
# --------------------------------------------------------------------------


def fig2_parameters_and_size(results: dict) -> Path:
    baseline = results["baseline"]
    keys = [k for k in baseline if not k.startswith("_")]
    labels = [MODEL_LABELS[k] for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    params_m = [baseline[k]["total_params"] / 1e6 for k in keys]
    bars = axes[0].bar(labels, params_m, color=colors)
    axes[0].set_title("Total parameter count")
    axes[0].set_ylabel("Parameters (millions)")
    _annotate_bars(axes[0], bars, "{:.2f}M")

    size_mib = [baseline[k]["model_size_mib"] for k in keys]
    bars = axes[1].bar(labels, size_mib, color=colors)
    axes[1].set_title("Serialized model size")
    axes[1].set_ylabel("Serialized model size (MiB)")
    _annotate_bars(axes[1], bars, "{:.2f} MiB")

    fig.suptitle(
        "Model Complexity: Parameter Count and Serialized state_dict() Size (FP32)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, "fig2_parameters_and_size.png")


# --------------------------------------------------------------------------
# Figure 3 (main): Accuracy vs. CPU inference latency (trade-off scatter)
# New figure requested explicitly; was NOT in the original 7-figure set.
# --------------------------------------------------------------------------


def fig3_accuracy_vs_latency(results: dict) -> Path:
    baseline = results["baseline"]
    keys = [k for k in baseline if not k.startswith("_")]

    fig, ax = plt.subplots(figsize=(6, 5))
    for k in keys:
        acc = baseline[k]["accuracy"] * 100
        lat = baseline[k]["latency_ms_batch1"]
        ax.scatter(lat, acc, s=140, color=COLORS[k], label=MODEL_LABELS[k], zorder=3)
        ax.annotate(
            MODEL_LABELS[k],
            (lat, acc),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_title("Accuracy vs. CPU Inference Latency\n(baseline FP32 models, batch=1)")
    ax.set_xlabel("Latency (ms, batch=1)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    return _save(fig, "fig3_accuracy_vs_latency.png")


# --------------------------------------------------------------------------
# Figures 4 & 5 (main): Quantization effects, one figure per model.
# Consolidates original figures #7, #8, #9 (quantization accuracy, size,
# latency) into a single 3-panel figure per model.
# --------------------------------------------------------------------------


def fig_quantization_comparison(results: dict, model_key: str, filename: str, fig_number: int) -> Path:
    q = results["quantization"][model_key]
    original = q["original"]
    optimized = q["optimized_dynamic_int8"]
    model_label = MODEL_LABELS[model_key]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Figure {fig_number}. {model_label}: FP32 Baseline vs. Dynamic INT8 Post-Training Quantization\n"
        f"(quantization targets supported nn.Linear layers only; convolutional layers remain FP32)",
        fontweight="bold",
        fontsize=10,
    )

    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
    x = range(len(metric_keys))
    width = 0.35
    orig_vals = [original[k] * 100 for k in metric_keys]
    opt_vals = [optimized[k] * 100 for k in metric_keys]

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], orig_vals, width, label="Original (FP32)", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], opt_vals, width, label="Optimized (Dynamic INT8)", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Classification metrics")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    bars = ax.bar(
        ["Original\n(FP32)", "Optimized\n(Dynamic INT8)"],
        [original["model_size_mib"], optimized["model_size_mib"]],
        color=["#4C72B0", "#55A868"],
    )
    ax.set_ylabel("Serialized model size (MiB)")
    ax.set_title("Serialized model size")
    _annotate_bars(ax, bars, "{:.3f}")

    ax = axes[2]
    bars = ax.bar(
        ["Original\n(FP32)", "Optimized\n(Dynamic INT8)"],
        [original["latency_ms_batch1"], optimized["latency_ms_batch1"]],
        color=["#4C72B0", "#55A868"],
    )
    ax.set_ylabel("Latency (ms, batch=1)")
    ax.set_title("CPU inference latency\n(optimization-run measurement)")
    _annotate_bars(ax, bars, "{:.2f}")

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _save(fig, filename)


# --------------------------------------------------------------------------
# Figure A1 (appendix): Training curves -- loss, accuracy, macro F1.
# Consolidates original figures #10, #11, #12 (training loss, training
# accuracy, training macro F1) into a single 3-panel appendix figure.
# Tier C provenance (script-only) -- explicitly labeled as such.
# --------------------------------------------------------------------------


def figA1_training_curves(results: dict) -> Path:
    history = results["training_history"]
    keys = [k for k in history if k in MODEL_LABELS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Figure A1 (Appendix). Training Trajectories by Epoch -- Tier C provenance (script-only; see "
        "provenance note)",
        fontweight="bold",
        fontsize=10,
    )

    for k in keys:
        epochs = [row["epoch"] for row in history[k]]
        loss = [row["train_loss"] for row in history[k]]
        acc = [row["accuracy"] * 100 for row in history[k]]
        f1 = [row["f1_macro"] * 100 for row in history[k]]

        axes[0].plot(epochs, loss, marker="o", color=COLORS[k], label=MODEL_LABELS[k])
        axes[1].plot(epochs, acc, marker="o", color=COLORS[k], label=MODEL_LABELS[k])
        axes[2].plot(epochs, f1, marker="o", color=COLORS[k], label=MODEL_LABELS[k])

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_xticks([1, 2])

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks([1, 2])
    axes[1].set_ylim(0, 100)

    axes[2].set_title("Validation macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1-score (%)")
    axes[2].set_xticks([1, 2])
    axes[2].set_ylim(0, 100)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _save(fig, "figA1_training_curves.png")


def main() -> None:
    results = load_results()

    fig1_baseline_accuracy_f1(results)
    fig2_parameters_and_size(results)
    fig3_accuracy_vs_latency(results)
    fig_quantization_comparison(results, "resnet18", "fig4_resnet18_quantization.png", 4)
    fig_quantization_comparison(results, "mobilenet_v3_small", "fig5_mobilenetv3_small_quantization.png", 5)
    figA1_training_curves(results)

    print("\nAll figures generated in:", OUTPUT_DIR)
    print("\nOriginal (12-figure) -> new (6-figure) mapping:")
    print("  #1 Accuracy comparison          -> Figure 1 (left panel)")
    print("  #2 Macro F1 comparison          -> Figure 1 (right panel)")
    print("  #3 Parameter count              -> Figure 2 (left panel)")
    print("  #4 Model size                   -> Figure 2 (right panel)")
    print("  #5 Baseline CPU latency         -> Figure 3 (x-axis) and Figure 2 companion table in paper text")
    print("  #6 Accuracy vs. latency         -> Figure 3 (new, explicit trade-off plot)")
    print("  #7 Quantization accuracy        -> Figures 4 & 5 (left panel, per model)")
    print("  #8 Quantization model size      -> Figures 4 & 5 (middle panel, per model)")
    print("  #9 Quantization latency         -> Figures 4 & 5 (right panel, per model)")
    print("  #10 Training loss               -> Figure A1, appendix (left panel)")
    print("  #11 Training accuracy           -> Figure A1, appendix (middle panel)")
    print("  #12 Training macro F1           -> Figure A1, appendix (right panel)")


if __name__ == "__main__":
    main()