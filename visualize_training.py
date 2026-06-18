"""
Visualize training metrics from training_log.json.

The script supports both older logs that only contain ranking metrics and newer
affine-transition logs that also contain transition scale/shift diagnostics.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRANSITION_METRICS = [
    "transition_scale_mean",
    "transition_scale_std",
    "transition_head_norm_mean",
    "transition_shift_norm_mean",
    "transition_shift_to_head_ratio",
]


def load_training_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def epoch_entries(log_data):
    return [entry for entry in log_data if "epoch" in entry]


def series(entries, key):
    return [
        (entry["epoch"], entry[key])
        for entry in entries
        if entry.get(key) is not None
    ]


def plot_metric(ax, entries, key, label, color=None, marker="o"):
    data = series(entries, key)
    if not data:
        ax.axis("off")
        ax.set_title(f"{label} (not logged)", fontsize=12, fontweight="bold")
        return False

    epochs, values = zip(*data)
    ax.plot(
        epochs,
        values,
        color=color,
        marker=marker,
        markersize=2,
        linewidth=1.5,
        label=label,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    return True


def hide_axes(axes):
    for ax in axes:
        ax.axis("off")


def plot_ranking_panels(axes, entries):
    plot_metric(axes[0], entries, "train_loss", "Training Loss", color="blue")
    plot_metric(axes[1], entries, "val_mrr", "Validation MRR", color="green")
    axes[1].set_ylim(bottom=0)
    plot_metric(axes[2], entries, "val_mr", "Validation Mean Rank", color="red")
    axes[2].set_ylim(bottom=0)
    plot_metric(axes[3], entries, "val_hits1", "Validation Hits@1", color="purple")
    axes[3].set_ylim([0, 1])
    plot_metric(axes[4], entries, "val_hits3", "Validation Hits@3", color="orange")
    axes[4].set_ylim([0, 1])
    plot_metric(axes[5], entries, "val_hits10", "Validation Hits@10", color="brown")
    axes[5].set_ylim([0, 1])

    ax = axes[6]
    for key, label, color, marker in [
        ("val_mrr", "MRR", "green", "o"),
        ("val_hits1", "Hits@1", "purple", "s"),
        ("val_hits3", "Hits@3", "orange", "^"),
        ("val_hits10", "Hits@10", "brown", "x"),
    ]:
        data = series(entries, key)
        if data:
            epochs, values = zip(*data)
            ax.plot(
                epochs,
                values,
                color=color,
                marker=marker,
                markersize=2,
                linewidth=1.5,
                label=label,
            )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("All Ranking Metrics", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_transition_panels(axes, entries):
    ax = axes[7]
    for key, label, color in [
        ("transition_scale_mean", "Scale mean", "navy"),
        ("transition_scale_std", "Scale std", "darkcyan"),
    ]:
        data = series(entries, key)
        if data:
            epochs, values = zip(*data)
            ax.plot(epochs, values, marker="o", markersize=2, label=label, color=color)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Identity")
    ax.set_title("Identity-Centered Scale", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Scale", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[8]
    for key, label, color in [
        ("transition_head_norm_mean", "Head component norm", "navy"),
        ("transition_shift_norm_mean", "Applied shift norm", "darkgreen"),
    ]:
        data = series(entries, key)
        if data:
            epochs, values = zip(*data)
            ax.plot(epochs, values, marker="o", markersize=2, label=label, color=color)
    ax.set_title("Head Component vs Shift", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("L2 norm", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plot_metric(
        axes[9],
        entries,
        "transition_shift_to_head_ratio",
        "Shift / Head Component Ratio",
        color="darkred",
    )
    axes[9].axhline(1.0, color="gray", linestyle="--", linewidth=1)

    ax = axes[10]
    scale_data = dict(series(entries, "transition_scale_mean"))
    ratio_data = dict(series(entries, "transition_shift_to_head_ratio"))
    common_epochs = sorted(set(scale_data) & set(ratio_data))
    if common_epochs:
        ax.plot(
            common_epochs,
            [scale_data[epoch] for epoch in common_epochs],
            color="navy",
            marker="o",
            markersize=2,
            label="Scale mean",
        )
        ax2 = ax.twinx()
        ax2.plot(
            common_epochs,
            [ratio_data[epoch] for epoch in common_epochs],
            color="darkred",
            marker="s",
            markersize=2,
            label="Shift/head ratio",
        )
        ax.set_ylabel("Scale mean", fontsize=11)
        ax2.set_ylabel("Shift/head ratio", fontsize=11)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=9)
    ax.set_title("Scale vs Shift Dominance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.grid(True, alpha=0.3)

    hide_axes(axes[11:12])


def plot_legacy_optional_panels(axes, entries):
    plotted_alpha = plot_metric(
        axes[7],
        entries,
        "train_alpha",
        "Training Gate Weight",
        color="blue",
    )
    if plotted_alpha:
        axes[7].set_ylim([0, 1])
    plot_metric(
        axes[8],
        entries,
        "train_sigreg",
        "Training SIGReg",
        color="teal",
    )


def plot_training_curves(log_data, output_path=None, show=True):
    entries = epoch_entries(log_data)
    if not entries:
        raise ValueError("No per-epoch entries found in the training log.")

    has_transition = any(
        any(entry.get(metric) is not None for metric in TRANSITION_METRICS)
        for entry in entries
    )

    nrows = 4 if has_transition else 3
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    plot_ranking_panels(axes, entries)
    if has_transition:
        plot_transition_panels(axes, entries)
        hide_axes(axes[12:])
    else:
        plot_legacy_optional_panels(axes, entries)
        hide_axes(axes[9:])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_best_metric(entries, key, label, higher_is_better=True):
    data = series(entries, key)
    if not data:
        return
    best = max(data, key=lambda item: item[1]) if higher_is_better else min(data, key=lambda item: item[1])
    print(f"  {label}: {best[1]:.4f} (epoch {best[0]})")


def print_metrics_summary(log_data):
    entries = epoch_entries(log_data)

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTotal epoch entries: {len(entries)}")

    train_losses = [entry["train_loss"] for entry in entries if "train_loss" in entry]
    if train_losses:
        print("\nTraining Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final:   {train_losses[-1]:.4f}")
        print(f"  Best:    {min(train_losses):.4f}")

    print("\nBest Validation Metrics:")
    print_best_metric(entries, "val_mrr", "MRR", higher_is_better=True)
    print_best_metric(entries, "val_hits1", "Hits@1", higher_is_better=True)
    print_best_metric(entries, "val_hits3", "Hits@3", higher_is_better=True)
    print_best_metric(entries, "val_hits10", "Hits@10", higher_is_better=True)
    print_best_metric(entries, "val_mr", "MR", higher_is_better=False)

    has_transition = any(
        any(entry.get(metric) is not None for metric in TRANSITION_METRICS)
        for entry in entries
    )
    if has_transition:
        print("\nTransition Diagnostics At Best MRR:")
        best_entry = max(
            (entry for entry in entries if entry.get("val_mrr") is not None),
            key=lambda entry: entry["val_mrr"],
        )
        for key in TRANSITION_METRICS:
            if best_entry.get(key) is not None:
                print(f"  {key}: {best_entry[key]:.6f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--log_path", type=str, required=True, help="Path to training_log.json file")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization")
    parser.add_argument("--no_show", action="store_true", help="Do not display plot")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    print(f"Loading training log from: {log_path}")
    log_data = load_training_log(log_path)
    print_metrics_summary(log_data)

    output_path = Path(args.output) if args.output else log_path.parent / "training_curves.png"
    plot_training_curves(
        log_data,
        output_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
