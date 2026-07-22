"""Visualize GWM training logs.

Current training logs contain aggregate validation metrics, training loss,
and compact relation-gated context diagnostics. The plotting code remains
tolerant of incomplete runs and older logs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_training_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    if not isinstance(log_data, list):
        raise ValueError("Training log must contain a JSON list of records.")
    return log_data


def epoch_entries(log_data):
    return [entry for entry in log_data if 'epoch' in entry]


def series(entries, key):
    points = []
    for entry in entries:
        value = entry.get(key)
        if value is not None:
            points.append((entry['epoch'], value))
    return points


def plot_lines(ax, entries, specs, title, ylabel, ylim=None):
    plotted = False
    for key, label, style in specs:
        points = series(entries, key)
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, style, markersize=2, linewidth=1.5, label=label)
        plotted = True

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            'No data',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color='gray',
        )


def plot_training_curves(log_data, output_path=None, show=True, run_name=None):
    entries = epoch_entries(log_data)
    if not entries:
        raise ValueError("Training log does not contain epoch entries.")

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.35)
    styles = ['g-o', 'b-s', 'r-^', 'm-d', 'c-x', 'y-*', 'k-p']

    ax1 = fig.add_subplot(grid[0, 0:2])
    plot_lines(
        ax1,
        entries,
        [('train_loss', 'train loss', 'b-')],
        'Training Loss',
        'Loss',
    )

    ax2 = fig.add_subplot(grid[0, 2:4])
    plot_lines(
        ax2,
        entries,
        [('val_mrr', 'validation MRR', 'g-o')],
        'Validation MRR',
        'MRR',
        ylim=(0, 1),
    )

    ax3 = fig.add_subplot(grid[0, 4:6])
    plot_lines(
        ax3,
        entries,
        [
            ('val_hits1', 'Hits@1', 'm-o'),
            ('val_hits3', 'Hits@3', 'c-s'),
            ('val_hits10', 'Hits@10', 'y-^'),
        ],
        'Validation Hits',
        'Score',
        ylim=(0, 1),
    )

    ax4 = fig.add_subplot(grid[1, 0:3])
    plot_lines(
        ax4,
        entries,
        [('val_mr', 'mean rank', 'r-o')],
        'Validation Mean Rank',
        'MR',
    )

    ax5 = fig.add_subplot(grid[1, 3:6])
    plot_lines(
        ax5,
        entries,
        [
            ('context_strength', 'context strength', styles[0]),
            ('context_gate_mean', 'gate mean', styles[1]),
            ('context_gate_std', 'gate std', styles[2]),
        ],
        'Relation-Gated Context',
        'Value',
        ylim=(-1, 1),
    )

    if run_name:
        fig.suptitle(
            f'GWM Training Diagnostics - {run_name}',
            fontsize=15,
            fontweight='bold',
        )
        fig.subplots_adjust(top=0.90)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_metrics_summary(log_data):
    entries = epoch_entries(log_data)
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Epoch entries: {len(entries)}")

    if entries and 'train_loss' in entries[0]:
        losses = [entry['train_loss'] for entry in entries if 'train_loss' in entry]
        best_loss_epoch, best_loss = min(
            ((entry['epoch'], entry['train_loss']) for entry in entries if 'train_loss' in entry),
            key=lambda item: item[1],
        )
        print("\nTraining Loss:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final:   {losses[-1]:.4f}")
        print(f"  Best:    {best_loss:.4f} (epoch {best_loss_epoch})")

    validation_entries = [entry for entry in entries if entry.get('val_mrr') is not None]
    if validation_entries:
        best_validation = max(validation_entries, key=lambda entry: entry['val_mrr'])
        final_validation = validation_entries[-1]
        validation_metrics = [
            ('MRR', 'val_mrr'),
            ('MR', 'val_mr'),
            ('Hits@1', 'val_hits1'),
            ('Hits@3', 'val_hits3'),
            ('Hits@10', 'val_hits10'),
        ]

        print(f"\nBest Validation Checkpoint (epoch {best_validation['epoch']}):")
        for label, key in validation_metrics:
            value = best_validation.get(key)
            if value is not None:
                print(f"  {label:7s}: {value:.4f}")

        print(f"\nFinal Validation Metrics (epoch {final_validation['epoch']}):")
        for label, key in validation_metrics:
            value = final_validation.get(key)
            if value is not None:
                print(f"  {label:7s}: {value:.4f}")

    context_keys = (
        'context_strength',
        'context_gate_mean',
        'context_gate_std',
    )
    if entries and any(key in entries[-1] for key in context_keys):
        print("\nFinal Context Values:")
        final_entry = entries[-1]
        for key in context_keys:
            if key in final_entry:
                label = key.replace('_', ' ')
                print(f"  {label:24s}: {final_entry[key]:.4f}")

    final_events = [entry for entry in log_data if entry.get('event') == 'training_complete']
    if final_events:
        event = final_events[-1]
        print("\nRuntime:")
        print(f"  Total seconds: {event.get('total_train_seconds', 'n/a')}")
        print(f"  Epochs completed: {event.get('epochs_completed', 'n/a')}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize GWM training metrics")
    parser.add_argument('--log_path', type=str, required=True, help='Path to training_log.json')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--no_show', action='store_true', help='Do not display the plot window')
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"Loading training log from: {log_path}")
    log_data = load_training_log(log_path)
    print_metrics_summary(log_data)

    output_path = args.output
    if output_path is None:
        output_path = log_path.parent / 'training_curves.png'

    plot_training_curves(
        log_data,
        output_path=output_path,
        show=not args.no_show,
        run_name=log_path.parent.name,
    )


if __name__ == '__main__':
    main()
