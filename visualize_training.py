"""
Visualize current GWM training logs.

The script is intentionally tolerant of missing fields so it can read older
training_log.json files as well as the current logs with bidirectional metrics
and GatedFusion gate statistics.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_training_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def gate_label(key):
    label = key
    if label.startswith('train_'):
        label = label[len('train_'):]
    return label.replace('_gate', '').replace('_', ' ')


def gate_specs(entries, styles):
    preferred = [
        'train_head_gate',
        'train_relation_gate',
        'train_context_entity_gate',
        'train_context_relation_gate',
        'train_target_gate',
    ]
    available = {
        key
        for entry in entries
        for key in entry
        if key.startswith('train_') and key.endswith('_gate')
    }
    keys = [key for key in preferred if key in available]
    return [
        (key, gate_label(key), styles[idx % len(styles)])
        for idx, key in enumerate(keys)
    ]


def plot_training_curves(log_data, output_path=None, show=True):
    entries = epoch_entries(log_data)
    if not entries:
        raise ValueError("Training log does not contain epoch entries.")

    fig = plt.figure(figsize=(18, 14))
    styles = ['g-o', 'b-s', 'r-^', 'm-d', 'c-x', 'y-*', 'k-p']

    ax1 = plt.subplot(3, 3, 1)
    plot_lines(
        ax1,
        entries,
        [('train_loss', 'train loss', 'b-')],
        'Training Loss',
        'Loss',
    )

    ax2 = plt.subplot(3, 3, 2)
    plot_lines(
        ax2,
        entries,
        [
            ('val_mrr', 'direction avg', 'g-o'),
            ('val_forward_mrr', 'forward', 'b-s'),
            ('val_backward_mrr', 'backward', 'r-^'),
            ('val_micro_mrr', 'micro', 'k--'),
        ],
        'Validation MRR',
        'MRR',
        ylim=(0, 1),
    )

    ax3 = plt.subplot(3, 3, 3)
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

    ax4 = plt.subplot(3, 3, 4)
    plot_lines(
        ax4,
        entries,
        [
            ('val_forward_hits10', 'forward Hits@10', 'b-s'),
            ('val_backward_hits10', 'backward Hits@10', 'r-^'),
            ('val_micro_hits10', 'micro Hits@10', 'k--'),
        ],
        'Directional Hits@10',
        'Hits@10',
        ylim=(0, 1),
    )

    ax5 = plt.subplot(3, 3, 5)
    plot_lines(
        ax5,
        entries,
        [('val_mr', 'mean rank', 'r-o')],
        'Validation Mean Rank',
        'MR',
    )

    ax6 = plt.subplot(3, 3, 6)
    plot_lines(
        ax6,
        entries,
        gate_specs(entries, styles),
        'GatedFusion Gate Values',
        'Mean gate',
        ylim=(0, 1),
    )

    ax7 = plt.subplot(3, 3, 7)
    plot_lines(
        ax7,
        entries,
        [
            ('avg_filtered_truths_per_query', 'filtered truths/query', 'b-o'),
            ('filtered_query_rate', 'rows with filtered truths', 'g-s'),
        ],
        'Filtered-Negative Diagnostics',
        'Value',
    )

    ax8 = plt.subplot(3, 3, 8)
    plot_lines(
        ax8,
        entries,
        [
            ('val_forward_mrr', 'forward MRR', 'b-s'),
            ('val_backward_mrr', 'backward MRR', 'r-^'),
        ],
        'Forward vs Backward MRR',
        'MRR',
        ylim=(0, 1),
    )

    ax9 = plt.subplot(3, 3, 9)
    plot_lines(
        ax9,
        entries,
        [
            ('avg_filtered_truths_per_query', 'filtered truths/query', 'b-o'),
            ('filtered_query_rate', 'rows with filtered truths', 'g-s'),
        ],
        'Filtered-Negative Diagnostics',
        'Value',
    )

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def best_epoch(entries, key, maximize=True):
    points = series(entries, key)
    if not points:
        return None
    return max(points, key=lambda item: item[1]) if maximize else min(points, key=lambda item: item[1])


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

    best_mrr = best_epoch(entries, 'val_mrr', maximize=True)
    if best_mrr:
        print("\nValidation MRR:")
        print(f"  Best direction-avg: {best_mrr[1]:.4f} (epoch {best_mrr[0]})")
        forward = series(entries, 'val_forward_mrr')
        backward = series(entries, 'val_backward_mrr')
        if forward and backward:
            print(f"  Final forward:     {forward[-1][1]:.4f}")
            print(f"  Final backward:    {backward[-1][1]:.4f}")

    gate_mean_keys = sorted({
        key
        for entry in entries
        for key in entry
        if key.startswith('train_') and key.endswith('_gate')
    })
    if gate_mean_keys:
        print("\nFinal Gate Values:")
        final_entry = entries[-1]
        for key in gate_mean_keys:
            if key in final_entry:
                print(f"  {gate_label(key):24s}: {final_entry[key]:.4f}")

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
    )


if __name__ == '__main__':
    main()
