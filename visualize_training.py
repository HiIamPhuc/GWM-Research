"""
Visualize training metrics from training_log.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_path):
    """Load training log from JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_training_curves(log_data, output_path=None):
    """Create comprehensive training visualization."""
    
    epochs = [entry['epoch'] for entry in log_data if 'epoch' in entry]
    train_loss = [entry['train_loss'] for entry in log_data if 'train_loss' in entry]
    
    val_mrr = [entry.get('val_mrr', None) for entry in log_data]
    val_mr = [entry.get('val_mr', None) for entry in log_data]
    val_hits1 = [entry.get('val_hits1', None) for entry in log_data]
    val_hits3 = [entry.get('val_hits3', None) for entry in log_data]
    val_hits10 = [entry.get('val_hits10', None) for entry in log_data]
    
    train_alpha = [entry.get('train_alpha', None) for entry in log_data]
    val_alpha = [entry.get('val_alpha', None) for entry in log_data]
    train_sigreg = [entry.get('train_sigreg', None) for entry in log_data]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss Plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, train_loss, 'b', markersize=1, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. MRR Plot
    ax2 = plt.subplot(3, 3, 2)
    valid_mrr = [(e, m) for e, m in zip(epochs, val_mrr) if m is not None]
    if valid_mrr:
        e_vals, m_vals = zip(*valid_mrr)
        ax2.plot(e_vals, m_vals, 'g-o', markersize=1, label='Val MRR')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('MRR', fontsize=11)
        ax2.set_title('Validation MRR (Higher is Better)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(bottom=0)
    
    # 3. Mean Rank Plot
    ax3 = plt.subplot(3, 3, 3)
    valid_mr = [(e, m) for e, m in zip(epochs, val_mr) if m is not None]
    if valid_mr:
        e_vals, m_vals = zip(*valid_mr)
        ax3.plot(e_vals, m_vals, 'r-o', markersize=1, label='Val MR')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Mean Rank', fontsize=11)
        ax3.set_title('Validation Mean Rank (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(bottom=0)
    
    # 4. Hits@1 Plot
    ax4 = plt.subplot(3, 3, 4)
    valid_h1 = [(e, h) for e, h in zip(epochs, val_hits1) if h is not None]
    if valid_h1:
        e_vals, h_vals = zip(*valid_h1)
        ax4.plot(e_vals, h_vals, 'purple', marker='o', markersize=1, label='Hits@1')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Hits@1', fontsize=11)
        ax4.set_title('Validation Hits@1', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([0, 1])
    
    # 5. Hits@3 Plot
    ax5 = plt.subplot(3, 3, 5)
    valid_h3 = [(e, h) for e, h in zip(epochs, val_hits3) if h is not None]
    if valid_h3:
        e_vals, h_vals = zip(*valid_h3)
        ax5.plot(e_vals, h_vals, 'orange', marker='o', markersize=1, label='Hits@3')
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Hits@3', fontsize=11)
        ax5.set_title('Validation Hits@3', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_ylim([0, 1])
    
    # 6. Hits@10 Plot
    ax6 = plt.subplot(3, 3, 6)
    valid_h10 = [(e, h) for e, h in zip(epochs, val_hits10) if h is not None]
    if valid_h10:
        e_vals, h_vals = zip(*valid_h10)
        ax6.plot(e_vals, h_vals, 'brown', marker='o', markersize=1, label='Hits@10')
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Hits@10', fontsize=11)
        ax6.set_title('Validation Hits@10', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.set_ylim([0, 1])
    
    # 7. All Metrics Together
    ax7 = plt.subplot(3, 3, 7)
    valid_mrr_data = [(e, m) for e, m in zip(epochs, val_mrr) if m is not None]
    valid_h1_data = [(e, h) for e, h in zip(epochs, val_hits1) if h is not None]
    valid_h3_data = [(e, h) for e, h in zip(epochs, val_hits3) if h is not None]
    valid_h10_data = [(e, h) for e, h in zip(epochs, val_hits10) if h is not None]
    
    if valid_mrr_data:
        e_vals, m_vals = zip(*valid_mrr_data)
        ax7.plot(e_vals, m_vals, 'g-o', markersize=1, label='MRR')
    if valid_h1_data:
        e_vals, h_vals = zip(*valid_h1_data)
        ax7.plot(e_vals, h_vals, 'purple', marker='s', markersize=1, label='Hits@1')
    if valid_h3_data:
        e_vals, h_vals = zip(*valid_h3_data)
        ax7.plot(e_vals, h_vals, 'orange', marker='^', markersize=1, label='Hits@3')
    if valid_h10_data:
        e_vals, h_vals = zip(*valid_h10_data)
        ax7.plot(e_vals, h_vals, 'brown', marker='x', markersize=1, label='Hits@10')
    
    ax7.set_xlabel('Epoch', fontsize=11)
    ax7.set_ylabel('Score', fontsize=11)
    ax7.set_title('All Ranking Metrics', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1])
    
    # 8. Text Weight (Alpha) - Train
    ax8 = plt.subplot(3, 3, 8)
    valid_train_alpha = [(e, a) for e, a in zip(epochs, train_alpha) if a is not None]
    if valid_train_alpha:
        e_vals, a_vals = zip(*valid_train_alpha)
        ax8.plot(e_vals, a_vals, 'b-o', markersize=1, label='Train Alpha')
        ax8.set_xlabel('Epoch', fontsize=11)
        ax8.set_ylabel('Alpha (Text Weight)', fontsize=11)
        ax8.set_title('Training Gate Weight (α)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        ax8.set_ylim([0, 1])

    # 9. SIGReg (Train)
    ax9 = plt.subplot(3, 3, 9)
    valid_sigreg = [(e, s) for e, s in zip(epochs, train_sigreg) if s is not None]
    if valid_sigreg:
        e_vals, s_vals = zip(*valid_sigreg)
        ax9.plot(e_vals, s_vals, 'teal', marker='o', markersize=1, label='Train SIGReg')
        ax9.set_xlabel('Epoch', fontsize=11)
        ax9.set_ylabel('SIGReg', fontsize=11)
        ax9.set_title('Training SIGReg', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
    
    plt.show()


def print_metrics_summary(log_data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nTotal Epochs: {len(log_data)}")
    
    # Training Loss
    train_losses = [entry['train_loss'] for entry in log_data if 'train_loss' in entry]
    print(f"\nTraining Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final:   {train_losses[-1]:.4f}")
    print(f"  Best:    {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses)) + 1})")
    
    # Validation Metrics
    val_mrr = [entry.get('val_mrr', None) for entry in log_data]
    valid_mrr = [m for m in val_mrr if m is not None]
    
    if valid_mrr:
        print(f"\nValidation MRR:")
        print(f"  Initial: {valid_mrr[0]:.4f}")
        print(f"  Final:   {valid_mrr[-1]:.4f}")
        print(f"  Best:    {max(valid_mrr):.4f} (Epoch {valid_mrr.index(max(valid_mrr)) + 1})")
        print(f"  Improvement: {(valid_mrr[-1] - valid_mrr[0]) / valid_mrr[0] * 100:.1f}%")
    
    # Hits@10
    val_h10 = [entry.get('val_hits10', None) for entry in log_data]
    valid_h10 = [h for h in val_h10 if h is not None]
    
    if valid_h10:
        print(f"\nValidation Hits@10:")
        print(f"  Initial: {valid_h10[0]:.4f} ({valid_h10[0]*100:.2f}%)")
        print(f"  Final:   {valid_h10[-1]:.4f} ({valid_h10[-1]*100:.2f}%)")
        print(f"  Best:    {max(valid_h10):.4f} ({max(valid_h10)*100:.2f}%) (Epoch {valid_h10.index(max(valid_h10)) + 1})")
    
    # Alpha values
    train_alpha = [entry.get('train_alpha', None) for entry in log_data]
    val_alpha = [entry.get('val_alpha', None) for entry in log_data]
    valid_train_alpha = [a for a in train_alpha if a is not None]
    valid_val_alpha = [a for a in val_alpha if a is not None]
    
    if valid_train_alpha:
        print(f"\nGate Weight (α) - Text Contribution:")
        print(f"  Train - Initial: {valid_train_alpha[0]:.4f}, Final: {valid_train_alpha[-1]:.4f}")
        # print(f"  Val   - Initial: {valid_val_alpha[0]:.4f}, Final: {valid_val_alpha[-1]:.4f}")
        print(f"  (Higher α = model relies more on text embeddings)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--log_path', type=str, help='Path to training_log.json file')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization (e.g., training_curves.png)')
    parser.add_argument('--no_show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    # Load log
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"❌ Error: Log file not found at {log_path}")
        return
    
    print(f"Loading training log from: {log_path}")
    log_data = load_training_log(log_path)
    
    # Print summary
    print_metrics_summary(log_data)
    
    # Generate visualization
    output_path = args.output
    if not output_path:
        # Default: save next to log file with same name
        output_path = log_path.parent / 'training_curves.png'
    
    plot_training_curves(log_data, output_path=output_path)


if __name__ == '__main__':
    main()
