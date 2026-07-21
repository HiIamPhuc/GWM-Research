import os
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
import json
import sys

# Paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import GWMDataset, CollateFN
from model.model import GWM
from utils.eval import (
    build_bidirectional_hr_map_for_filtering,
    build_entity_loader,
    compute_bidirectional_filtered_ranking_metrics,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)

def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir: config_dict['data_dir'] = args.data_dir
    if args.output_dir: config_dict['output_dir'] = args.output_dir
    
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    return Config(config_dict)

def evaluate(args):
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use conservative defaults for evaluation to avoid OOM on large configs.
    eval_batch_size = int(getattr(config, 'eval_batch_size', min(int(config.batch_size), 128)))
    candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(eval_batch_size * 2, 256)))

    # 1. Load Model
    print("Loading model...")
    # Get num entities/relations
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))

    model = GWM(config).to(device)
    
    # Load Checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}, trying latest...")
        checkpoint_path = os.path.join(config.output_dir, 'latest_checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}."
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            "Unsupported legacy checkpoint. Expected a full training "
            "checkpoint containing 'model_state_dict'."
        )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model.eval()

    # 2. Encode All Candidates (Target Embeddings)
    print("Encoding all entities as targets...")
    entity_loader = build_entity_loader(
        data_dir=config.data_dir,
        batch_size=candidate_batch_size,
        num_workers=4,
    )

    all_entity_embeddings = encode_all_entities_as_targets(
        model=model,
        entity_loader=entity_loader,
        device=device
    )
    print(f"Encoded {all_entity_embeddings.size(0)} entities.")

    # 3. Evaluation Loop
    split = 'test'
    print(f"Evaluating on {split} set...")
    if not os.path.exists(os.path.join(config.data_dir, f'{split}_triples.pt')):
        print(f"Test triples not found, using 'valid' set.")
        split = 'valid'

    test_dataset = GWMDataset(config.data_dir, split=split)
    collate_fn = CollateFN()
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size,
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )

    # Filtering Setup
    if split == 'test':
        hr_map = build_bidirectional_hr_map_for_filtering(
            config.data_dir,
            splits=['train', 'valid', 'test'],
        )
    else:
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file=None,
            fallback_splits=['train']
        )

    predictions_path = os.path.join(config.output_dir, f'predictions_{split}.jsonl')

    if split == 'test':
        metrics = compute_bidirectional_filtered_ranking_metrics(
            model=model,
            data_loader=test_loader,
            all_entity_embeddings=all_entity_embeddings,
            hr_map=hr_map,
            device=device,
            desc="Evaluating",
            save_predictions_path=predictions_path,
        )
    else:
        metrics = compute_filtered_ranking_metrics(
            model=model,
            data_loader=test_loader,
            all_entity_embeddings=all_entity_embeddings,
            hr_map=hr_map,
            device=device,
            desc="Evaluating",
            save_predictions_path=predictions_path,
        )

    final_mrr = metrics['MRR']
    final_h1 = metrics['Hits@1']
    final_h3 = metrics['Hits@3']
    final_h10 = metrics['Hits@10']

    if split == 'test':
        print(f"\n--- Bidirectional Evaluation Results ({split}) ---")
    else:
        print(f"\n--- Tail-Only Evaluation Results ({split}) ---")
    print(f"MRR       : {final_mrr:.4f}")
    print(f"Hits@1    : {final_h1:.4f}")
    print(f"Hits@3    : {final_h3:.4f}")
    print(f"Hits@10   : {final_h10:.4f}")
    if split == 'test':
        print(
            f"Forward  : MRR {metrics['forward']['MRR']:.4f} | "
            f"Hits@10 {metrics['forward']['Hits@10']:.4f} | "
            f"count {metrics['forward']['count']}"
        )
        print(
            f"Backward : MRR {metrics['backward']['MRR']:.4f} | "
            f"Hits@10 {metrics['backward']['Hits@10']:.4f} | "
            f"count {metrics['backward']['count']}"
        )
    print("-------------------------------")
    
    # Save results
    results = {
        'mrr': final_mrr,
        'hits1': final_h1,
        'hits3': final_h3,
        'hits10': final_h10,
        'evaluation_mode': 'bidirectional_test' if split == 'test' else 'tail_only',
    }
    if split == 'test':
        results.update({
            'forward': {
            'mrr': metrics['forward']['MRR'],
            'mr': metrics['forward']['MR'],
            'hits1': metrics['forward']['Hits@1'],
            'hits3': metrics['forward']['Hits@3'],
            'hits10': metrics['forward']['Hits@10'],
            'count': metrics['forward']['count'],
            },
            'backward': {
            'mrr': metrics['backward']['MRR'],
            'mr': metrics['backward']['MR'],
            'hits1': metrics['backward']['Hits@1'],
            'hits3': metrics['backward']['Hits@3'],
            'hits10': metrics['backward']['Hits@10'],
            'count': metrics['backward']['count'],
            },
            'micro': {
            'mrr': metrics['micro']['MRR'],
            'mr': metrics['micro']['MR'],
            'hits1': metrics['micro']['Hits@1'],
            'hits3': metrics['micro']['Hits@3'],
            'hits10': metrics['micro']['Hits@10'],
            'count': metrics['micro']['count'],
            },
        })
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    evaluate(args)
