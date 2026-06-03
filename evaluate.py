import os
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
import json
import sys

# Paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.reranker import GWMReranker
from model.dataset import GWMDataset, CollateFN
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)

def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir: config_dict['data_dir'] = args.data_dir
    if args.output_dir: config_dict['output_dir'] = args.output_dir
    if args.eval_mode: config_dict['eval_mode'] = args.eval_mode
    
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
    
    # Load retriever checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint found. Evaluating initialized model (random).")

    # Optional standalone reranker for stage-2 end-to-end evaluation.
    reranker = None
    reranker_checkpoint = os.path.join(config.output_dir, 'reranker_best.pt')

    if reranker_checkpoint is not None and os.path.exists(reranker_checkpoint):
        print(f"Loading standalone reranker: {reranker_checkpoint}")
        reranker = GWMReranker(config).to(device)
        reranker_state = torch.load(reranker_checkpoint, map_location=device)
        reranker.load_state_dict(reranker_state, strict=True)
        reranker.eval()
    else:
        print("No reranker checkpoint found. Running retrieval-only evaluation.")

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
    print(f"Encoded {all_entity_embeddings[0].size(0)} entities.")

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
        # Standard test protocol: filter with train+valid
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth_train_valid.json',
            fallback_splits=['train', 'valid']
        )
    else:
        # Validation protocol: filter with train only
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth_train.json',
            fallback_splits=['train']
        )

    def _run_single_eval(mode_name, mode_reranker):
        predictions_path = os.path.join(config.output_dir, f'predictions_{split}_{mode_name}.jsonl')
        metrics = compute_filtered_ranking_metrics(
            model=model,
            data_loader=test_loader,
            all_entity_embeddings=all_entity_embeddings,
            hr_map=hr_map,
            device=device,
            desc=f"Evaluating ({mode_name})",
            save_predictions_path=predictions_path,
            rerank_topk=int(getattr(config, 'reranker_eval_topk', 100)),
            reranker=mode_reranker,
        )

        print(f"\n--- Evaluation Results ({split}) [{mode_name}] ---")
        print(f"MRR       : {metrics['MRR']:.4f}")
        print(f"Hits@1    : {metrics['Hits@1']:.4f}")
        print(f"Hits@3    : {metrics['Hits@3']:.4f}")
        print(f"Hits@10   : {metrics['Hits@10']:.4f}")
        print("-----------------------------------------------")
        return {
            'metrics': {
                'mrr': metrics['MRR'],
                'hits1': metrics['Hits@1'],
                'hits3': metrics['Hits@3'],
                'hits10': metrics['Hits@10'],
                'mr': metrics['MR'],
            },
            'predictions_path': predictions_path,
        }

    eval_mode = getattr(config, 'eval_mode', getattr(args, 'eval_mode', 'both')).lower()
    if eval_mode not in {'both', 'retriever', 'full'}:
        raise ValueError("eval_mode must be one of: both, retriever, full")

    results = {
        'split': split,
        'retriever_checkpoint': checkpoint_path,
        'reranker_checkpoint': reranker_checkpoint if reranker is not None else None,
        'runs': {},
    }

    if eval_mode in {'both', 'retriever'}:
        results['runs']['retriever_only'] = _run_single_eval('retriever_only', None)

    if eval_mode in {'both', 'full'}:
        if reranker is None:
            print("Skipping retriever+rereanker run because reranker checkpoint is unavailable.")
        else:
            results['runs']['retriever_reranker'] = _run_single_eval('retriever_reranker', reranker)

    if 'retriever_only' in results['runs'] and 'retriever_reranker' in results['runs']:
        r = results['runs']['retriever_only']['metrics']
        rr = results['runs']['retriever_reranker']['metrics']
        results['delta'] = {
            'mrr': rr['mrr'] - r['mrr'],
            'hits1': rr['hits1'] - r['hits1'],
            'hits3': rr['hits3'] - r['hits3'],
            'hits10': rr['hits10'] - r['hits10'],
            'mr': rr['mr'] - r['mr'],
        }

        print("\n--- Delta (Retriever+Reranker - Retriever) ---")
        print(f"MRR       : {results['delta']['mrr']:+.4f}")
        print(f"Hits@1    : {results['delta']['hits1']:+.4f}")
        print(f"Hits@3    : {results['delta']['hits3']:+.4f}")
        print(f"Hits@10   : {results['delta']['hits10']:+.4f}")
        print(f"MR        : {results['delta']['mr']:+.4f}")

    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--eval_mode', type=str, default='both', choices=['both', 'retriever', 'full'],
                        help='both: run retriever-only and retriever+reranker; retriever: retriever-only; full: retriever+reranker only')
    args = parser.parse_args()
    evaluate(args)
