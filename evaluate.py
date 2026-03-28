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
    text_cache_batch_size = int(getattr(config, 'text_cache_batch_size', 128))

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
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found. Evaluating initialized model (random).")

    if not config.finetune_text_encoder:
        print("Precomputing frozen text embeddings for evaluation...")
        with open(os.path.join(config.data_dir, 'entity_text.json'), 'r') as f:
            entity_text_map = json.load(f)
        with open(os.path.join(config.data_dir, 'relation_text.json'), 'r') as f:
            relation_text_map = json.load(f)

        model.build_text_embedding_cache(
            entity_text_map=entity_text_map,
            relation_text_map=relation_text_map,
            device=device,
            batch_size=text_cache_batch_size,
            max_entity_length=getattr(config, 'max_length', 512),
            max_relation_length=getattr(config, 'max_length', 256),
        )
        print("Frozen text cache ready for evaluation.")

    model.eval()

    # 2. Encode All Candidates (Target Embeddings)
    print("Encoding all entities as targets...")
    entity_loader = build_entity_loader(
        model=model,
        data_dir=config.data_dir,
        batch_size=candidate_batch_size,
        finetune_text_encoder=config.finetune_text_encoder,
        num_workers=4,
        max_length=512,
    )

    all_entity_embeddings = encode_all_entities_as_targets(
        model=model,
        entity_loader=entity_loader,
        device=device,
        desc="Encoding Entities",
    )
    print(f"Encoded {all_entity_embeddings.size(0)} entities.")

    # 3. Evaluation Loop
    split = 'test'
    print(f"Evaluating on {split} set...")
    if not os.path.exists(os.path.join(config.data_dir, f'{split}_triples.pt')):
        print(f"Test triples not found, using 'valid' set.")
        split = 'valid'

    test_dataset = GWMDataset(config.data_dir, split=split)
    collate_fn = CollateFN(model.tokenizer)
    
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

    metrics = compute_filtered_ranking_metrics(
        model=model,
        data_loader=test_loader,
        all_entity_embeddings=all_entity_embeddings,
        hr_map=hr_map,
        device=device,
        desc="Evaluating",
    )

    final_mrr = metrics['MRR']
    final_h1 = metrics['Hits@1']
    final_h3 = metrics['Hits@3']
    final_h10 = metrics['Hits@10']

    print(f"\n--- Evaluation Results ({split}) ---")
    print(f"MRR       : {final_mrr:.4f}")
    print(f"Hits@1    : {final_h1:.4f}")
    print(f"Hits@3    : {final_h3:.4f}")
    print(f"Hits@10   : {final_h10:.4f}")
    print("-------------------------------")
    
    # Save results
    results = {
        'mrr': final_mrr,
        'hits1': final_h1,
        'hits3': final_h3,
        'hits10': final_h10
    }
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    evaluate(args)
