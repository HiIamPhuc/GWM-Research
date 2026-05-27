import os
import json
import torch
import argparse
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from utils.seed import seed_everything

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def align_and_save_embeddings(pykeen_model, pykeen_tf, our_e2id_path, our_r2id_path, out_ent_path, out_rel_path, target_dim):
    """
    Aligns PyKEEN's internally numbered embeddings to our GWM project's entity2id/relation2id mapping,
    and saves them as ready-to-load .pt files.
    """
    print(f"\nAligning embeddings to our project's dictionary...")
    our_e2id = load_json(our_e2id_path)
    our_r2id = load_json(our_r2id_path)
    
    # 1. Extract raw embeddings from PyKEEN
    # Some models like RotatE use complex numbers; we flatten them to real numbers: (num_entities, dim * 2) or (num_entities, dim)
    ent_emb_full = pykeen_model.entity_representations[0]().detach().cpu()
    rel_emb_full = pykeen_model.relation_representations[0]().detach().cpu()
    
    if torch.is_complex(ent_emb_full):
        ent_emb_full = torch.view_as_real(ent_emb_full).view(ent_emb_full.shape[0], -1)
        rel_emb_full = torch.view_as_real(rel_emb_full).view(rel_emb_full.shape[0], -1)
        
    actual_dim = ent_emb_full.shape[1]
    print(f"Extracted PyKEEN embedding dimension: {actual_dim} (Target: {target_dim})")
    
    # 2. Create blank tensors for our ordering
    num_our_entities = len(our_e2id)
    num_our_relations = len(our_r2id)
    
    aligned_entities = torch.zeros((num_our_entities, actual_dim))
    aligned_relations = torch.zeros((num_our_relations, actual_dim))
    
    # 3. Align Entities
    pykeen_e2id = pykeen_tf.entity_to_id
    unmatched_ent = 0
    for str_ent, pykeen_id in pykeen_e2id.items():
        if str_ent in our_e2id:
            our_id = our_e2id[str_ent]
            aligned_entities[our_id] = ent_emb_full[pykeen_id]
        else:
            unmatched_ent += 1
            
    # 4. Align Relations
    pykeen_r2id = pykeen_tf.relation_to_id
    unmatched_rel = 0
    for str_rel, pykeen_id in pykeen_r2id.items():
        if str_rel in our_r2id:
            our_id = our_r2id[str_rel]
            aligned_relations[our_id] = rel_emb_full[pykeen_id]
        else:
            unmatched_rel += 1
            
    print(f"Alignment Complete.")
    print(f"Unmatched Entities: {unmatched_ent} | Unmatched Relations: {unmatched_rel}")
    
    # 5. Save
    torch.save(aligned_entities, out_ent_path)
    torch.save(aligned_relations, out_rel_path)
    print(f"Saved structural entities to: {out_ent_path}")
    print(f"Saved structural relations to: {out_rel_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Structural Priors using PyKEEN")
    parser.add_argument('--dataset', type=str, choices=['wn18rr', 'fb15k-237'], required=True)
    parser.add_argument('--model', type=str, default='RotatE', choices=['RotatE', 'ComplEx', 'TransE'])
    parser.add_argument('--dim', type=int, default=384, help="Embedding dimension. Note: RotatE complex space means internal dim is halved.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible PyKEEN runs')
    args = parser.parse_args()

    dataset_lower = args.dataset.lower()
    dataset_upper = "WN18RR" if dataset_lower == "wn18rr" else "FB15k-237"
    
    raw_data_dir = os.path.join('data', dataset_upper)
    processed_dir = os.path.join('data-processed', dataset_lower)
    
    train_path = os.path.join(raw_data_dir, 'train.txt')
    valid_path = os.path.join(raw_data_dir, 'valid.txt')
    test_path = os.path.join(raw_data_dir, 'test.txt')
    
    print(f"Loading Triples for {dataset_upper}...")
    tf_train = TriplesFactory.from_path(train_path)
    tf_valid = TriplesFactory.from_path(valid_path, entity_to_id=tf_train.entity_to_id, relation_to_id=tf_train.relation_to_id)
    tf_test = TriplesFactory.from_path(test_path, entity_to_id=tf_train.entity_to_id, relation_to_id=tf_train.relation_to_id)

    seed_everything(args.seed)
    # For RotatE, the user-facing dimension represents both real and imaginary combined in our PyTorch tensor.
    # PyKEEN's internal embedding_dim for RotatE is usually dim // 2.
    internal_dim = args.dim // 2 if args.model == 'RotatE' else args.dim

    print(f"Training {args.model} on {dataset_upper}...")
    pipeline_result = pipeline(
        training=tf_train,
        testing=tf_test,
        validation=tf_valid,
        model=args.model,
        model_kwargs=dict(embedding_dim=internal_dim),
        training_kwargs=dict(num_epochs=args.epochs, use_tqdm_batch=True),
        evaluator_kwargs=dict(filtered=True),
        stopper='early',
        stopper_kwargs=dict(frequency=5, patience=3, relative_delta=0.002),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        random_seed=args.seed,
    )
    
    metrics = pipeline_result.metric_results
    mrr = metrics.get_metric('mrr')
    hits_at_1 = metrics.get_metric('hits_at_1')
    hits_at_3 = metrics.get_metric('hits_at_3')
    hits_at_10 = metrics.get_metric('hits_at_10')
    
    print(f"\nFinal {args.model} Metrics:")
    print(f"  MRR:        {mrr:.4f}")
    print(f"  Hits@1:     {hits_at_1:.4f}")
    print(f"  Hits@3:     {hits_at_3:.4f}")
    print(f"  Hits@10:    {hits_at_10:.4f}")

    # Paths for alignment validation
    our_e2id_path = os.path.join(processed_dir, 'entity2id.json')
    our_r2id_path = os.path.join(processed_dir, 'relation2id.json')
    out_ent_path = os.path.join(processed_dir, 'structural_entities.pt')
    out_rel_path = os.path.join(processed_dir, 'structural_relations.pt')

    if os.path.exists(our_e2id_path) and os.path.exists(our_r2id_path):
        align_and_save_embeddings(
            pykeen_model=pipeline_result.model,
            pykeen_tf=tf_train,
            our_e2id_path=our_e2id_path,
            our_r2id_path=our_r2id_path,
            out_ent_path=out_ent_path,
            out_rel_path=out_rel_path,
            target_dim=args.dim
        )
    else:
        print(f"\nCould not find our project's {our_e2id_path} to align! Please ensure preprocessing is run.")

if __name__ == '__main__':
    main()
