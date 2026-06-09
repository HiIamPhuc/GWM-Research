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

def create_pykeen_mappings(our_e2id, our_r2id):
    """Create complete, compact PyKEEN mappings without inverse relations."""
    entity_to_id = {
        entity: index
        for index, entity in enumerate(
            sorted(our_e2id, key=our_e2id.get)
        )
    }
    original_relations = [
        relation
        for relation in sorted(our_r2id, key=our_r2id.get)
        if not relation.endswith('_inv')
    ]
    relation_to_id = {
        relation: index for index, relation in enumerate(original_relations)
    }
    return entity_to_id, relation_to_id

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
    ent_emb_raw = pykeen_model.entity_representations[0]().detach().cpu()
    rel_emb_raw = pykeen_model.relation_representations[0]().detach().cpu()

    if torch.is_complex(ent_emb_raw):
        ent_emb_full = torch.view_as_real(ent_emb_raw).reshape(ent_emb_raw.shape[0], -1)
    else:
        ent_emb_full = ent_emb_raw

    if torch.is_complex(rel_emb_raw):
        rel_emb_full = torch.view_as_real(rel_emb_raw).reshape(rel_emb_raw.shape[0], -1)
        inverse_rel_emb_full = torch.view_as_real(
            rel_emb_raw.conj().resolve_conj()
        ).reshape(rel_emb_raw.shape[0], -1)
    else:
        rel_emb_full = rel_emb_raw
        inverse_rel_emb_full = -rel_emb_raw

    entity_dim = ent_emb_full.shape[1]
    relation_dim = rel_emb_full.shape[1]
    print(
        f"Extracted PyKEEN dimensions: entities={entity_dim}, "
        f"relations={relation_dim} (Target: {target_dim})"
    )
    if entity_dim != target_dim or relation_dim != target_dim:
        raise ValueError(
            f"Structural prior dimension mismatch. Expected {target_dim}, "
            f"got entities={entity_dim}, relations={relation_dim}."
        )
    
    # 2. Create blank tensors for our ordering
    num_our_entities = len(our_e2id)
    num_our_relations = len(our_r2id)
    
    aligned_entities = torch.zeros((num_our_entities, target_dim))
    aligned_relations = torch.zeros((num_our_relations, target_dim))
    
    # 3. Align Entities
    pykeen_e2id = pykeen_tf.entity_to_id
    assigned_entities = set()
    for str_ent, pykeen_id in pykeen_e2id.items():
        if str_ent in our_e2id:
            our_id = our_e2id[str_ent]
            aligned_entities[our_id] = ent_emb_full[pykeen_id]
            assigned_entities.add(our_id)
            
    # 4. Align Relations
    pykeen_r2id = pykeen_tf.relation_to_id
    assigned_relations = set()
    for str_rel, pykeen_id in pykeen_r2id.items():
        if str_rel in our_r2id:
            our_id = our_r2id[str_rel]
            aligned_relations[our_id] = rel_emb_full[pykeen_id]

            assigned_relations.add(our_id)

        inverse_name = str_rel + '_inv'
        if inverse_name in our_r2id:
            inverse_id = our_r2id[inverse_name]
            aligned_relations[inverse_id] = inverse_rel_emb_full[pykeen_id]
            assigned_relations.add(inverse_id)

    missing_entities = sorted(set(range(num_our_entities)) - assigned_entities)
    missing_relations = sorted(set(range(num_our_relations)) - assigned_relations)
    if missing_entities or missing_relations:
        raise ValueError(
            "Could not align all structural priors. "
            f"Missing entity rows: {missing_entities[:10]}; "
            f"missing relation rows: {missing_relations[:10]}."
        )
            
    print(f"Alignment Complete.")
    print(
        f"Aligned {len(assigned_entities)} entities and "
        f"{len(assigned_relations)} relations (including inverses)."
    )
    
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
    our_e2id_path = os.path.join(processed_dir, 'entity2id.json')
    our_r2id_path = os.path.join(processed_dir, 'relation2id.json')

    if not os.path.exists(our_e2id_path) or not os.path.exists(our_r2id_path):
        raise FileNotFoundError(
            "Processed entity2id.json and relation2id.json are required. "
            "Run preprocessing before training structural priors."
        )

    our_e2id = load_json(our_e2id_path)
    our_r2id = load_json(our_r2id_path)
    pykeen_e2id, pykeen_r2id = create_pykeen_mappings(our_e2id, our_r2id)
    
    train_path = os.path.join(raw_data_dir, 'train.txt')
    valid_path = os.path.join(raw_data_dir, 'valid.txt')
    test_path = os.path.join(raw_data_dir, 'test.txt')
    
    print(f"Loading Triples for {dataset_upper}...")
    tf_train = TriplesFactory.from_path(
        train_path,
        entity_to_id=pykeen_e2id,
        relation_to_id=pykeen_r2id,
        compact_id=False,
    )
    tf_valid = TriplesFactory.from_path(
        valid_path,
        entity_to_id=pykeen_e2id,
        relation_to_id=pykeen_r2id,
        compact_id=False,
    )
    tf_test = TriplesFactory.from_path(
        test_path,
        entity_to_id=pykeen_e2id,
        relation_to_id=pykeen_r2id,
        compact_id=False,
    )

    missing_factory_entities = set(pykeen_e2id) - set(tf_train.entity_to_id)
    missing_factory_relations = set(pykeen_r2id) - set(tf_train.relation_to_id)
    if missing_factory_entities or missing_factory_relations:
        raise ValueError(
            "PyKEEN did not preserve the supplied complete vocabulary. "
            f"Missing entities: {sorted(missing_factory_entities)[:10]}; "
            f"missing relations: {sorted(missing_factory_relations)[:10]}."
        )

    train_entity_ids = set(tf_train.mapped_triples[:, [0, 2]].reshape(-1).tolist())
    untrained_entity_count = len(pykeen_e2id) - len(train_entity_ids)
    if untrained_entity_count:
        print(
            f"Note: {untrained_entity_count} entities do not occur in train.txt. "
            "PyKEEN allocates embeddings for alignment, but those rows receive "
            "no structural training signal."
        )

    seed_everything(args.seed)
    # The saved tensor flattens real and imaginary parts, so complex models
    # use half the requested output dimension internally.
    if args.model in {'RotatE', 'ComplEx'} and args.dim % 2 != 0:
        raise ValueError(f"{args.model} requires an even --dim value.")
    internal_dim = args.dim // 2 if args.model in {'RotatE', 'ComplEx'} else args.dim

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

    out_ent_path = os.path.join(processed_dir, 'structural_entities.pt')
    out_rel_path = os.path.join(processed_dir, 'structural_relations.pt')

    align_and_save_embeddings(
        pykeen_model=pipeline_result.model,
        pykeen_tf=tf_train,
        our_e2id_path=our_e2id_path,
        our_r2id_path=our_r2id_path,
        out_ent_path=out_ent_path,
        out_rel_path=out_rel_path,
        target_dim=args.dim
    )

if __name__ == '__main__':
    main()
