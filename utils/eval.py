import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EntityDataset(Dataset):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)

        self.num_entities = len(self.entity2id)

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        return {
            'id': idx,
        }


class TensorTripleDataset(Dataset):
    def __init__(self, base_dataset, triples):
        self.base_dataset = base_dataset
        self.data_dir = base_dataset.data_dir
        self.split = base_dataset.split
        self.num_entities = base_dataset.num_entities
        self.num_relations = base_dataset.num_relations
        self.triples = torch.as_tensor(triples, dtype=torch.long)

    def __len__(self):
        return int(self.triples.size(0))

    def __getitem__(self, idx):
        return self.base_dataset.make_item(*self.triples[idx])


def load_triples_for_filtering(data_dir, splits=None):
    if splits is None:
        splits = ['train']

    all_triples = set()
    for split in splits:
        path = os.path.join(data_dir, f'{split}_triples.pt')
        if os.path.exists(path):
            triples = torch.load(path)
            for h, r, t in triples:
                all_triples.add((h.item(), r.item(), t.item()))
    return all_triples


def load_hr_map_for_filtering(data_dir, preferred_ground_truth_file=None, fallback_splits=None):
    if fallback_splits is None:
        fallback_splits = ['train']

    if preferred_ground_truth_file is not None:
        gt_path = os.path.join(data_dir, preferred_ground_truth_file)
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_json = json.load(f)

            hr_map = {}
            for key, tails in gt_json.items():
                h, r = map(int, key.split(','))
                hr_map[(h, r)] = set(int(t) for t in tails)
            return hr_map

    all_triples = load_triples_for_filtering(data_dir, splits=fallback_splits)
    hr_map = {}
    for h, r, t in all_triples:
        if (h, r) not in hr_map:
            hr_map[(h, r)] = set()
        hr_map[(h, r)].add(t)
    return hr_map


def load_inverse_relation_id_map(data_dir):
    with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
        relation2id = json.load(f)

    inverse_relation_ids = {}
    missing = []
    for relation, relation_id in relation2id.items():
        inverse_relation = (
            relation[:-4] if relation.endswith('_inv') else relation + '_inv'
        )
        if inverse_relation not in relation2id:
            missing.append(relation)
            continue
        inverse_relation_ids[int(relation_id)] = int(relation2id[inverse_relation])

    if missing:
        preview = ', '.join(missing[:10])
        raise ValueError(
            "Strict bidirectional evaluation requires every relation to have "
            f"an inverse relation ID. Missing inverse entries for: {preview}"
        )

    return inverse_relation_ids


def load_inverse_relation_ids(data_dir):
    """Return IDs explicitly assigned to `_inv` relation vocabulary entries."""
    with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
        relation2id = json.load(f)

    inverse_relation_ids = sorted(
        int(relation_id)
        for relation, relation_id in relation2id.items()
        if relation.endswith('_inv')
    )
    if not inverse_relation_ids:
        raise ValueError(
            "Directional scoring requires relation names ending in `_inv`."
        )
    return inverse_relation_ids


def build_inverse_triples(triples, inverse_relation_ids):
    triples = torch.as_tensor(triples, dtype=torch.long)
    inverse_rows = []
    for h, r, t in triples.tolist():
        inverse_rows.append((t, inverse_relation_ids[int(r)], h))
    return torch.tensor(inverse_rows, dtype=torch.long)


def build_bidirectional_eval_dataset(base_dataset, data_dir):
    """Build forward and backward eval datasets from original-direction triples."""
    inverse_relation_ids = load_inverse_relation_id_map(data_dir)
    forward_dataset = TensorTripleDataset(
        base_dataset=base_dataset,
        triples=base_dataset.triples,
    )
    backward_dataset = TensorTripleDataset(
        base_dataset=base_dataset,
        triples=build_inverse_triples(base_dataset.triples, inverse_relation_ids),
    )
    return forward_dataset, backward_dataset


def combine_forward_backward_metrics(forward_metrics, backward_metrics):
    combined = {}
    for key in ('MRR', 'MR', 'Hits@1', 'Hits@3', 'Hits@10'):
        combined[key] = 0.5 * (forward_metrics[key] + backward_metrics[key])
    combined['count'] = forward_metrics['count'] + backward_metrics['count']
    combined['forward'] = forward_metrics
    combined['backward'] = backward_metrics

    total = combined['count']
    combined['micro'] = {
        key: (
            forward_metrics[key] * forward_metrics['count']
            + backward_metrics[key] * backward_metrics['count']
        ) / total
        for key in ('MRR', 'MR', 'Hits@1', 'Hits@3', 'Hits@10')
    }
    combined['micro']['count'] = total
    return combined


def build_bidirectional_hr_map_for_filtering(data_dir, splits=None):
    if splits is None:
        splits = ['train', 'valid', 'test']

    inverse_relation_ids = load_inverse_relation_id_map(data_dir)
    triples = load_triples_for_filtering(data_dir, splits=splits)
    hr_map = {}

    def add_truth(h, r, t):
        hr_map.setdefault((h, r), set()).add(t)

    for h, r, t in triples:
        add_truth(h, r, t)
        if int(r) in inverse_relation_ids:
            add_truth(t, inverse_relation_ids[int(r)], h)

    return hr_map


def build_loader_like(data_loader, dataset):
    return DataLoader(
        dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=data_loader.collate_fn,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )


def compute_bidirectional_filtered_ranking_metrics(
    model,
    data_loader,
    all_entity_embeddings,
    hr_map,
    device,
    desc="Bidirectional Evaluation",
    save_predictions_path=None,
    topk=50,
):
    forward_dataset, backward_dataset = build_bidirectional_eval_dataset(
        base_dataset=data_loader.dataset,
        data_dir=data_loader.dataset.data_dir,
    )
    forward_loader = build_loader_like(data_loader, forward_dataset)
    backward_loader = build_loader_like(data_loader, backward_dataset)

    forward_metrics = compute_filtered_ranking_metrics(
        model=model,
        data_loader=forward_loader,
        all_entity_embeddings=all_entity_embeddings,
        hr_map=hr_map,
        device=device,
        desc=f"{desc} [forward]",
        save_predictions_path=save_predictions_path,
        topk=topk,
    )

    backward_path = None
    if save_predictions_path is not None:
        root, ext = os.path.splitext(save_predictions_path)
        backward_path = f"{root}_backward{ext}"

    backward_metrics = compute_filtered_ranking_metrics(
        model=model,
        data_loader=backward_loader,
        all_entity_embeddings=all_entity_embeddings,
        hr_map=hr_map,
        device=device,
        desc=f"{desc} [backward]",
        save_predictions_path=backward_path,
        topk=topk,
    )

    return combine_forward_backward_metrics(
        forward_metrics,
        backward_metrics,
    )


def build_entity_loader(data_dir, batch_size, num_workers=2):
    entity_dataset = EntityDataset(data_dir)

    def entity_collate(batch):
        ids = [x['id'] for x in batch]
        return {
            'id': torch.tensor(ids)
        }

    return DataLoader(
        entity_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=entity_collate,
        num_workers=num_workers
    )


def encode_all_entities_as_targets(model, entity_loader, device):
    all_chunks = []
    model.eval()
    with torch.no_grad():
        for batch in entity_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_chunks.append(model.encode_target(batch).cpu())
    return torch.cat(all_chunks, dim=0).to(device)


def _new_metric_state():
    return {
        'hits1': 0,
        'hits3': 0,
        'hits10': 0,
        'mrr': 0.0,
        'mr': 0.0,
        'count': 0,
    }


def _add_ranks_to_state(state, ranks):
    if ranks.numel() == 0:
        return
    ranks = ranks.float()
    state['hits1'] += (ranks <= 1).sum().item()
    state['hits3'] += (ranks <= 3).sum().item()
    state['hits10'] += (ranks <= 10).sum().item()
    state['mrr'] += (1.0 / ranks).sum().item()
    state['mr'] += ranks.sum().item()
    state['count'] += ranks.numel()


def _finalize_metric_state(state):
    total = state['count']
    if total == 0:
        raise ValueError("Cannot compute ranking metrics from zero queries.")

    return {
        'MRR': state['mrr'] / total,
        'MR': state['mr'] / total,
        'Hits@1': state['hits1'] / total,
        'Hits@3': state['hits3'] / total,
        'Hits@10': state['hits10'] / total,
        'count': total,
    }


def compute_filtered_ranking_metrics(
    model,
    data_loader,
    all_entity_embeddings,
    hr_map,
    device,
    desc="Filtered Ranking",
    save_predictions_path=None,
    topk=50,
):
    micro_state = _new_metric_state()

    writer = None
    if save_predictions_path is not None:
        os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
        writer = open(save_predictions_path, 'w', encoding='utf-8')

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {
                k: v.to(device) for k, v in batch['context_batch'].items()
            }

            t_ids = batch['t_batch']['id'].to(device)
            h_ids_tensor = batch['h_batch']['id']
            r_ids_tensor = batch['r_batch']['id']
            h_ids = h_ids_tensor.cpu().numpy()
            r_ids = r_ids_tensor.cpu().numpy()

            scores = model.score_all_entities(
                h_batch,
                r_batch,
                context_batch=context_batch,
                candidate_vectors=all_entity_embeddings,
            )

            # Prevent NaNs/Infs from masquerading as perfect ranks.
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

            for i in range(scores.size(0)):
                h_id = h_ids[i]
                r_id = r_ids[i]
                true_t = t_ids[i].item()

                filter_mask_indices = list(hr_map.get((h_id, r_id), []))
                if true_t in filter_mask_indices:
                    filter_mask_indices.remove(true_t)

                if filter_mask_indices:
                    scores[i, filter_mask_indices] = -float('inf')

            target_scores = scores.gather(1, t_ids.unsqueeze(1))
            ranks = (scores > target_scores).sum(dim=1) + 1

            if writer is not None:
                topk_val = min(topk, scores.size(1))
                fused_scores, fused_indices = torch.topk(scores, k=topk_val, dim=1)

                for row_idx in range(scores.size(0)):
                    record = {
                        'h': int(h_ids[row_idx]),
                        'r': int(r_ids[row_idx]),
                        't': int(t_ids[row_idx].item()),
                        'rank': int(ranks[row_idx].item()),
                        'topk': fused_indices[row_idx].tolist(),
                        'topk_scores': fused_scores[row_idx].tolist(),
                    }
                    writer.write(json.dumps(record) + '\n')

            _add_ranks_to_state(micro_state, ranks)

    if writer is not None:
        writer.close()

    return _finalize_metric_state(micro_state)
