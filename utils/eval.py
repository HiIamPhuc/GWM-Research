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


def compute_filtered_ranking_metrics(
    model,
    data_loader,
    all_entity_embeddings,
    hr_map,
    device,
    desc="Filtered Ranking",
    save_predictions_path=None,
    topk=50,
    candidate_batch_size=None,
):
    hits1, hits3, hits10, mrr, mr = 0, 0, 0, 0.0, 0.0
    total = 0
    particle_usage = torch.zeros(
        model.num_particles, dtype=torch.long
    )
    num_candidates = all_entity_embeddings.size(0)
    if candidate_batch_size is None:
        candidate_batch_size = num_candidates
    candidate_batch_size = max(1, int(candidate_batch_size))

    writer = None
    if save_predictions_path is not None:
        os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
        writer = open(save_predictions_path, 'w', encoding='utf-8')

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            t_ids = batch['t_batch']['id'].to(device)
            h_ids = batch['h_batch']['id'].cpu().tolist()
            r_ids = batch['r_batch']['id'].cpu().tolist()

            query_particles = model(h_batch, r_batch, context_batch)
            target_vectors = all_entity_embeddings.index_select(0, t_ids)
            target_scores = (
                model.score_aligned_targets(query_particles, target_vectors)
                / model.temperature
            )
            target_scores = torch.nan_to_num(
                target_scores,
                nan=-1e9,
                posinf=1e9,
                neginf=-1e9,
            )
            winning_particles = model.winning_particle_ids(
                query_particles,
                target_vectors,
            ).cpu()
            particle_usage += torch.bincount(
                winning_particles,
                minlength=model.num_particles,
            )

            ranks = torch.ones(
                query_particles.size(0),
                dtype=torch.long,
                device=device,
            )
            top_scores = None
            top_indices = None
            topk_val = min(topk, num_candidates)

            for start in range(0, num_candidates, candidate_batch_size):
                end = min(start + candidate_batch_size, num_candidates)
                candidate_vectors = all_entity_embeddings[start:end]
                chunk_scores = (
                    model.score_candidates(
                        query_particles,
                        candidate_vectors,
                    )
                    / model.temperature
                )
                chunk_scores = torch.nan_to_num(
                    chunk_scores,
                    nan=-1e9,
                    posinf=1e9,
                    neginf=-1e9,
                )

                for i in range(chunk_scores.size(0)):
                    true_t = int(t_ids[i].item())
                    for filtered_t in hr_map.get(
                        (int(h_ids[i]), int(r_ids[i])),
                        (),
                    ):
                        if filtered_t != true_t and start <= filtered_t < end:
                            chunk_scores[i, filtered_t - start] = -float('inf')

                ranks += (chunk_scores > target_scores.unsqueeze(1)).sum(dim=1)

                if writer is not None:
                    chunk_k = min(topk_val, end - start)
                    chunk_top_scores, chunk_top_local = torch.topk(
                        chunk_scores,
                        k=chunk_k,
                        dim=1,
                    )
                    chunk_top_indices = chunk_top_local + start
                    if top_scores is None:
                        top_scores = chunk_top_scores
                        top_indices = chunk_top_indices
                    else:
                        merged_scores = torch.cat(
                            [top_scores, chunk_top_scores], dim=1
                        )
                        merged_indices = torch.cat(
                            [top_indices, chunk_top_indices], dim=1
                        )
                        top_scores, selected = torch.topk(
                            merged_scores,
                            k=min(topk_val, merged_scores.size(1)),
                            dim=1,
                        )
                        top_indices = merged_indices.gather(1, selected)

            if writer is not None:

                for row_idx in range(query_particles.size(0)):
                    record = {
                        'h': int(h_ids[row_idx]),
                        'r': int(r_ids[row_idx]),
                        't': int(t_ids[row_idx].item()),
                        'rank': int(ranks[row_idx].item()),
                        'winning_particle': int(winning_particles[row_idx]),
                        'topk': top_indices[row_idx].tolist(),
                        'topk_scores': top_scores[row_idx].tolist(),
                    }
                    writer.write(json.dumps(record) + '\n')

            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()
            mrr += (1.0 / ranks.float()).sum().item()
            mr += ranks.float().sum().item()
            total += ranks.size(0)

    if writer is not None:
        writer.close()

    metrics = {
        'MRR': mrr / total,
        'MR': mr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }
    metrics['ParticleUsage'] = (
        particle_usage.float() / max(total, 1)
    ).tolist()
    return metrics
