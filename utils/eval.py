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
    all_text_chunks = []
    all_struct_chunks = []
    model.eval()
    with torch.no_grad():
        for batch in entity_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            txt, struct = model.encode_target(batch)
            all_text_chunks.append(txt.cpu())
            all_struct_chunks.append(struct.cpu())
    return (
        torch.cat(all_text_chunks, dim=0).to(device),
        torch.cat(all_struct_chunks, dim=0).to(device)
    )


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
    hits1, hits3, hits10, mrr, mr = 0, 0, 0, 0.0, 0.0
    total = 0

    all_t_text, all_t_struct = all_entity_embeddings

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
            h_ids = batch['h_batch']['id'].cpu().numpy()
            r_ids = batch['r_batch']['id'].cpu().numpy()

            q_out = model(h_batch, r_batch, context_batch)
            q_text, q_struct, rel_text, rel_struct = q_out
            
            scores_text = torch.mm(q_text, all_t_text.t())
            scores_struct = torch.mm(q_struct, all_t_struct.t())

            scores_text = scores_text / model.temperature
            scores_struct = scores_struct / model.temperature

            alpha = model.alpha_mlp(torch.cat([rel_text, rel_struct], dim=-1))
            scores = alpha * scores_text + (1.0 - alpha) * scores_struct

            # Prevent NaNs/Infs from masquerading as perfect ranks.
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
            scores_text = torch.nan_to_num(scores_text, nan=-1e9, posinf=1e9, neginf=-1e9)
            scores_struct = torch.nan_to_num(scores_struct, nan=-1e9, posinf=1e9, neginf=-1e9)
            alpha = torch.nan_to_num(alpha, nan=0.5, posinf=1.0, neginf=0.0)

            for i in range(scores.size(0)):
                h_id = h_ids[i]
                r_id = r_ids[i]
                true_t = t_ids[i].item()

                filter_mask_indices = list(hr_map.get((h_id, r_id), []))
                if true_t in filter_mask_indices:
                    filter_mask_indices.remove(true_t)

                if filter_mask_indices:
                    scores[i, filter_mask_indices] = -float('inf')
                    scores_text[i, filter_mask_indices] = -float('inf')
                    scores_struct[i, filter_mask_indices] = -float('inf')

            target_scores = scores.gather(1, t_ids.unsqueeze(1))
            ranks = (scores > target_scores).sum(dim=1) + 1

            if writer is not None:
                topk_val = min(topk, scores.size(1))
                fused_scores, fused_indices = torch.topk(scores, k=topk_val, dim=1)
                text_scores, text_indices = torch.topk(scores_text, k=topk_val, dim=1)
                struct_scores, struct_indices = torch.topk(scores_struct, k=topk_val, dim=1)

                for row_idx in range(scores.size(0)):
                    record = {
                        'h': int(h_ids[row_idx]),
                        'r': int(r_ids[row_idx]),
                        't': int(t_ids[row_idx].item()),
                        'rank_fused': int(ranks[row_idx].item()),
                        'topk_fused': fused_indices[row_idx].tolist(),
                        'topk_fused_scores': fused_scores[row_idx].tolist(),
                        'topk_text': text_indices[row_idx].tolist(),
                        'topk_text_scores': text_scores[row_idx].tolist(),
                        'topk_struct': struct_indices[row_idx].tolist(),
                        'topk_struct_scores': struct_scores[row_idx].tolist(),
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

    return {
        'MRR': mrr / total,
        'MR': mr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }
