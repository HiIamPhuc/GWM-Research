import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EntityDataset(Dataset):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'entity_text.json'), 'r') as f:
            self.entity_text = json.load(f)

        with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)

        self.num_entities = len(self.entity2id)
        self.texts = [self.entity_text.get(str(i), "") for i in range(self.num_entities)]

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        return {
            'id': idx,
            'text': self.texts[idx]
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


def build_entity_loader(model, data_dir, batch_size, finetune_text_encoder, num_workers=2, max_length=64):
    entity_dataset = EntityDataset(data_dir)

    def entity_collate(batch):
        ids = [x['id'] for x in batch]

        # In frozen-text mode, encode_target can use cached text via IDs only.
        if model.use_text_cache and not finetune_text_encoder:
            return {
                'id': torch.tensor(ids)
            }

        texts = [x['text'] for x in batch]
        inputs = model.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'id': torch.tensor(ids)
        }

    return DataLoader(
        entity_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=entity_collate,
        num_workers=num_workers
    )


def encode_all_entities_as_targets(model, entity_loader, device, desc="Encoding Entities"):
    all_chunks = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(entity_loader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
            all_chunks.append(model.encode_target(batch).cpu())
    return torch.cat(all_chunks, dim=0).to(device)


def compute_filtered_ranking_metrics(model, data_loader, all_entity_embeddings, hr_map, device, desc="Filtered Ranking"):
    hits1, hits3, hits10, mrr, mr = 0, 0, 0, 0.0, 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            t_ids = batch['t_batch']['id'].to(device)
            h_ids = batch['h_batch']['id'].cpu().numpy()
            r_ids = batch['r_batch']['id'].cpu().numpy()

            query_vector = model(h_batch, r_batch, context_batch)
            scores = torch.mm(query_vector, all_entity_embeddings.t())

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

            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()
            mrr += (1.0 / ranks.float()).sum().item()
            mr += ranks.float().sum().item()
            total += ranks.size(0)

    return {
        'MRR': mrr / total,
        'MR': mr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }
