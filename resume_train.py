import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN
from utils.eval import (
	build_entity_loader,
	compute_filtered_ranking_metrics,
	encode_all_entities_as_targets,
	load_hr_map_for_filtering,
)
from utils.early_stopping import EarlyStopping


def _to_namespace(config_dict):
	return SimpleNamespace(**config_dict)


def _to_serializable(value):
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, (list, tuple)):
		return [_to_serializable(item) for item in value]
	if isinstance(value, dict):
		return {str(key): _to_serializable(item) for key, item in value.items()}
	return str(value)


def load_training_config(checkpoint_dir, data_dir=None, output_dir=None):
	config_path = os.path.join(checkpoint_dir, 'training_config.json')
	if not os.path.exists(config_path):
		raise FileNotFoundError(
			f"training_config.json not found in checkpoint directory: {config_path}"
		)

	with open(config_path, 'r', encoding='utf-8') as f:
		config_dict = json.load(f)

	if data_dir:
		config_dict['data_dir'] = data_dir
	if output_dir:
		config_dict['output_dir'] = output_dir

	return _to_namespace(config_dict)


def load_training_history(checkpoint_dir):
	history_path = os.path.join(checkpoint_dir, 'training_log.json')
	if not os.path.exists(history_path):
		return []

	with open(history_path, 'r', encoding='utf-8') as f:
		history = json.load(f)

	if not isinstance(history, list):
		raise ValueError(f"Expected training_log.json to contain a list, got {type(history)}")

	return history


def save_training_history(history, checkpoint_dir):
	history_path = os.path.join(checkpoint_dir, 'training_log.json')
	with open(history_path, 'w', encoding='utf-8') as f:
		json.dump(history, f, indent=2)


def save_training_config(config, checkpoint_dir, args=None):
	config_dict = {k: _to_serializable(v) for k, v in vars(config).items()}
	if args is not None:
		config_dict['resume_cli_args'] = {k: _to_serializable(v) for k, v in vars(args).items()}

	config_path = os.path.join(checkpoint_dir, 'training_config.json')
	with open(config_path, 'w', encoding='utf-8') as f:
		json.dump(config_dict, f, indent=2)


def resolve_checkpoint_path(checkpoint_dir, checkpoint_name='latest_checkpoint.pt'):
	checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
	if os.path.exists(checkpoint_path):
		return checkpoint_path

	fallback_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
	if os.path.exists(fallback_path):
		return fallback_path

	best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
	if os.path.exists(best_path):
		return best_path

	raise FileNotFoundError(
		f"No checkpoint found in {checkpoint_dir}. Expected latest_checkpoint.pt or best_checkpoint.pt."
	)


def load_model_state(model, checkpoint_path, device):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict):
		state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
	else:
		state_dict = checkpoint
	model.load_state_dict(state_dict, strict=False)
	return checkpoint


def infer_vocab_sizes(data_dir):
	with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
		num_entities = len(json.load(f))
	with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
		num_relations = len(json.load(f))
	return num_entities, num_relations


def resume_training(args):
	checkpoint_dir = os.path.abspath(args.checkpoint_dir)
	if not os.path.isdir(checkpoint_dir):
		raise NotADirectoryError(f"Checkpoint directory does not exist: {checkpoint_dir}")

	config = load_training_config(
		checkpoint_dir=checkpoint_dir,
		data_dir=args.data_dir,
		output_dir=checkpoint_dir,
	)

	if not hasattr(config, 'data_dir') or not config.data_dir:
		raise ValueError("data_dir must be provided either via training_config.json or --data_dir")

	if not hasattr(config, 'output_dir') or not config.output_dir:
		config.output_dir = checkpoint_dir

	if args.output_dir:
		config.output_dir = args.output_dir

	os.makedirs(config.output_dir, exist_ok=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	print(f"Checkpoint directory: {checkpoint_dir}")
	print(f"Data directory: {config.data_dir}")
	print(f"Output directory: {config.output_dir}")

	history = load_training_history(checkpoint_dir)
	start_epoch = len(history)
	print(f"Loaded {start_epoch} logged epochs from training_log.json")

	if args.num_epochs is not None:
		config.num_epochs = args.num_epochs

	if start_epoch >= int(config.num_epochs):
		print(f"Nothing to resume: start_epoch={start_epoch} is already >= num_epochs={config.num_epochs}")
		return

	num_entities, num_relations = infer_vocab_sizes(config.data_dir)
	config.num_entities = num_entities
	config.num_relations = num_relations

	save_training_config(config, config.output_dir, args=args)

	print("Loading model...")
	model = GWM(config).to(device)

	entity_emb_path = os.path.join(config.data_dir, 'entity_text_embeddings.pt')
	relation_emb_path = os.path.join(config.data_dir, 'relation_text_embeddings.pt')
	if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
		raise FileNotFoundError(
			"Missing precomputed text embedding cache files. "
			"Expected entity_text_embeddings.pt and relation_text_embeddings.pt in data_dir."
		)

	cache_device = getattr(config, 'text_cache_device', 'cpu')
	model.load_precomputed_text_embedding_cache(
		entity_source=entity_emb_path,
		relation_source=relation_emb_path,
		cache_device=cache_device,
	)

	checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_name=args.checkpoint_name)
	print(f"Restoring weights from: {checkpoint_path}")
	checkpoint = load_model_state(model, checkpoint_path, device)

	collate_fn = CollateFN()
	train_dataset = GWMDataset(config.data_dir, split='train')
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		collate_fn=collate_fn,
		num_workers=4,
		pin_memory=(device.type == 'cuda'),
		drop_last=True,
	)

	valid_loader = None
	if os.path.exists(os.path.join(config.data_dir, 'valid_triples.pt')):
		valid_dataset = GWMDataset(config.data_dir, split='valid')
		valid_loader = DataLoader(
			valid_dataset,
			batch_size=config.batch_size,
			shuffle=False,
			collate_fn=collate_fn,
			num_workers=2,
			pin_memory=(device.type == 'cuda'),
			drop_last=False,
		)

	hr_map = None
	entity_loader = None
	if valid_loader is not None:
		hr_map = load_hr_map_for_filtering(
			config.data_dir,
			preferred_ground_truth_file='ground_truth_train.json',
			fallback_splits=['train'],
		)
		candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(int(config.batch_size), 256)))
		entity_loader = build_entity_loader(
			data_dir=config.data_dir,
			batch_size=candidate_batch_size,
			num_workers=2,
		)

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
	if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			print("Restored optimizer state from checkpoint.")
		except Exception as exc:
			print(f"Warning: could not restore optimizer state ({exc}); starting optimizer fresh.")

	early_stopping = EarlyStopping(
		patience=getattr(config, 'early_stopping_patience', getattr(config, 'early_stopping', 10)),
		mode='max',
	)
	best_mrr = 0.0
	if history:
		best_mrr = max((float(item.get('val_mrr', 0.0)) for item in history if isinstance(item, dict)), default=0.0)
		early_stopping.best_value = best_mrr
		print(f"Resuming with best logged validation MRR: {best_mrr:.4f}")

	if hasattr(model, 'reset_alpha_stats'):
		model.reset_alpha_stats()

	for epoch in range(start_epoch, int(config.num_epochs)):
		model.train()
		total_loss = 0.0

		if hasattr(model, 'reset_alpha_stats'):
			model.reset_alpha_stats()

		pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Resume Train]")
		for batch in pbar:
			h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
			r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
			t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
			context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

			optimizer.zero_grad()
			query_vector = model(h_batch, r_batch, context_batch)
			t_fused = model.encode_target(t_batch)
			loss, _ = model.compute_loss(query_vector, t_fused)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			total_loss += loss.item()
			pbar.set_postfix({'loss': loss.item()})

		avg_train_loss = total_loss / max(1, len(train_loader))
		train_alpha = model.get_alpha_mean(reset=True) if hasattr(model, 'get_alpha_mean') else None
		print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")
		if train_alpha is not None:
			print(f"Epoch {epoch + 1} Train Alpha (text weight): {train_alpha:.4f}")

		eval_every = getattr(config, 'eval_every', 1)
		epoch_log = {
			'epoch': epoch + 1,
			'train_loss': avg_train_loss,
		}

		if valid_loader is not None and (epoch + 1) % eval_every == 0:
			model.eval()
			if hasattr(model, 'reset_alpha_stats'):
				model.reset_alpha_stats()

			all_entity_embeddings = encode_all_entities_as_targets(
				model=model,
				entity_loader=entity_loader,
				device=device,
				desc='Encoding Validation Candidates',
			)
			val_metrics = compute_filtered_ranking_metrics(
				model=model,
				data_loader=valid_loader,
				all_entity_embeddings=all_entity_embeddings,
				hr_map=hr_map,
				device=device,
				desc='Filtered Validation',
			)

			val_mrr = val_metrics['MRR']
			val_h1 = val_metrics['Hits@1']
			val_h3 = val_metrics['Hits@3']
			val_h10 = val_metrics['Hits@10']
			val_mr = val_metrics['MR']
			val_alpha = model.get_alpha_mean(reset=True) if hasattr(model, 'get_alpha_mean') else None

			print(
				f"Epoch {epoch + 1} Val (Filtered) | "
				f"MRR: {val_mrr:.4f} | MR: {val_mr:.2f} | "
				f"Hits@1: {val_h1:.4f} | Hits@3: {val_h3:.4f} | Hits@10: {val_h10:.4f}"
			)
			if val_alpha is not None:
				print(f"Epoch {epoch + 1} Val Alpha (text weight): {val_alpha:.4f}")

			epoch_log.update({
				'val_mrr': val_mrr,
				'val_mr': val_mr,
				'val_hits1': val_h1,
				'val_hits3': val_h3,
				'val_hits10': val_h10,
			})
			if train_alpha is not None:
				epoch_log['train_alpha'] = train_alpha
			if val_alpha is not None:
				epoch_log['val_alpha'] = val_alpha

			if val_mrr > best_mrr:
				best_mrr = val_mrr
				torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_checkpoint.pt'))

			if early_stopping(val_mrr):
				print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
				print(f"  Best MRR: {early_stopping.best_value:.4f}")
				print(f"  No improvement for {early_stopping.patience} epochs")
				history.append(epoch_log)
				save_training_history(history, config.output_dir)
				torch.save(model.state_dict(), os.path.join(config.output_dir, 'latest_checkpoint.pt'))
				break
		else:
			if train_alpha is not None:
				epoch_log['train_alpha'] = train_alpha

		history.append(epoch_log)
		save_training_history(history, config.output_dir)
		torch.save(model.state_dict(), os.path.join(config.output_dir, 'latest_checkpoint.pt'))

	print("Resume training finished.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--checkpoint_dir',
		type=str,
		required=True,
		help='Directory containing best_checkpoint.pt, latest_checkpoint.pt, training_config.json, training_log.json',
	)
	parser.add_argument('--data_dir', type=str, default=None, help='Override processed data directory if needed')
	parser.add_argument('--output_dir', type=str, default=None, help='Override output directory; defaults to checkpoint_dir')
	parser.add_argument('--num_epochs', type=int, default=None, help='Optional new total number of epochs to train to')
	parser.add_argument(
		'--checkpoint_name',
		type=str,
		default='latest_checkpoint.pt',
		help='Checkpoint file to restore from (default: latest_checkpoint.pt)',
	)
	args = parser.parse_args()
	resume_training(args)
