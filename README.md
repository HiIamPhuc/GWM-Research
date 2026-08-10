# WST-KGC

Official research code for **WST-KGC**, a world-model-inspired approach to
knowledge graph completion. The model constructs a local world state from a
head entity's graph context and applies a role-aware LSTM transition. Learned
head/relation role embeddings define the transition order, while
relation-conditioned FiLM generates a mixture of plausible successor states
for candidate retrieval.

## Setup

```bash
pip install -r requirements.txt
```

The experiments use Python 3.10+ and PyTorch. CUDA is selected automatically
when available.

## Data preparation

Raw datasets are expected under `data/<dataset>/`. Preprocessing creates ID
vocabularies, train/validation/test tensors, filtered-evaluation maps, and
frozen text embeddings:

```bash
python -m utils.preprocess_data \
  --dataset fb15k-237 \
  --data_dir data/FB15k-237 \
  --output_dir data-processed/fb15k-237

python -m utils.compute_context \
  --data_dir data-processed/fb15k-237 \
  --k 10 \
  --seed 42
```

Supported dataset names are `fb15k-237`, `wn18rr`, `nell-995`, and `umls`.
Only training triples are augmented with inverse relations. Context facts are
sampled once and saved in `context_neighbors.pt`.

## Training

```bash
python train.py --config configs/fb15k-237.yaml
python train.py --config configs/wn18rr.yaml
```

Use `--data_dir` or `--output_dir` to override the corresponding config entry.
The best and latest checkpoints, effective configuration, and epoch log are
written to the configured output directory.

Text-only and structure-only ablations use the same command with one of the
`*_text.yaml` or `*_struct.yaml` configs.

`num_successor_modes` controls the number of FiLM-conditioned successor states.
Training logs their mixture entropy and average dominant-mode weight so mode
collapse can be diagnosed directly.

## Evaluation

```bash
python evaluate.py \
  --config configs/fb15k-237.yaml \
  --output_dir output/fb15k-237
```

Evaluation loads `best_checkpoint.pt` and reports bidirectional filtered MRR,
MR, Hits@1, Hits@3, and Hits@10. It also writes ranked test predictions to the
experiment directory.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover context masking, filtered in-batch negatives, fusion gradients,
embedding freezing, and bidirectional filtered evaluation.
