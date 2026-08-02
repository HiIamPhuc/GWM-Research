# Graph World Model for Static Knowledge Graph Completion

## Status

This document describes the structural-only GWM experimental branch on
2026-08-02. The current code decodes four routed next-state slots and scores
their spherical mixture. It retains the context Transformer, shared
forward/inverse relation parameterization, full-entity cross-entropy, and masked
world-state reconstruction. This routed-slot combination has not yet been
trained, so historical results below are labeled by their actual variants.

## Problem and Protocol

Given a query `(h, r, ?)`, GWM predicts a tail entity from the complete entity
set. Training triples are augmented with `(t, r_inv, h)`, while validation and
test triples remain in their original direction. Evaluation constructs inverse
queries on the fly and reports filtered forward, backward, and micro-averaged
ranking metrics. Validation filtering uses train and validation truths; test
filtering uses train, validation, and test truths.

The system is transductive: every entity and base relation has a learned ID
embedding, and no text encoder or textual embedding is active.

## End-to-End Architecture

### 1. Relation-diverse local observations

Preprocessing builds a directed adjacency list from the inverse-augmented
training graph. For each entity, it precomputes at most `k` outgoing facts. The
selector first covers distinct relation types and then randomly fills remaining
slots. Selection is seeded and fixed for the whole run.

For a training or evaluation triple, the context is

`N(h) = {(r_i, e_i)}`.

If the exact answer edge `(r, t)` occurs in this memory, its mask is disabled to
prevent direct answer leakage. The vacant slot is not replaced.

### 2. Structural entity and relation representations

All entities share one trainable embedding table `E` of dimension 512. The same
table represents query heads, context entities, reconstruction targets, and
candidate tails.

Forward and inverse relations share a base row. For relation ID `r`, let `b(r)`
be its base-relation ID and `d(r)` be 0 for forward or 1 for inverse. The encoded
relation is

`z_r = LN(B[b(r)] + D[d(r)] + d(r) A_inv B[b(r)])`.

Here `B` is the base-relation table, `D` contains two direction embeddings, and
`A_inv` is one shared linear inverse adapter. It is initialized to zero, so
forward and inverse relations initially differ only by direction embeddings.

### 3. Head-centered world-state construction

The context memory contains one head token and one token per observed fact:

`x_h = E[h] + role_head`,

`x_i = LN(E[e_i] + z_{r_i}) + role_fact`.

A Transformer encoder processes `[x_h, x_1, ..., x_k]` with padding masks and no
positional encoding. Its outputs form a contextualized, permutation-equivariant
local memory `M_h`. The query relation is deliberately absent from this stage;
relation-aware use of memory is delegated to the transition decoder.

### 4. Relation-conditioned latent transition

The decoder receives one relation token and four learned next-state slots:

`[z_r + role_relation, next_state_1, ..., next_state_4]`.

A mask prevents the relation token from reading the state slots. The first,
anchor slot reads only the relation and itself, preserving the former two-token
transition path. Alternative slots can coordinate with all target-side tokens.
All tokens cross-attend to `M_h`. After two decoder layers, the four state
outputs are projected and normalized:

`q_(h,r,p) = normalize(W_next Decoder(z_r, next_slots; M_h)[p])`.

A relation-conditioned router produces mixture weights

`pi(r) = softmax(W_router z_r)`.

The anchor begins with more than 94% of the router mass; other slots can gain
mass during training for relations that benefit from multiple successor modes.

This is the core world-model operation: construct a latent state from local
observations, apply an action-like relation condition, and decode a predicted
next state. It is a deterministic one-step latent transition, not a graph walk,
multi-step rollout, or generative environment simulator.

### 5. Observation retrieval

Every candidate tail is normalized from the shared entity table:

`v_t = normalize(E[t])`.

Candidates are ranked by a relation-weighted mixture of spherical components:

`s(h,r,t) = log sum_p pi_p(r) exp(q_(h,r,p)^T v_t / tau)`.

FB15k-237 uses `tau=0.10`; WN18RR uses `tau=0.07`.

### 6. Training objectives

The principal objective is triple-level, unfiltered full-entity cross-entropy:

`L_KGC = -log exp(s(h,r,t)) / sum_(e in Entities) exp(s(h,r,e))`.

Each sampled triple has one target. Other known valid tails remain in the
denominator and are therefore treated as negatives.

Masked world-state reconstruction randomly replaces the head token with a
learned mask token, encodes its context, and retrieves the original head from
all entities using the same dot-product codebook. The total loss is

`L = L_KGC + lambda_state L_state`, with `lambda_state=0.1`.

The reconstruction sampling ratio is 0.20 on FB15k-237 and 0.05 on WN18RR.
Training uses AdamW, cosine decay with warmup, gradient clipping, and early
stopping on bidirectional validation micro-MRR.

## Measured Results

The closest plain dot-product Transformer artifacts are from 2026-jul-28:

| Dataset | MRR | H@1 | H@3 | H@10 | Forward MRR | Backward MRR |
|---|---:|---:|---:|---:|---:|---:|
| FB15k-237 | 0.3436 | 0.2541 | 0.3742 | 0.5255 | 0.4419 | 0.2454 |
| WN18RR | 0.4594 | 0.4202 | 0.4735 | 0.5375 | 0.4828 | 0.4359 |

The strongest recent dot-product artifacts, from 2026-jul-29, additionally used
query-context retrieval and masked reconstruction:

| Dataset | MRR | H@1 | H@3 | H@10 | Forward MRR | Backward MRR |
|---|---:|---:|---:|---:|---:|---:|
| FB15k-237 | 0.3495 | 0.2560 | 0.3842 | 0.5368 | 0.4418 | 0.2572 |
| WN18RR | 0.4663 | 0.4244 | 0.4802 | 0.5536 | 0.4938 | 0.4388 |

These are not results for the routed-slot code and should not be presented as
such. A controlled rerun is required.

## Arity Evidence from the Strongest Dot Variant

| Dataset | 1-1 MRR | 1-N MRR | N-1 MRR | N-N MRR |
|---|---:|---:|---:|---:|
| FB15k-237 | 0.4909 | 0.1169 | 0.6997 | 0.3258 |
| WN18RR | 0.9707 | 0.1017 | 0.2691 | 0.9352 |

FB15k-237 is dominated by N-N queries, where performance is moderate. WN18RR is
strongly bimodal: 1-1 and N-N relations are modeled extremely well, while 1-N
and N-1 relations remain weak. This shows that aggregate MRR hides a systematic
failure on multi-answer transition directions.

## Experiment History and Negative Results

- Replacing the LSTM with a context-memory Transformer and using full-entity CE
  produced the largest recent improvement.
- Shared forward/inverse relation parameters improved WN18RR directional balance
  but did not remove the FB15k-237 gap.
- Query-context retrieval plus reconstruction improved the July 28 result by
  about 0.6 MRR points on FB15k-237 and 0.7 on WN18RR, but the two changes were
  combined and their individual contributions are not isolated.
- An atomic-fact Transformer reduced MRR to 0.3435 and 0.4487.
- Separate state and target entity tables reduced MRR to 0.3363 and 0.4518.
- Diagonal-Gaussian next-state scoring reached approximately 0.3049 and 0.4121.
- A 16-basis asymmetric low-rank readout reached 0.3120 and 0.4123. Its bounded
  coefficients saturated near an absolute mean of 0.94 and it was removed.
- Earlier LSTM particles already used LogSumExp, but they were generated by a
  post-LSTM projection and trained with in-batch negatives. The current test
  instead decodes slots directly, uses a relation router, and trains their final
  mixture with full-entity CE. Earlier multi-positive variants did not improve
  the main metrics consistently.

## Strengths

1. The architecture has a clear observation-state-transition-retrieval flow.
2. Context facts are encoded as a set rather than an arbitrary ordered sequence.
3. Training and evaluation support inverse directions without duplicating
   validation or test artifacts.
4. Forward and inverse relations share statistical strength while retaining a
   learned directional component.
5. Full-entity training directly matches all-entity retrieval at inference.
6. The architecture is compact compared with dual-codebook and complex decoder
   variants, and all structural representations are learned end to end.

## Weaknesses

1. **Fixed-width mixture and possible slot collapse.** Four slots can represent
   several successor modes, but the router may retain only the anchor or use all
   slots indiscriminately. Effective slot count must be analyzed by arity.
2. **False-negative supervision.** Full-entity CE penalizes other known valid
   tails because each triple supplies only one positive target.
3. **Directional gap.** The strongest FB15k-237 run has a 0.1846 absolute MRR
   gap between forward and backward prediction.
4. **Static, query-independent observations.** The same fixed neighborhood is
   used for every relation queried at a head. The decoder can attend selectively,
   but relevant evidence may not have survived precomputation.
5. **Shallow local evidence.** Only fixed one-hop training facts are observed;
   paths and dynamically retrieved evidence are absent.
6. **Weak fact composition.** A context fact is represented by an additive
   entity-relation token, which may not preserve direction or interaction
   patterns sufficiently.
7. **Restricted inverse operator.** Every inverse relation uses the same linear
   adapter. This shares information efficiently but may underfit relation-specific
   inverse behavior.
8. **Isotropic retrieval geometry.** A single cosine space and global temperature
   must serve relations with very different cardinalities and semantics.
9. **Auxiliary-task uncertainty.** Reconstruction may improve entity identity
   encoding without improving relation transitions, and its independent effect
   has not been isolated.
10. **Transductive-only representations.** Removing text simplifies the model but
    prevents encoding unseen entities and discards semantic evidence.
11. **Scalability.** Full-entity logits scale as `batch_size * num_entities`, and
    candidate embeddings/logits are held on the accelerator.
12. **Experimental variance is unknown.** Most comparisons use one seed, while
    several architecture changes were tested together.

## Current Bottlenecks, Ranked

1. **State/objective mismatch for multi-answer relations.** Routed slots improve
   state capacity, but single-positive CE still suppresses other valid answers.
2. **Relation-direction modeling on FB15k-237.** Backward MRR remains much lower
   even after inverse augmentation and parameter sharing.
3. **Useful context acquisition.** More elaborate context composition repeatedly
   failed, suggesting that evidence selection may matter more than encoder depth.
4. **Entangled ablations.** The exact current dot-plus-reconstruction baseline has
   no artifact, making architectural attribution unreliable.
5. **Optimization and calibration across relation families.** One temperature,
   one codebook, and one loss weighting serve highly heterogeneous relations.

## Questions for External Consultation

1. How should a world model represent a set-valued next state without sacrificing
   Hits@1, as the earlier fixed-particle model did?
2. Which objective supports all valid tails while retaining the strong optimization
   behavior of full-entity softmax?
3. How can relation-conditioned evidence retrieval remain efficient and avoid
   answer leakage in a transductive graph?
4. Should inverse actions use a relation-specific reversible operator, an
   algebraic constraint, or independent parameters with consistency loss?
5. Would a relation-family-conditioned metric or codebook address arity variation
   more effectively than changing the Transformer transition?
6. How should masked state reconstruction be redesigned so that it regularizes
   transition dynamics rather than only entity identification?
