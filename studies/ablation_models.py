"""Model factory retained for study scripts.

The current experiment has one architecture only: trainable structural
embeddings followed by a causal two-token Transformer transition.
"""

from model.model import GWM


def build_model(config):
    return GWM(config)
