"""Model factory retained for study scripts.

The current experiment has one architecture only: trainable structural
embeddings with relation-gated context followed by a two-step LSTM.
"""

from model.model import GWM


def build_model(config):
    return GWM(config)
