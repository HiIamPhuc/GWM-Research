"""Model factory retained for study scripts.

The current starting point has one architecture only: trainable structural
embeddings followed by a two-step head-relation LSTM.
"""

from model.model import GWM


def build_model(config):
    return GWM(config)
