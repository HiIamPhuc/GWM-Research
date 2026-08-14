"""Model factory retained for study scripts.

The current experiment has one architecture only: structural PairRE-conditioned
graph memory followed by a relation-conditioned Transformer decoder.
"""

from model.model import GWM


def build_model(config):
    return GWM(config)
