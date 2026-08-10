"""Model factory retained for study scripts.

The current experiment has one architecture only: a residual text-fused
context-memory encoder followed by a causal state-action Transformer decoder.
"""

from model.model import GWM


def build_model(config):
    return GWM(config)
