"""Model factory retained for study scripts."""

from model.model import GWM


def build_model(config):
    return GWM(config)
