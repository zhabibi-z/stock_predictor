"""ML Stock Direction Predictor — source package."""

import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def load_config() -> dict:
    with open(os.path.abspath(_CONFIG_PATH)) as f:
        return yaml.safe_load(f)
