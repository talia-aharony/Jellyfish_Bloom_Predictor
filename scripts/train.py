#!/usr/bin/env python3
"""Entrypoint wrapper for model training."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import train_all_models


if __name__ == "__main__":
    train_all_models()
