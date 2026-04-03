#!/usr/bin/env python3
"""Entrypoint wrapper for prediction examples."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from predict_example import main


if __name__ == "__main__":
    main()
