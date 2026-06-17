# src/utils.py
import os
import logging


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
