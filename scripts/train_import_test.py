from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import logging
import torch
import torch.nn as nn
import yaml
from torch.utils.data import random_split
from src.data.dataset import StreamSimDataset, compute_dataset_normalizer
from src.models.baseline import DensityProfileCNN, compute_1d_profile
from src.models.gnn import StreamGNNMultiTask
from src.models.utils import build_scheduler, count_parameters, load_checkpoint, save_checkpoint
print("ALL IMPORTS OK")
