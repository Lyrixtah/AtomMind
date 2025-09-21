"""Configuration file for AtomMind project."""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
HIDDEN_SIZE = 512
NUM_ATTENTION_HEADS = 8

# Layer counts for different expert modules
DEN_LAYERS = 30  # Domain Expert Network
GKB_LAYERS = 20  # General Knowledge Backbone
SRM_LAYERS = 10  # Symbolic Reasoning Module
OAM_LAYERS = 10  # Optimization & Algorithmic Module
CEN_LAYERS = 15  # Chat Expert Network

# Sequence length
MAX_SEQ_LEN = 512

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5

# Supported domains
DOMAINS = ["math", "physics", "chemistry", "biology"]

# Paths
DATA_PATH = "./data/"
LOG_PATH = "./logs/"
SAVE_PATH = "./checkpoints/"
