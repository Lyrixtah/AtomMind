import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
HIDDEN_SIZE = 512
NUM_ATTENTION_HEADS = 8
DEN_LAYERS = 30
GKB_LAYERS = 20
SRM_LAYERS = 10
OAM_LAYERS = 10
CEN_LAYERS = 15
MAX_SEQ_LEN = 512

# Training
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5

# Domains
DOMAINS = ["math", "physics", "chemistry", "biology"]

# Paths
DATA_PATH = "./data/"
LOG_PATH = "./logs/"
SAVE_PATH = "./checkpoints/"
