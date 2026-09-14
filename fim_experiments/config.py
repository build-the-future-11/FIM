import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training
BATCH_SIZE = 32
STEPS = 50
EPOCHS = 20
LR = 1e-3

# Systems to test
SYSTEMS = ["lorenz", "fractional", "levy"]

# Models
MODELS = ["mlp", "fim"]

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 128

# Evaluation
ROLLOUT_STEPS = 50
EVAL_BATCH_SIZE = 32

# Logging
LOG_INTERVAL = 1

# Reproducibility
SEED = 42