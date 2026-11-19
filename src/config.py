import torch

SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"
NUM_TRAIN, NUM_VAL = 45_000, 5_000
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")