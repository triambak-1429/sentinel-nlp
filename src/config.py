import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_NAME = "roberta-base"
ONNX_MODEL_PATH = str(MODEL_DIR / "pii_detector.onnx")

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

# Label Definitions (IOB2 Format)
ID2LABEL = {
    0: "O",
    1: "B-EMAIL",
    2: "I-EMAIL",
    3: "B-API_KEY",
    4: "I-API_KEY",
    5: "B-PHONE",
    6: "I-PHONE",
    7: "B-SECRET",
    8: "I-SECRET",
}

LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)