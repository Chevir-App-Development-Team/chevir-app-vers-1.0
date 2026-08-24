import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw_videos')
EXTRACTED_KEYPOINTS_DIR = os.path.join(DATA_DIR, 'extracted_keypoints')

# Model parameters
SEQUENCE_LENGTH = 30
INPUT_SIZE = 1662
NUM_CLASSES = 3  # Update as dataset grows
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 5
