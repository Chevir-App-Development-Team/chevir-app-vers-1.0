import os
import numpy as np
from src.config import EXTRACTED_KEYPOINTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_keypoint_sequence(sequence, label, video_id):
    """
    Saves a sequence of keypoints to disk as a numpy array.
    """
    label_dir = os.path.join(EXTRACTED_KEYPOINTS_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    file_path = os.path.join(label_dir, f"{video_id}.npy")
    np.save(file_path, sequence)
    logger.info(f"Saved keypoint sequence to {file_path}")

def load_keypoint_sequence(file_path):
    """
    Loads a sequence of keypoints from disk.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    return np.load(file_path)
