import numpy as np

class PoseGenerator:
    def __init__(self):
        """
        A mock dictionary mapping standard glosses to dummy keypoint arrays.
        Demonstrates how text (gloss) will drive a 3D avatar.
        """
        self.frames_per_gloss = 10
        self.keypoints_size = 1662
        
        # Mock dictionary: Gloss -> Tensor of shape (frames_per_gloss, keypoints_size)
        self.gloss_dict = {
            "BEN": np.random.rand(self.frames_per_gloss, self.keypoints_size),
            "OKUL": np.random.rand(self.frames_per_gloss, self.keypoints_size),
            "GİTMEK": np.random.rand(self.frames_per_gloss, self.keypoints_size),
            "SEN": np.random.rand(self.frames_per_gloss, self.keypoints_size),
            "SEVMEK": np.random.rand(self.frames_per_gloss, self.keypoints_size),
            # Default idle state
            "IDLE": np.zeros((self.frames_per_gloss, self.keypoints_size))
        }

    def generate_pose(self, gloss_array):
        """
        Takes an array of glosses and returns a concatenated sequence of keypoints.
        """
        sequence = []
        for gloss in gloss_array:
            pose_frames = self.gloss_dict.get(gloss, self.gloss_dict["IDLE"])
            sequence.append(pose_frames)
            
        if not sequence:
            return np.empty((0, self.keypoints_size))
            
        return np.concatenate(sequence, axis=0)
