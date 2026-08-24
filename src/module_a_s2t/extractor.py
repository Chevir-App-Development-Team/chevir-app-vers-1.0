import cv2
import mediapipe as mp
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class HolisticExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info("HolisticExtractor initialized.")

    def process_image(self, image):
        """Processes the image and returns raw results."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        return results

    def normalize_keypoints(self, keypoints, center_point_idx=0):
        """
        Normalizes keypoints to make them translation-invariant.
        Currently simple mean-centering.
        """
        # Reshape to (N, features) to allow vector ops
        if np.all(keypoints == 0):
            return keypoints
        
        # Simple normalization: center by mean of non-zero coordinates
        non_zero = keypoints[keypoints != 0]
        if len(non_zero) > 0:
            mean_val = np.mean(non_zero)
            std_val = np.std(non_zero) + 1e-6
            # Only normalize the non-zero elements
            normalized = np.copy(keypoints)
            normalized[keypoints != 0] = (keypoints[keypoints != 0] - mean_val) / std_val
            return normalized
        return keypoints

    def extract_keypoints(self, results, normalize=True):
        """
        Extracts pose, face, left_hand, right_hand keypoints from MediaPipe results.
        Returns a flattened numpy array of 1662 features (33*4 + 468*3 + 21*3 + 21*3).
        Zero-pads if a landmark is not detected.
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        
        concat_kps = np.concatenate([pose, face, lh, rh])
        if normalize:
            concat_kps = self.normalize_keypoints(concat_kps)
            
        return concat_kps

    def draw_landmarks(self, image, results):
        """
        Draws the landmarks on the provided image (in-place).
        """
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, 
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
                                 
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
                                 
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))