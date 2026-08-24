import mediapipe as mp
import numpy as np

class HolisticExtractor:
    """İşaret dili videolarından yüz, beden ve el anahtar noktalarını (ROI) çıkarır."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def extract_keypoints(self, image_rgb):
        results = self.holistic.process(image_rgb)
        
        # Hata önleme (Error Handling) - Eksik veriyi sıfırlarla doldur (Padding)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        
        # Tüm vektörleri tek bir numpy array'de birleştir
        return np.concatenate([pose, face, lh, rh])