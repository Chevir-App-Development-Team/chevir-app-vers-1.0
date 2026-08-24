import cv2

class VideoStreamer:
    """NSosyal platformu için canlı kamera veya video dosyasından akış alır."""
    
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # MediaPipe için RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, frame_rgb
        
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()