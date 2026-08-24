import cv2
import numpy as np

class SkeletonVisualizer:
    """
    Utility for drawing custom visualization for the UI, separate from MediaPipe's default.
    Useful for debugging and presenting the Text-to-Sign Avatar skeleton.
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        
    def draw_skeleton(self, keypoints):
        """
        Takes flattened keypoints and draws a basic 2D projection on a blank canvas.
        keypoints shape: (1662,) 
        """
        # Create a black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        cv2.putText(canvas, "3D Avatar Skeleton View", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Simulate drawing some joints
        center_x, center_y = self.width // 2, self.height // 2
        cv2.circle(canvas, (center_x, center_y - 100), 30, (0, 255, 0), -1) # Head
        cv2.line(canvas, (center_x, center_y - 70), (center_x, center_y + 50), (255, 0, 0), 5) # Spine
        cv2.line(canvas, (center_x, center_y - 50), (center_x - 60, center_y), (0, 0, 255), 5) # Left Arm
        cv2.line(canvas, (center_x, center_y - 50), (center_x + 60, center_y), (0, 0, 255), 5) # Right Arm
        
        return canvas
