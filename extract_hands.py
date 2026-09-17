"""
Extract two-hand 3D poses from Welcome.mp4 using MediaPipe HandLandmarker.

Output: poses.json — array of frames, each frame is 42 points (21 left + 21 right),
each point [x, y, z] in a normalized coordinate system where:
  - Scene is centered on the midpoint of both wrists
  - Y is up (inverted from MediaPipe's Y-down)
  - Scale: middle-finger-tip distance from wrist ≈ 1.0

Also handles frames where only 1 hand is detected by carrying forward the
last known position of the missing hand. Frames where 0 hands are detected
are skipped entirely.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import math
import numpy as np


def normalize_all_poses(raw_poses):
    """
    Normalizes all frames together using a GLOBAL center and scale.
    This preserves the translational movement of the hands across frames.
    """
    raw_poses = np.array(raw_poses) # (num_frames, 42, 3)
    
    # Calculate a global center across all frames (using the midpoint of wrists)
    all_centers = (raw_poses[:, 0] + raw_poses[:, 21]) / 2
    global_center = np.mean(all_centers, axis=0)
    
    # Calculate a global scale across all frames
    # Scale: average distance from each wrist to its middle finger tip
    d1 = np.linalg.norm(raw_poses[:, 12] - raw_poses[:, 0], axis=1)
    d2 = np.linalg.norm(raw_poses[:, 33] - raw_poses[:, 21], axis=1)
    global_scale = np.mean((d1 + d2) / 2)
    
    # Apply global translation and scale
    normalized = raw_poses - global_center
    if global_scale > 0:
        normalized *= (1.0 / global_scale)
        
    return normalized.tolist()


def process_video(video_path, output_path):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Target ~20 fps for smooth playback (every 3rd frame of 60fps source)
    sample_rate = max(1, round(fps / 20))
    
    print(f"Source: {fps} fps, {total_frames} frames, sampling every {sample_rate}")
    
    poses = []
    frame_idx = 0
    last_left = None
    last_right = None
    skipped = 0
    one_hand_filled = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue
            
        height, width, _ = frame.shape
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = detector.detect(mp_image)
        
        n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        
        if n_hands == 0:
            skipped += 1
            continue
        
        # Sort detected hands by wrist X coordinate (left hand has smaller X in mirrored view)
        hand_data = []
        for i in range(n_hands):
            hl = result.hand_landmarks[i]
            pts = []
            for lm in hl:
                # Convert to pixel space to keep proportions correct
                # Invert Y so up is positive
                pts.append([lm.x * width, -(lm.y * height), lm.z * width])
            hand_data.append(pts)
        
        # Sort by wrist X so we get consistent left/right ordering
        hand_data.sort(key=lambda h: h[0][0])
        
        if n_hands == 2:
            left_hand = hand_data[0]
            right_hand = hand_data[1]
        elif n_hands == 1:
            # Decide if this is left or right based on X position
            wrist_x = hand_data[0][0][0]
            if last_left is not None and last_right is not None:
                # Compare to last known wrist positions
                dist_to_left = abs(wrist_x - last_left[0][0])
                dist_to_right = abs(wrist_x - last_right[0][0])
                if dist_to_left < dist_to_right:
                    left_hand = hand_data[0]
                    right_hand = last_right
                else:
                    left_hand = last_left
                    right_hand = hand_data[0]
            elif last_left is not None:
                right_hand = hand_data[0]
                left_hand = last_left
            elif last_right is not None:
                left_hand = hand_data[0]
                right_hand = last_right
            else:
                # First frame, only 1 hand - skip until we see 2
                skipped += 1
                continue
            one_hand_filled += 1
        
        last_left = left_hand
        last_right = right_hand
        
        full_pose = left_hand + right_hand  # 42 points
        poses.append(full_pose)
    
    cap.release()
    
    if len(poses) == 0:
        print("No poses extracted.")
        return
        
    normalized = normalize_all_poses(poses)
    
    # Round to 4 decimal places to keep file size reasonable
    rounded_poses = []
    for pose in normalized:
        rounded_poses.append([[round(x, 4), round(y, 4), round(z, 4)] for x, y, z in pose])
    
    print(f"Extracted {len(rounded_poses)} pose frames ({skipped} skipped, {one_hand_filled} single-hand fills)")
    
    with open(output_path, 'w') as f:
        json.dump(rounded_poses, f)
    
    # Print file size
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Output: {output_path} ({size_kb:.1f} KB)")


process_video(r'Sign2.mp4', 'poses_sign2.json')
