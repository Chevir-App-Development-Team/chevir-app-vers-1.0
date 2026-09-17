"""
Extract two-hand word poses from video using MediaPipe HandLandmarker.

Output format matches the letter poses in public/tercume/assets/poses.json exactly:
  - Each frame has 21 points per hand (left frames & right frames stored separately)
  - Points are {x, y, z} objects in MediaPipe normalized image coordinates (0-1 range)
  - Y is DOWN (raw MediaPipe convention — retarget.js handles the inversion)
  - world landmarks are also extracted (metric 3D, in meters)
  - size [width, height] is stored per frame
  - handedness is stored per frame

The VRM pipeline (retarget.js → handTarget → solveArm → solveFingers) was built for
this exact format. Do NOT pre-invert Y or pre-scale coordinates.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import numpy as np


def process_video(video_path, word_name, output_path):
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

    # Target ~20 fps
    sample_rate = max(1, round(fps / 20))
    print(f"Source: {fps} fps, {total_frames} frames, sampling every {sample_rate}")

    left_norm_frames = []  # list of 21-point frames for left hand
    right_norm_frames = []  # list of 21-point frames for right hand
    left_world_frames = []
    right_world_frames = []
    sizes = []

    frame_idx = 0
    last_left_norm = None
    last_right_norm = None
    last_left_world = None
    last_right_world = None
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
            left_norm = last_left_norm
            right_norm = last_right_norm
            left_world = last_left_world
            right_world = last_right_world
            skipped += 1
        else:
            # Extract normalized + world landmarks for each detected hand
            hand_data = []
            for i in range(n_hands):
                hl = result.hand_landmarks[i]
                norm_pts = [{"x": round(lm.x, 5), "y": round(lm.y, 5), "z": round(lm.z, 5)} for lm in hl]
    
                wl = result.hand_world_landmarks[i] if result.hand_world_landmarks else None
                world_pts = None
                if wl:
                    world_pts = [[round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)] for lm in wl]
    
                hand_data.append({
                    "norm": norm_pts,
                    "world": world_pts,
                    "wrist_x": hl[0].x,
                })
    
            # Sort by wrist X (left side of image = smaller X)
            hand_data.sort(key=lambda h: h["wrist_x"])
    
            if n_hands == 2:
                # Person facing camera: left side of image (smaller x) is their RIGHT hand.
                right_norm = hand_data[0]["norm"]
                left_norm = hand_data[1]["norm"]
                right_world = hand_data[0]["world"]
                left_world = hand_data[1]["world"]
            elif n_hands == 1:
                wrist_x = hand_data[0]["wrist_x"]
                if last_left_norm is not None and last_right_norm is not None:
                    dist_l = abs(wrist_x - last_left_norm[0]["x"])
                    dist_r = abs(wrist_x - last_right_norm[0]["x"])
                    if dist_r < dist_l:
                        right_norm = hand_data[0]["norm"]
                        left_norm = last_left_norm
                        right_world = hand_data[0]["world"]
                        left_world = last_left_world
                    else:
                        right_norm = last_right_norm
                        left_norm = hand_data[0]["norm"]
                        right_world = last_right_world
                        left_world = hand_data[0]["world"]
                elif last_left_norm is not None:
                    # We only have a left hand from before.
                    # Just map the current hand to left, because usually they don't swap mid-air.
                    left_norm = hand_data[0]["norm"]
                    right_norm = last_right_norm
                    left_world = hand_data[0]["world"]
                    right_world = last_right_world
                elif last_right_norm is not None:
                    right_norm = hand_data[0]["norm"]
                    left_norm = last_left_norm
                    right_world = hand_data[0]["world"]
                    left_world = last_left_world
                else:
                    # First frame(s) have only 1 hand. Guess left/right based on screen position.
                    if wrist_x < 0.5:
                        # Left side of screen = Signer's RIGHT hand
                        right_norm = hand_data[0]["norm"]
                        right_world = hand_data[0]["world"]
                        left_norm = None
                        left_world = None
                    else:
                        # Right side of screen = Signer's LEFT hand
                        left_norm = hand_data[0]["norm"]
                        left_world = hand_data[0]["world"]
                        right_norm = None
                        right_world = None
                    one_hand_filled += 1

        last_left_norm = left_norm
        last_right_norm = right_norm
        last_left_world = left_world
        last_right_world = right_world

        left_norm_frames.append(left_norm)
        right_norm_frames.append(right_norm)
        left_world_frames.append(left_world)
        right_world_frames.append(right_world)
        sizes.append([width, height])

    cap.release()

    if not left_norm_frames:
        print("No poses extracted.")
        return

    print(f"Extracted {len(left_norm_frames)} frames ({skipped} skipped, {one_hand_filled} single-hand fills)")

    # Build word entry matching letter format exactly
    # Each hand stored as its own pose entry with type, frames, world, size, handedness
    word_entry = {
        "type": "dynamic",
        "left": {
            "type": "dynamic",
            "frames": left_norm_frames,
            "world": left_world_frames,
            "size": sizes,
            "handedness": ["Left"] * len(left_norm_frames),
        },
        "right": {
            "type": "dynamic",
            "frames": right_norm_frames,
            "world": right_world_frames,
            "size": sizes,
            "handedness": ["Right"] * len(right_norm_frames),
        },
    }

    # Load existing poses.json and merge
    poses_path = 'public/tercume/assets/poses.json'
    with open(poses_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'words' not in data:
        data['words'] = {}

    data['words'][word_name] = word_entry
    print(f"Added word to dictionary")

    with open(poses_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    import os
    size_kb = os.path.getsize(poses_path) / 1024
    print(f"Output: {poses_path} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        video = sys.argv[1]
        name = sys.argv[2]
        process_video(video, name, 'public/tercume/assets/poses.json')
    else:
        print("Usage: python extract_word.py <video.mp4> <word_name>")
