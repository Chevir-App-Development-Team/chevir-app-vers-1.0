import cv2
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.module_a_s2t.extractor import HolisticExtractor
from src.module_a_s2t.model import SignLanguageLSTM
from src.module_b_t2s.nlp_gloss import NLPGlossPipeline
from src.module_b_t2s.pose_generator import PoseGenerator

def run_sign_to_text():
    print("\n--- Running Sign-to-Text (Webcam) ---")
    print("Press 'q' to quit.")
    
    extractor = HolisticExtractor()
    cap = cv2.VideoCapture(0)
    
    # Using a dummy model since we don't have a trained one
    model = SignLanguageLSTM(num_classes=3)
    model.eval()
    
    sequence = []
    classes = ["HELLO", "THANK YOU", "I LOVE YOU"]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
            
        results = extractor.process_image(frame)
        keypoints = extractor.extract_keypoints(results)
        extractor.draw_landmarks(frame, results)
        
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep last 30 frames
        
        if len(sequence) == 30:
            input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor)
                class_idx = torch.argmax(prediction, dim=1).item()
                pred_text = classes[class_idx]
                
            cv2.putText(frame, f"Pred: {pred_text}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.imshow('Chevir Sign-to-Text', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_text_to_sign():
    print("\n--- Running Text-to-Sign Pipeline ---")
    nlp_pipeline = NLPGlossPipeline()
    pose_gen = PoseGenerator()
    
    while True:
        text_input = input("\nEnter Turkish/Azerbaijani text (or 'q' to quit): ")
        if text_input.lower() == 'q':
            break
            
        gloss_array = nlp_pipeline.process(text_input)
        print(f"1. Extracted Gloss Array: {gloss_array}")
        
        generated_pose = pose_gen.generate_pose(gloss_array)
        print(f"2. Generated Pose Tensor Shape: {generated_pose.shape}")
        print("   (This tensor would be sent to a 3D Avatar Engine/Unity)")

def main():
    while True:
        print("\n=== Chevir AI Accessibility Layer ===")
        print("1. Test Webcam Sign-to-Text Extraction")
        print("2. Test Text-to-Gloss Pipeline")
        print("3. Quit")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '1':
            run_sign_to_text()
        elif choice == '2':
            run_text_to_sign()
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()