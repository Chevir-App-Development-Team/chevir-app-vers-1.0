import cv2
import numpy as np
import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.module_a_s2t import HolisticExtractor, SignLanguageLSTM
from src.module_b_t2s import NLPGlossPipeline, PoseGenerator
from src.utils import get_logger
from src.config import INPUT_SIZE, NUM_CLASSES, SEQUENCE_LENGTH

logger = get_logger(__name__)

def run_sign_to_text(camera_id=0):
    logger.info(f"--- Running Sign-to-Text (Webcam {camera_id}) ---")
    logger.info("Press 'q' to quit.")
    
    extractor = HolisticExtractor()
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        return

    # Using a dummy model since we don't have a trained one
    model = SignLanguageLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model.eval()
    
    sequence = []
    classes = ["HELLO", "THANK YOU", "I LOVE YOU"]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Ignoring empty camera frame.")
            continue
            
        results = extractor.process_image(frame)
        keypoints = extractor.extract_keypoints(results, normalize=True)
        extractor.draw_landmarks(frame, results)
        
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:] # Keep last frames
        
        if len(sequence) == SEQUENCE_LENGTH:
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
    logger.info("Camera released, exiting Sign-to-Text.")

def run_text_to_sign(text_input=None):
    logger.info("--- Running Text-to-Sign Pipeline ---")
    nlp_pipeline = NLPGlossPipeline()
    pose_gen = PoseGenerator()
    
    if text_input:
        process_text_to_sign(text_input, nlp_pipeline, pose_gen)
    else:
        while True:
            text_input = input("\nEnter Turkish/Azerbaijani text (or 'q' to quit): ")
            if text_input.lower() == 'q':
                break
            process_text_to_sign(text_input, nlp_pipeline, pose_gen)

def process_text_to_sign(text, nlp_pipeline, pose_gen):
    gloss_array = nlp_pipeline.process(text)
    logger.info(f"Extracted Gloss Array: {gloss_array}")
    
    generated_pose = pose_gen.generate_pose(gloss_array)
    logger.info(f"Generated Pose Tensor Shape: {generated_pose.shape}")
    logger.info("This tensor would be sent to a 3D Avatar Engine (e.g. Unity).")

def main():
    parser = argparse.ArgumentParser(description="Chevir AI Accessibility Layer CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Pipeline mode to run")
    
    # Sign-to-Text mode
    s2t_parser = subparsers.add_parser("s2t", help="Run Sign-to-Text via Webcam")
    s2t_parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    
    # Text-to-Sign mode
    t2s_parser = subparsers.add_parser("t2s", help="Run Text-to-Sign")
    t2s_parser.add_argument("--text", type=str, help="Text to translate into sign language (optional)")
    
    args = parser.parse_args()
    
    if args.mode == "s2t":
        run_sign_to_text(camera_id=args.camera)
    elif args.mode == "t2s":
        run_text_to_sign(text_input=args.text)
    else:
        # Fallback to interactive mode if no arguments passed
        logger.info("No mode selected. Run with --help for CLI options.")
        logger.info("Starting interactive mode...")
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