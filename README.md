# Chevir: AI-Powered Sign Language Accessibility Layer

Chevir is a bidirectional AI-powered sign language translation and generation layer, initially designed as a prototype for the NSosyal social media platform. 

It handles two main pipelines:
1. **Module A (Sign-to-Text)**: Extracts 3D landmarks via MediaPipe Holistic and translates sequences of keypoints into Turkish/Azerbaijani text using a lightweight PyTorch network (LSTM).
2. **Module B (Text-to-Sign)**: Parses text using an NLP pipeline to extract a normalized "Gloss", which then maps to standard keypoint arrays to animate a 3D Avatar.

## Folder Structure
- `data/`: Contains raw videos and extracted numpy keypoints (ignored in Git).
- `src/`: 
  - `module_a_s2t/`: Extraction and PyTorch Model.
  - `module_b_t2s/`: NLP Gloss pipeline and Pose Generator.
  - `pipeline/`: Training and evaluation scripts.
  - `utils/`: Video streaming and visualization helpers.
- `inference.py`: CLI tool for testing the full pipeline.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the inference script: `python inference.py`
