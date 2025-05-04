# Object Tracking with Natural Language Queries

This repository extends the capabilities of YOLOv9 by integrating a vision-language model (e.g., CLIP) to enable object tracking based on natural language queries. The project is divided into four main phases, combining state-of-the-art object detection, natural language processing, and tracking techniques.

## Features

- **Natural Language Query Processing**: Uses CLIP's ViT-B/32 text encoder to process user-provided natural language queries.
- **Object Detection**: Leverages YOLOv9 for generating candidate bounding boxes.
- **Similarity Matching**: Employs cosine similarity, FAISS, and L2 distance to select the candidate most relevant to the query.
- **Object Tracking**: Utilizes DeepSort for robust multi-object tracking.

## Workflow

1. **Natural Language Query Processing**: The input query is encoded using CLIP's text encoder to generate a feature vector.
2. **Candidate Generation**: YOLOv9 detects objects in the video frames, producing bounding boxes for potential candidates.
3. **Similarity Matching**: The detected candidates are compared to the query using similarity metrics (cosine similarity, FAISS, or L2 distance) to identify the best match.
4. **Object Tracking**: The selected candidate is tracked across frames using DeepSort.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Object-tracking-Natural-Language.git
   cd Object-tracking-Natural-Language


2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Download the YOLOv9 weights:
    Place the yolov9s.pt file in the root directory.

4. Run the web inference
    ```bash 
    python main_gradio.py

## Acknowledgments
- **YOLOv9** for object detection.
- **CLIP** for vision-language processing.
- **DeepSort** for object tracking.

