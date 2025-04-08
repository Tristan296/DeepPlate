# DeepPlate - Smart ML Rego Detection

![CodeQL](https://github.com/Tristan296/DeepPlate/actions/workflows/codeql.yml/badge.svg)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Tristan296_DeepPlate&metric=alert_status)](https://sonarcloud.io/summary/overall?id=Tristan296_DeepPlate&branch=main)


## Table of Contents

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Model Training Summary](#model-training-summary)
* [Room for Improvement](#room-for-improvement)
* [Contributing](CONTRIBUTIONS.md)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information

- This project aims to simplify the process of detecting and classifying text in vehicle regos using advanced machine learning techniques.

## Technologies Used

- Python - version 3.9
- TensorFlow - version 2.10
- OpenCV - version 4.5

## Features
### License Plate Detection & OCR
- Detects license plates using YOLO object detection.
- Extracts text from plates using PaddleOCR.

### State & License Plate Validation
- Validates license plate text against regex patterns for Australian states and territories.
- Supports standard, premium, historic, and special-purpose license plate formats.

### Database Integration
- Stores detected license plates and their associated states in a SQLite database (`regos.db`).
- Prevents duplicate license plates from being inserted.

### Real-Time Video Processing
- Captures and processes video frames in real-time using OpenCV.
- Annotates frames with bounding boxes and recognized text.

### Multiprocessing
- Utilizes Python’s `multiprocessing` module for parallel video capture and frame processing.
- Uses a Queue for inter-process communication.

### Image Preprocessing for OCR
- Improves OCR accuracy through resizing, grayscale conversion, histogram equalization, Gaussian blur, and binarization.
- Corrects rotated license plates using deskewing techniques.

### Frame Annotation
- Draws bounding boxes and overlays recognized license plate text on video frames.

### Video File Processing
- Processes video files for license plate detection and recognition.
- Annotates detected plates in video frames and saves the results to a new video file.

### GPU Acceleration
- Leverages GPU capabilities (CUDA, MPS, or ROCm) for faster inference and processing.

### Enhanced Model Training
- Includes training configurations and results for YOLO-based object detection models.
- Provides visualizations such as confusion matrices, precision-recall curves, and training batch predictions.

### Validation and Testing
- Validates model performance on unseen data with detailed metrics and visual results.
- Tests the model on real-world video streams and images for generalization.

<img width="598" alt="Screenshot 2025-04-05 at 6 30 53 PM" src="https://github.com/user-attachments/assets/06c183b3-6ee8-4a6c-bb18-0b5a18a36620" />


## Setup

- Install the required dependencies listed in `requirements.txt`.
- Clone the repository and navigate to the project directory.
- Run the following command to install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Usage

- To detect objects using webcam, run the following command:
  ```
  python src/run_live_stream.py
  ```

- To detect objects from a video, run this command instead:
```
  python src/run_video_file_stream.py
```
 
## Project Status

Project is: _in progress_. Further improvements and optimizations are being worked on.

## Acknowledgements

### YOLO by Ultralytics

```bibtex
@misc{yolo2023,
  author       = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title        = {YOLO by Ultralytics},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ultralytics/yolov5}}
}
```

### PaddleOCR

```bibtex
@article{paddleocr2021,
  author       = {PaddleOCR Contributors},
  title        = {PaddleOCR: An Open-Source Optical Character Recognition Tool Based on PaddlePaddle},
  year         = {2021},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}}
}
```

## Contact

Created by [@tristan](https://github.com/tristan296) - feel free to contact me!
