# DeepPlate 
![close-up-of-camera-lens-548569043-586fc07d3df78c17b6d34728-small](https://github.com/user-attachments/assets/bcded706-ff55-43b8-b94a-8ca391a48661)

## Image Detection and License Plate Recognition

This project is an advanced image detection system designed to recognize and process license plates from live video feeds and images. It uses YOLO for object detection, PaddleOCR for text recognition, and SQLite for storing detected license plates.

## Features

- **Live Video Feed Processing**: Captures frames from a live video feed and detects license plates in real-time.
- **License Plate Validation**: Validates detected text against Australian license plate formats using regex patterns.
- **Database Integration**: Stores detected license plates and their states in an SQLite database.
- **Customizable Training**: Includes a script to train YOLO models on custom datasets.
- **Image-Based Detection**: Processes static images to detect and validate license plates.

## Project Structure

- `liveFeedRecognition.py`: Main script for live video feed processing and license plate recognition.
- `trainModel.py`: Script for training YOLO models on custom datasets.
- `main.py`: Script for processing static images and detecting license plates.
- `chat_template.jinja`: Template for OCR-related tasks.
- `yolo11n.pt`: Pre-trained YOLO model weights.
- `last.pt`: Custom model located in `train14/weights/` trained for license plates
- `regos.db`: SQLite database for storing detected license plates.

## Requirements

- Python 3.8+
- Required Python libraries:
  - `opencv-python`
  - `numpy`
  - `ultralytics`
  - `paddleocr`
  - `sqlite3`
- Tesseract OCR installed on your system.
- A webcam or video input device for live feed processing.

## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd ImageDetection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Attribution

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
