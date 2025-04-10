# ML Rego Detection
*A fast, accurate, and real-time vehicle registration plate detection system.*

![CodeQL](https://github.com/Tristan296/DeepPlate/actions/workflows/codeql.yml/badge.svg)  
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Tristan296_DeepPlate&metric=alert_status)](https://sonarcloud.io/summary/overall?id=Tristan296_DeepPlate&branch=main)

---

## Table of Contents
- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](.github/CONTRIBUTING.md)
- [Acknowledgements](#acknowledgements)
- [License](.github/LICENSE.md)
- [Contact](#contact)

---

## Overview

DeepPlate streamlines the process of detecting and classifying vehicle registration plates using advanced machine learning techniques. By leveraging YOLO for object detection and PaddleOCR for text extraction, DeepPlate validates plate formats in real-time, ensuring accuracy across various Australian states and license plate types.

**Why DeepPlate?**  
- **Speed & Accuracy:** Real-time video processing with GPU acceleration.
- **Advanced Preprocessing:** Optimized image enhancements for reliable OCR.
- **Multiprocessing:** Scalable deployment using Python’s multiprocessing and Queue.
- **Comprehensive Features:** From live-stream detection to video file processing, integrated storage to prevent duplicates.

---

## Installation & Setup

1. **Create a Virtual Environment**  
   Set up a virtual environment to isolate project dependencies from your global Python installation.

2. **Clone the Repository**  
   Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/yourusername/DeepPlate.git
   cd DeepPlate
   ```
3. **Install Dependencies**
    Install all required packages listed in reqs.txt (use pip3 if necessary):
    ```bash
    pip install -r reqs.txt
    ```
## Usage
### Run the main.py file and choose one of the following options:

[1] Live Stream Detection – Launches detection via your webcam.

[2] Video File Streaming – Analyzes and processes a saved video file.
    

## Technologies Used
- Python (v3.9)
- TensorFlow (v2.10)
- OpenCV (v4.5)
- YOLO for Object Detection
- PaddleOCR


## Screenshots

<img width="598" alt="Screenshot 2025-04-05 at 6 30 53 PM" src="https://github.com/user-attachments/assets/06c183b3-6ee8-4a6c-bb18-0b5a18a36620" />


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
