# ML Rego Detection

<p>
  <em>A fast, accurate, and real-time vehicle registration plate detection system.</em>
  <img src="https://github.com/user-attachments/assets/c96ae9f0-1b9f-448d-b0aa-fbcf8512795e" alt="rego icon" width="140" align="right" />
</p>

![CodeQL](https://github.com/Tristan296/DeepPlate/actions/workflows/codeql.yml/badge.svg)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Tristan296_DeepPlate&metric=alert_status)](https://sonarcloud.io/summary/overall?id=Tristan296_DeepPlate&branch=main)

---

## Table of Contents

- [Overview](#overview)
- [Installation &amp; Setup](#installation--setup)
- [Usage](#usage)
- [Contributing](.github/CONTRIBUTIONS.md)
- [Acknowledgements](#acknowledgements)
- [License](.github/LICENSE)
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

1. Ensure docker is installed on your computer
2. Run setup.py:
   ```bash
   python setup.py
   ```
   Note:
   This setup may take a few minutes to install all the appropriate dependencies.

## Usage

**MacOS**:

```bash
docker run --rm -it --env DISPLAY=host.docker.internal:0 --device /dev/video0 --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged deepplate-img
```

**Windows:**

```bash
docker run -e DISPLAY=host.docker.internal:0 -v . -it deepplate-img
```

**Linux**:

```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix deepplate-img
```

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
