# Image Detection

> A project that utilizes machine learning to detect and classify objects in images.
> Live demo [_here_](https://www.example.com). 

## Table of Contents

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information

- This project aims to simplify the process of detecting and classifying objects in images using advanced machine learning techniques.
- It solves the problem of manual image classification by automating the process.
- The purpose of this project is to provide an easy-to-use tool for developers and researchers working with image data.

## Technologies Used

- Python - version 3.9
- TensorFlow - version 2.10
- OpenCV - version 4.5

## Features

List the ready features here:

- Object detection in real-time
- Classification of detected objects
- Support for multiple image formats

## Screenshots

![Example screenshot](./img/screenshot.png)

<!-- Add actual screenshots of the project -->

## Setup

- Install the required dependencies listed in `requirements.txt`.
- Clone the repository and navigate to the project directory.
- Run the following command to install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Usage

- To detect objects in an image, run the following command:
  ```
  python detect.py --image path/to/image.jpg
  ```
- For real-time detection using a webcam:
  ```
  python detect.py --webcam
  ```

## Project Status

Project is: _in progress_. Further improvements and optimizations are being worked on.

## Room for Improvement

Room for improvement:

- Enhance the accuracy of object detection.
- Add support for additional machine learning models.

To do:

- Implement a user-friendly GUI.
- Add functionality for batch image processing.

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
