import os
import cv2
import numpy as np
import re
import logging
import sqlite3
import torch
import yaml
from paddleocr import PaddleOCR
from ultralytics import YOLO
from typing import Tuple, Optional
from shapely.geometry import box as shapely_box
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from concurrent.futures import ThreadPoolExecutor
import time

with open((os.getcwd() + "/configs/license_plate_patterns.yaml"), "r") as file:
    patterns = yaml.safe_load(file)

# get the license plate patterns from the yaml file
country_code = "N/A"
plate_type = "N/A"
region = "N/A"

try:
    pattern = (
        patterns.get("license_plate_patterns", {})
        .get(country_code, {})
        .get(plate_type, {})
        .get(region)
    )
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("Pattern must be a non-empty string.")
except (AttributeError, ValueError) as e:
    pattern = r"^[A-Z0-9]+$"  # Default fallback pattern
    print(
        f"Pattern not found or invalid for the specified country, type, and region. Using default pattern: {e}"
    )

replace_pattern = r"[^A-Za-z0-9]"


def predict(chosen_model, img, classes=[], conf=0.4):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocesses a single frame for better OCR results:
        - Upscales the frame using SuperResolution.
        - Converts to grayscale.
        - Applies Gaussian blur to reduce noise.
        - Applies adaptive histogram equalization for better contrast.

    Args:
        frame: A single frame to preprocess.

    Returns:
        Preprocessed frame.
    """

    frame = upscale_frame(frame)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    equalized_frame = clahe.apply(blurred_frame)
    return equalized_frame

def upscale_frame(frame):
    """
    Upscales the frame using a SuperResolution model and measures time performance.
    """

    # Upscale using ESPCN_x2 model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = os.path.join(
        os.getcwd(), "weights", "FSRCNN_x4.pb"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Super-resolution model not found at {model_path}. Please check the path.")
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    upscaled_frame = sr.upsample(frame)

    return upscaled_frame


def preprocess_frames_parallel(frames: list) -> list:
    """
    Preprocesses frames in parallel for better CPU utilization.

    Args:
        frames: List of frames to preprocess.

    Returns:
        List of preprocessed frames.
    """
    with ThreadPoolExecutor() as executor:
        preprocessed_frames = list(executor.map(preprocess_frame, frames))
    return preprocessed_frames


def predict_and_detect(
    chosen_model, img, classes=[], conf=0.4, rectangle_thickness=2, text_thickness=1
):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                text_thickness,
            )
    return img, results


def calculate_iou(box1, box2) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box as a Shapely box.
        box2: Second bounding box as a Shapely box.

    Returns:
        float: IoU value between 0 and 1.
    """
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union > 0 else 0


# Adding ROI extraction and annotation logic
def validate_and_annotate(frame: np.ndarray, results: list) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame  # Return the original frame if it's invalid
    detected_texts = []
    processed_boxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = shapely_box(x1, y1, x2, y2)

            # Check if this box overlaps significantly with any already processed box
            if any(calculate_iou(current_box, b) > 0.5 for b in processed_boxes):
                continue  # Skip processing this box

            roi = frame[y1:y2, x1:x2]

            # Preprocess the ROI before OCR
            preprocessed_roi = preprocess_frame(roi)

            fixed_size = (300, 150)  # Set a fixed size for the ROI display
            resized_roi = cv2.resize(
                preprocessed_roi, fixed_size, interpolation=cv2.INTER_LINEAR
            )
            cv2.imshow("ROI", resized_roi)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            country, state, detected_text = extract_text_from_roi(preprocessed_roi)

            if detected_text and state:
                detected_texts.append((detected_text, state, country, (x1, y1, x2, y2)))

            # Add the current box to the list of processed boxes
            processed_boxes.append(current_box)

    return annotate_frame(frame, detected_texts)


def format_plate(text: str) -> str:
    """
    Formats the detected text to a standard format for license plates.

    Args:
        text (str): The detected text to format.

    Returns:
        str: The formatted text.
    """
    text = re.sub(r"[-\s]+", "", text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


ocr = None  # Initialize the OCR variable globally


def get_ocr_result(roi: np.ndarray) -> list:
    global ocr  # Declare ocr as global to access and modify it
    if ocr is None:
        ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # Initialize OCR if not already done
    return ocr.ocr(roi, cls=False)


def extract_text_and_confidence(word_info: list) -> Tuple[str, float]:
    if isinstance(word_info[1], tuple) and len(word_info[1]) == 2:
        text = word_info[1][0]
        confidence = word_info[1][1]
    else:
        text = word_info[1]
        confidence = 0.0
    return text, confidence


def validate_combined_text(combined_text: str) -> Optional[str]:
    """
    Validates the combined text to check if it matches any license plate pattern.
    """
    if not pattern or not isinstance(pattern, str):
        raise ValueError("Pattern must be a valid string before compiling.")
    regex_pattern = re.compile(pattern)
    if regex_pattern.match(combined_text):
        return combined_text
    else:
        return None


def is_valid_plate(text: str) -> bool:
    if len(text) < 3 or len(text) > 10:
        return False
    if not re.match(r"^[A-Za-z0-9]+$", text):
        return False
    if not re.search(r"[A-Za-z]", text) or not re.search(r"\d", text):
        return False
    return True


def join_valid_plates(line: list) -> str:
    valid_plates = [
        format_plate(text)
        for word_info in line
        if isinstance(word_info, list) and len(word_info) >= 2
        for text, confidence in [extract_text_and_confidence(word_info)]
        if confidence >= 0.65 and is_valid_plate(text)
    ]
    return "".join(valid_plates)


def process_line(line: list) -> Optional[str]:
    """
    Processes a line of detected text and extracts valid license plate information.

    Args:
        line (list): A list of detected words and their confidence scores.

    Returns:
        str: The combined text of valid license plates.
        None: If no valid license plate is found.
    """
    combined_text = join_valid_plates(line)
    return validate_combined_text(combined_text)


def get_rego_metadata(
    plate_text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts metadata (country code, plate type, region) from the plate text using the regex pattern.
    """
    regex = re.compile(pattern)
    match = regex.match(plate_text)
    for key, value in patterns.get("license_plate_patterns", {}).items():
        if isinstance(value, dict):
            for plate_type, region_data in value.items():
                if isinstance(region_data, dict):
                    for region, _ in region_data.items():
                        # Process country_code, plate_type, and region here
                        if match:
                            # return the plate text, country code, and region
                            return plate_text, key, region

    return None, None, None


def extract_text_from_roi(roi: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    result = get_ocr_result(roi)

    if not result or not isinstance(result, list):
        return None, None, None

    for line in result:
        if not isinstance(line, list):
            continue
        detected_text = process_line(line)
        if detected_text:
            return get_rego_metadata(detected_text)
        else:
            draw_invalid_plate(roi)
    return None, None, None


def draw_invalid_plate(roi: np.ndarray):
    cv2.rectangle(roi, (0, 0), (roi.shape[1], roi.shape[0]), (0, 0, 255), 2)
    cv2.putText(
        roi, "Invalid Plate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )


def annotate_frame(frame: np.ndarray, detected_texts: list) -> np.ndarray:
    for detected_text, state, country, (x1, y1, x2, y2) in detected_texts:
        if detected_text:  # Ensure detected_text is not None or empty
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{detected_text} ({state}) [{country}]",  # Annotate with detected text, state, and country
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    return frame


model = YOLO(os.getcwd() + "/weights/license_plate_detector.pt")
model.fuse()

# Check if GPU is available and set the model to use it
if torch.backends.mps.is_available():  # Check for Metal Performance Shaders (macOS/AMD)
    model.to("mps")
    print("Using MPS (Metal Performance Shaders) for inference.")
elif torch.cuda.is_available():  # Check for NVIDIA CUDA
    model.to("cuda")
    print("Using CUDA for inference.")
elif torch.has_mps:  # Check for AMD ROCm support
    model.to("hip")
    print("Using ROCm (AMD GPU) for inference.")
else:
    print("No compatible GPU found, using CPU for inference.")


video_path = os.path.join(os.getcwd(), "videos", "1280_720_60fps.mp4")

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}. Please check the path.")
    exit(1)

cap = cv2.VideoCapture(video_path)
logging.disable(logging.CRITICAL)


def run_video_stream():
    """
    Main function to run the video stream processing.
    """
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read frames from the video stream
    while True:
        success, img = cap.read()
        if not success:
            break

        # Perform prediction and detection
        result_img, results = predict_and_detect(model, img)

        # Validate and annotate the frame
        annotated_img = validate_and_annotate(result_img, results)

        # Display the resized annotated frame if valid
        if annotated_img is not None and annotated_img.size > 0:
            cv2.imshow("Image", annotated_img)
        else:
            print("Warning: Annotated image is invalid and cannot be displayed.")

        # Stop process when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopping the process...")
            cap.release()
            cv2.destroyAllWindows()
            break
