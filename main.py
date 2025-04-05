import os
import cv2
import numpy as np
import re
import logging
import sqlite3
import yaml
from paddleocr import PaddleOCR
from ultralytics import YOLO
from multiprocessing import Process, Queue
from typing import Tuple, Optional
from shapely.geometry import box as shapely_box

class LicensePlateRecognition:
    def __init__(self, license_patterns_file: str = "license_plate_patterns.yaml"):
        self.license_plate_patterns, self.replace_pattern = self.load_license_plate_patterns(license_patterns_file)
        self.model = YOLO(os.getcwd() + "/train14/weights/last.pt")
        self.model.fuse()
        self.ocr = None

    def get_db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect("regos.db")

    def get_ocr_result(self, roi: np.ndarray) -> list:
        if self.ocr is None:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Initialize OCR inside the process
        return self.ocr.ocr(roi, cls=False)
    
    def load_license_plate_patterns(self, file_path: str) -> Tuple[dict, str]:
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                return data.get("license_plate_patterns", {}), r"[^A-Za-z0-9]"
        except FileNotFoundError:
            print("Error: Pattern file not found.")
            exit()

    def initialize_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect("regos.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS regos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rego TEXT NOT NULL,
                state TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return conn

    def insert_rego(self, rego: str, state: str) -> None:
        conn = self.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM regos WHERE rego = ? AND state = ?", (rego, state))
        count = c.fetchone()[0]

        if count > 0:
            print(f"Rego {rego} ({state}) already exists in the database.")
            return
        
        c.execute("INSERT INTO regos (rego, state) VALUES (?, ?)", (rego, state))
        conn.commit()
        print(f"Rego {rego} ({state}) inserted into the database.")
        conn.close()

    def detect_state_and_plate(self, plate_text: str) -> Tuple[Optional[str], Optional[str]]:
        for state, pattern in self.license_plate_patterns.items():
            if re.match(pattern, plate_text):
                return state, plate_text
        return None, None

    def validate_combined_text(self, combined_text: str) -> Optional[str]:
        """
        Validates the combined text to check if it matches any license plate pattern.
        """
        combined_text = re.sub(self.replace_pattern, "", combined_text.strip())
        for state, pattern in self.license_plate_patterns.items():
            if re.match(pattern, combined_text):
                return combined_text
        return None


class VideoProcessor:
    def __init__(self, license_plate_recognition: LicensePlateRecognition):
        self.lpr = license_plate_recognition

    def initialize_video_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open video feed.")
            exit()
        return cap

    def feed_worker(self, feed_queue: Queue) -> None:
        cap = self.initialize_video_capture()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            if feed_queue.full():
                feed_queue.get()
            feed_queue.put(cv2.resize(frame, (800, 500)))
        cap.release()

    def process_worker(self, feed_queue: Queue, processed_queue: Queue) -> None:
        while True:
            if self.lpr.ocr is None:  # Ensure OCR is initialized in the subprocess
                self.lpr.ocr = PaddleOCR(use_angle_cls=True, lang="en")
            
            if not feed_queue.empty():
                frame = feed_queue.get()
                batch_frames = [frame]  # Add more frames to the batch if available
                self.lpr.model.fp16 = True  # Enable FP16
                results = self.lpr.model(batch_frames, conf=0.1, device="mps", verbose=False)
                results = results[0]  # Process the first frame's results
                frame = self.validate_and_annotate(frame, results)
                processed_queue.put(frame)

    def preprocess_frames(self, frames: list) -> list:
        """
        Preprocess the frames using SuperResolution before passing them to the YOLO model.
        """
        preprocessed_frames = []
        for frame in frames:
            # Upscale the frame using SuperResolution 
            frame = self.upscale_frame(frame)
            
            # Convert to grayscale for further processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply Gaussian blur to reduce noise
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized_frame = clahe.apply(gray_frame)
            
            # Convert back to BGR for compatibility with the YOLO model
            preprocessed_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
            
            # Append the preprocessed frame to the list
            preprocessed_frames.append(preprocessed_frame)
        
        return preprocessed_frames
    
    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscales the frame using a SuperResolution model.
        """
        height, width = frame.shape[:2]
        scale_factor = max(1, 300 // min(height, width))  # Calculate scale factor to ensure minimum size of 300
        max_dimension = 1000  # Define a maximum dimension for upscaling
        max_scale_factor = max_dimension // max(height, width)  # Ensure it doesn't exceed max dimension
        scale_factor = min(scale_factor, max_scale_factor)  # Use the smaller scale factor
        if scale_factor > 1:
            frame = cv2.resize(frame, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)

        return frame
    
    def validate_and_annotate(self, frame: np.ndarray, results: list) -> np.ndarray:
        detected_texts = []
        processed_boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = shapely_box(x1, y1, x2, y2)

                # Check if this box overlaps significantly with any already processed box
                if any(self.calculate_iou(current_box, b) > 0.5 for b in processed_boxes):
                    continue  # Skip processing this box

                roi = frame[y1:y2, x1:x2]
                # Preprocess the ROI before OCR
                preprocessed_roi = self.preprocess_frames([roi])[0]
                # preview the ROI for debugging
               
                fixed_size = (300, 150)  # Set a fixed size for the ROI display
                resized_roi = cv2.resize(preprocessed_roi, fixed_size, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("ROI", resized_roi)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
                state, detected_text = self.extract_text_from_roi(preprocessed_roi)

                if detected_text and state:
                    detected_texts.append((detected_text, state, (x1, y1, x2, y2)))
                    self.lpr.insert_rego(detected_text, state)

                # Add the current box to the list of processed boxes
                processed_boxes.append(current_box)

        return self.annotate_frame(frame, detected_texts)
    

    def calculate_iou(self, box1, box2) -> float:
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

    def extract_text_from_roi(self, roi: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        result = self.lpr.get_ocr_result(roi)

        if not result or not isinstance(result, list):
            return None, None

        for line in result:
            if not isinstance(line, list):
                continue
            detected_text = self.process_line(line)
            if detected_text:
                return self.lpr.detect_state_and_plate(detected_text)
            else:
                self.draw_invalid_plate(roi)
        return None, None

    def draw_invalid_plate(self, roi: np.ndarray):
        cv2.rectangle(roi, (0, 0), (roi.shape[1], roi.shape[0]), (0, 0, 255), 2)
        cv2.putText(roi, "Invalid Plate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def annotate_frame(self, frame: np.ndarray, detected_texts: list) -> np.ndarray:
        for detected_text, state, (x1, y1, x2, y2) in detected_texts:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{detected_text} ({state})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        return frame

    def process_line(self, line: list) -> Optional[str]:
        """
        Processes a line of detected text and extracts valid license plate information.

        Args:
            line (list): A list of detected words and their confidence scores.

        Returns:
            str: The combined text of valid license plates.
            None: If no valid license plate is found.
        """
        combined_text = self.join_valid_plates(line)
        return self.lpr.validate_combined_text(combined_text)

    def join_valid_plates(self, line: list) -> str:
        valid_plates = [
            self.format_plate(text)
            for word_info in line
            if isinstance(word_info, list) and len(word_info) >= 2
            for text, confidence in [self.extract_text_and_confidence(word_info)]
            if confidence >= 0.65 and self.is_valid_plate(text)
        ]
        return "".join(valid_plates)

    def extract_text_and_confidence(self, word_info: list) -> Tuple[str, float]:
        if isinstance(word_info[1], tuple) and len(word_info[1]) == 2:
            text = word_info[1][0]
            confidence = word_info[1][1]
        else:
            text = word_info[1]
            confidence = 0.0
        return text, confidence

    def is_valid_plate(self, text: str) -> bool:
        if len(text) < 3 or len(text) > 10:
            return False
        if not re.match(r"^[A-Za-z0-9]+$", text):
            return False
        if not re.search(r"[A-Za-z]", text) or not re.search(r"\d", text):
            return False
        return True

    def format_plate(self, text: str) -> str:
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


if __name__ == "__main__":
    lpr = LicensePlateRecognition()
    vp = VideoProcessor(lpr)
    feed_queue = Queue(maxsize=10)
    processed_queue = Queue(maxsize=10)
    feed_process = Process(target=vp.feed_worker, args=(feed_queue,))
    process_process = Process(target=vp.process_worker, args=(feed_queue, processed_queue))
    process_process.start()
    feed_process.start()

    while True:
        if not processed_queue.empty():
            frame = processed_queue.get()
            cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    feed_process.join()
    process_process.join()
    cv2.destroyAllWindows()
    lpr.db_connection.close()