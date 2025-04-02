import os
import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from multiprocessing import Process, Queue
import logging
import sqlite3


# SQL database setup
def initialize_db():
    """
    Initialize the SQLite database and create the regos table if it doesn't exist.
    Returns:
        conn (sqlite3.Connection): The SQLite connection object.
        c (sqlite3.Cursor): The SQLite cursor object.
    """
    conn = sqlite3.connect("regos.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS regos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rego TEXT NOT NULL,
            state TEXT NOT NULL, -- Ensure the state column is created
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()
    return conn, c


def insert_rego(rego, state):
    """
    Insert a new rego into the database if it doesn't already exist.
    params:
        rego (str): The license plate number.
        state (str): The state code.

    """
    conn = sqlite3.connect("regos.db")  # Open a new connection
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM regos WHERE rego = ? AND state = ?", (rego, state))
    count = c.fetchone()[0]

    if count > 0:
        print(f"Rego {rego} ({state}) already exists in the database.")
        conn.close()  # Close connection if no insertion
        return

    c.execute("INSERT INTO regos (rego, state) VALUES (?, ?)", (rego, state))
    conn.commit()
    conn.close()  # Ensure connection is closed after insertion
    print(f"Rego {rego} ({state}) inserted into the database.")


license_plate_patterns = {
    # Current standard issue plates
    "ACT|JBT": r"^[A-Z]{3}\d{1,4}[A-Z]?$",  # Australian Capital Territory and Jervis Bay Territory
    "NSW": r"^(?:[A-Z]{2}\d{2}[A-Z]{2}|[A-Z]{3}\d{3}|[A-Z]{2}\d{3}|[A-Z]{1}\d{3}[A-Z]{1})$",  # New South Wales
    "NT": r"^[A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]{2}[A-Z]{1}$",  # Northern Territory
    "QLD": r"^\d{3}[A-Z]{2}\d{1}$",  # Queensland
    "SA": r"^[A-Z]{1}[0-9]{3}[A-Z]{1}[0-9]{2}[A-Z]{1}$",  # South Australia
    "TAS": r"^[A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]{2}[A-Z]{1}$",  # Tasmania
    "VIC": r"^[A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]{2}[A-Z]{1}$",  # Victoria
    "WA": r"^1[A-Z]{2}[A-Z]?\d{3}$",  # Western Australia
    # Other issue plates
    "ACT_PREMIUM": r"^[A-Z]{3}\d{1,4}[A-Z]?$",  # ACT Premium
    "NSW_PREMIUM": r"^[A-Z]{2}[A-Z]\d{1,4}[A-Z]?$",  # NSW Premium
    "NSW_HISTORIC": r"^\d{5}J$",  # NSW Historic Vehicle
    "QLD_SEQUENTIAL": r"^\d{3}·[A-Z]\d{1}[A-Z]$",  # QLD Sequential Series
    "QLD_TAXI": r"^T·\d{5}$",  # QLD Taxi Plates
    "QLD_LIMOUSINE": r"^L·\d{5}$",  # QLD Limousine Plates
    "QLD_SPECIAL_LIMOUSINE": r"^SL·\d{2}·[A-Z]{2}$",  # QLD Special Purpose Limousines
    "QLD_FARM_1": r"^1[A-Z]·\d{3}$",  # QLD Farm Plates (1A·NNN format)
    "QLD_FARM_2": r"^F·\d{5}$",  # QLD Farm Plates (F·NNNNN format)
    "QLD_GOVERNMENT": r"^QG·[A-Z]{2}\d{2}$",  # QLD Government Used Vehicles
    "NSW_CONDITIONAL": r"^\d{5}E$",  # NSW Conditional
    "NSW_RALLY": r"^\d{5}R$",  # NSW Rally Permit
    "NSW_CLASSIC": r"^\d{4}D$",  # NSW Classic Cycle
    "SA_HEAVY": r"^SB\d{2}[A-Z]{2}$",  # SA Heavy Vehicle
    "SA_PREMIUM": r"^[A-Z]{2}\d{3}[A-Z]$",  # SA Premium
    "VIC_PREMIUM": r"^[A-Z]{3}\d{3}$",  # VIC Premium
    "VIC_PRIMARY": r"^\d{5}P$",  # VIC Primary Producer
    "VIC_CLUB": r"^\d{4}H\d$",  # VIC Club Permit
    "QLD_FARM": r"^[A-Z]\d{5}$",  # QLD Farm Plates
    "WA_PLATINUM": r"^1F[A-Z]{2}\d{3}$",  # WA Platinum Slimline
    # Trailer plates
    "ACT_TRAILER": r"^T\d{4}[A-Z]$",  # ACT Trailer
    "NSW_TRAILER": r"^T[A-Z]\d{2}[A-Z]{2}$",  # NSW Trailer
    "NT_TRAILER": r"^T[A-Z]\d{4}$",  # NT Trailer
    "QLD_TRAILER": r"^\d{3}U[A-Z]{2}$",  # QLD Trailer
    "SA_TRAILER": r"^S\d{3}T[A-Z]{2}$",  # SA Trailer
    "TAS_TRAILER": r"^[A-Z]\d{2}[A-Z]{2}$",  # TAS Trailer
    "VIC_TRAILER": r"^\d{3}\d{2}C$",  # VIC Trailer
    "WA_TRAILER": r"^1U[A-Z]{2}\d{3}$",  # WA Trailer
    # Motorcycle plates
    "ACT_MOTORCYCLE": r"^C\d{4}$",  # ACT Motorcycle
    "NSW_MOTORCYCLE": r"^[A-Z]{3}\d{2}$",  # NSW Motorcycle
    "NT_MOTORCYCLE": r"^C\d{4}$",  # NT Motorcycle
    "QLD_MOTORCYCLE": r"^3[A-Z]{2}\d{2}$",  # QLD Motorcycle
    "SA_MOTORCYCLE": r"^S\d{2}[A-Z]{3}$",  # SA Motorcycle
    "TAS_MOTORCYCLE": r"^C\d{3}[A-Z]$",  # TAS Motorcycle
    "VIC_MOTORCYCLE": r"^3D[A-Z]\d{2}$",  # VIC Motorcycle
    "WA_MOTORCYCLE": r"^1L[A-Z]{2}\d{3}$",  # WA Motorcycle
}
# Define a pattern to replace unwanted characters (e.g., spaces and hyphens)
replace_pattern = r"[-\s\.·]+"

# Load YOLO and OCR
path = os.getcwd() + "/train14/weights/last.pt"
model = YOLO(path)  # Update to the correct path
model.fuse()
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en")

def initialize_video_capture():
    if os.name == "nt":  # Check if the OS is Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use AVFoundation for macOS

    if not cap.isOpened():
        print("Error: Could not open video feed.")
        exit()
    return cap


def detect_state_and_plate(plate_text):
    """
    Detects the state and license plate from the given text.
    Args:
        plate_text (str): The text to analyze.
    Returns:
        tuple: A tuple containing the detected state and license plate text.
    """
    for state, pattern in license_plate_patterns.items():
        if re.match(pattern, plate_text):
            return state, plate_text
    return None, None


def feed_worker(feed_queue):
    """
    Captures video frames and pushes them to the feed queue.
    Args:
        feed_queue (Queue): The queue to store captured frames.
    """
    cap = initialize_video_capture()
    while True:
        ret, frame = cap.read()
        # change the frame size to 640x480
        frame = cv2.resize(frame, (800, 500))
        if not ret:
            print("Error: Could not read frame.")
            break

        if feed_queue.full():
            feed_queue.get()  # Discard a frame if the queue is full (non-blocking)

        feed_queue.put(frame)  # Push the frame to the queue

    cap.release()

def preprocess_frame(roi):
    """
    Preprocesses the frame for license plate detection.
    Args:
        frame (numpy.ndarray): The video frame to preprocess.
    Returns:
        numpy.ndarray: The preprocessed frame.
    """
    #Modify the preprocess_roi function to handle small ROIs
    upsampled = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upsampled, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = deskew_image(binary)
    return deskewed

def deskew_image(img):
    coords = np.column_stack(np.nonzero(img > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Correct the rotation to ensure horizontal alignment
    if angle < -45:
        angle += 90  # Fix for incorrectly rotated images
    elif angle > 45:
        angle -= 90  # Another fix for vertical flipping

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def validate_combined_text(combined_text):
    """
    Validates the combined text to check if it matches any license plate pattern.
    """
    # remove unwanted characters
    combined_text = re.sub(replace_pattern, "", combined_text.strip())
    for state, pattern in license_plate_patterns.items():
        if re.match(pattern, combined_text):
            return combined_text
    return None


def extract_text_from_roi(roi):
    """
    Extracts text from a region of interest (ROI) using PaddleOCR.
    Args:
        roi (numpy.ndarray): The region of interest from which to extract text.
    Returns:
        tuple: A tuple containing the detected state and license plate text.
    """
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    result = ocr.ocr(roi, cls=False)

    if not result or not isinstance(result, list):
        return None, None

    for line in result:
        if not isinstance(line, list):
            continue
        detected_text = process_line(line)
        if detected_text:
            return detect_state_and_plate(detected_text)
        else:
            cv2.rectangle(roi, (0, 0), (roi.shape[1], roi.shape[0]), (0, 0, 255), 2)
            cv2.putText(
                roi,
                "Invalid Plate",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    return None, None


def process_line(line):
    """
    Processes a line of detected text and extracts valid license plate information.
    Args:
        line (list): A list of detected words and their confidence scores.
    Returns:
        str: The combined text of valid license plates.
        None: If no valid license plate is found.
    """
    combined_text = "".join(
        format_plate(text)
        for word_info in line
        if isinstance(word_info, list) and len(word_info) >= 2
        for text, confidence in [extract_text_and_confidence(word_info)]
        if confidence >= 0.65 and is_valid_plate(text)
    )
    return validate_combined_text(combined_text)


def extract_text_and_confidence(word_info):
    if isinstance(word_info[1], tuple) and len(word_info[1]) == 2:
        text = word_info[1][0]
        confidence = word_info[1][1]
    else:
        text = word_info[1]
        confidence = 0.0

    return text, confidence


def is_valid_plate(text):
    if len(text) < 3 or len(text) > 10:
        return False

    # Check if the text contains only alphanumeric characters
    if not re.match(r"^[A-Za-z0-9]+$", text):
        return False

    # Check if the text contains at least one letter and one digit
    if not re.search(r"[A-Za-z]", text) or not re.search(r"\d", text):
        return False

    return True


def format_plate(text):
    """
    Formats the detected text to a standard format for license plates.
    Args:
        Text (str): The detected text to format.
    Returns:
        str: The formatted text.
    """
    # Remove spaces and hyphens
    text = re.sub(r"[-\s]+", "", text)
    # Convert to uppercase
    text = text.upper()
    # Replace any unwanted characters
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def process_worker(feed_queue, processed_queue):
    """
    Processes frames from the feed queue and performs license plate detection and recognition.
    Args:
        feed_queue (Queue): The queue containing frames to process.
        processed_queue (Queue): The queue to store processed frames.
    """
    while True:
        conn, _ = initialize_db()  # Open a new connection per process
        if not feed_queue.empty():
            frame = feed_queue.get()
            results = model(frame, conf=0.1, device="mps", verbose=False)[0]
            frame = validate_and_annotate(frame, results)
            processed_queue.put(frame)
        conn.close()  # Ensure the connection is closed after processing each frame


def validate_and_annotate(frame, results):
    """
    Validates detected regions of interest (ROIs) and annotates the frame with detected license plates and states.

    Args:
        frame (numpy.ndarray): The video frame to process.
        results (list): Detection results from the YOLO model.

    Returns:
        numpy.ndarray: The annotated frame.
    """
    detected_texts = []
    rois = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # predict the roi based on the bounding box
            roi = frame[y1:y2, x1:x2]
            rois.append(roi)
            cv2.waitKey(1)
            state, detected_text = extract_text_from_roi(roi)

            if detected_text and state:
                detected_texts.append((detected_text, state, (x1, y1, x2, y2)))
                insert_rego(detected_text, state)
    

    return annotate_frame(frame, detected_texts)


def annotate_frame(frame, detected_texts): 
    """
    Annotates the frame with bounding boxes and text for detected license plates.

    Args:
        frame (numpy.ndarray): The video frame to annotate.
        detected_texts (list): List of detected license plates and their states.

    Returns:
        numpy.ndarray: The annotated frame.
    """
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


if __name__ == "__main__":
    conn, c = initialize_db()
    feed_queue = Queue(maxsize=10)
    processed_queue = Queue(maxsize=10)
    feed_process = Process(target=feed_worker, args=(feed_queue,))
    process_process = Process(target=process_worker, args=(feed_queue, processed_queue))
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
    conn.close()
