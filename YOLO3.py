import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import os

# Hide Tkinter root window
root = tk.Tk()
root.withdraw()

# Ask user for input choice
choice = simpledialog.askinteger(
    "Input",
    "Choose an option:\n1. Image\n2. Video\n3. Webcam",
    minvalue=1,
    maxvalue=3
)

# File selection based on choice
if choice == 1:
    file_types = [("Image Files", ".jpg;.jpeg;*.png")]
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=file_types)
elif choice == 2:
    file_types = [("Video Files", ".mp4;.avi;.mov;.mkv")]
    file_path = filedialog.askopenfilename(title="Select a Video", filetypes=file_types)
else:
    file_path = None  # For webcam

if choice in [1, 2] and not file_path:
    print("No file selected. Exiting...")
    exit()

# Load YOLO model
labelsPath = r"C:\Users\RAJ KUMARI\OneDrive\Desktop\yolo-coco\coco.names"
weightsPath = r"C:\Users\RAJ KUMARI\OneDrive\Desktop\yolo-coco\yolov3.weights"
configPath = r"C:\Users\RAJ KUMARI\OneDrive\Desktop\yolo-coco\yolov3.cfg"

# Ensure files exist
if not os.path.exists(labelsPath) or not os.path.exists(weightsPath) or not os.path.exists(configPath):
    print("YOLO model files not found. Please ensure that 'coco.names', 'yolov3.cfg', and 'yolov3.weights' are present in the specified directory.")
    exit()

LABELS = open(labelsPath).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in np.ravel(net.getUnconnectedOutLayers())]

# Initialize colors for labels
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

def detect_objects(frame):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    )

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(
                frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return frame

# Process Image
if choice == 1:
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image!")
        exit()

    processed_image = detect_objects(image)
    cv2.imshow("Object Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process Video / Webcam
else:
    if choice == 2:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()
    else:
        # Attempt to find an available webcam
        cap = None
        for index in range(5):
            temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                print(f"Camera found at index {index}")
                cap = temp_cap
                break
            temp_cap.release()
        if cap is None:
            print("No available camera found.")
            exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if choice == 2:
        output_path = os.path.splitext(file_path)[0] + "_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, 30, (frame_width, frame_height)
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Processing complete.")
            break

        processed_frame = detect_objects(frame)

        if choice == 2:
            out.write(processed_frame)

        cv2.imshow("YOLO Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if choice == 2:
        out.release()
    cv2.destroyAllWindows()