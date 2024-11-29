import cv2
import pandas as pd
from datetime import datetime

# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize logging
log_columns = ["Timestamp", "Detected Object"]
detection_log = pd.DataFrame(columns=log_columns)

# Open the webcam
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    height, width, _ = img.shape

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Get object bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    if len(indexes) > 0:  # Check if there are any valid detections
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            confidence = confidences[i]

            # Draw the bounding box and label on the frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Log detections
    for obj in detected_objects:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame({"Timestamp": [current_time], "Detected Object": [obj]})
        detection_log = pd.concat([detection_log, new_entry], ignore_index=True)

    # Show frame
    cv2.imshow("YOLO Object Detection", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Save detection log to an Excel file
excel_file = "detection_log.xlsx"
detection_log.to_excel(excel_file, index=False)
print(f"Detection log saved to {excel_file}")

# Release resources
cam.release()
cv2.destroyAllWindows()
