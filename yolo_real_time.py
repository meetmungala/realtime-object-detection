import cv2
import numpy as np
import time

# Load YOLO
print("[INFO] Loading YOLOv4-Tiny from disk (Major accuracy upgrade)...")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (600, 450))
    height, width, channels = frame.shape

    # Detecting objects (Increased resolution to 608x608 for maximum detection accuracy)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Showing informations on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.45:  # Increased threshold to only show confident and accurate detections
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf = str(round(confidences[i]*100, 2)) + "%"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + conf, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO Real Time Object Detection (80 Classes)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
