import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
import winsound

# ===============================
# CONFIGURATION
# ===============================

CONF_THRESHOLD = 0.45
STABILITY_FRAMES = 10
MIN_DETECTIONS_IN_WINDOW = 4
DASHBOARD_HEIGHT = 140
TITLE_BAR_HEIGHT = 40

ALERT_LOUD = True
SHOW_DASHBOARD = True

# ===============================
# INITIAL SETUP
# ===============================

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible.")
    exit()

logs_dir = "../logs"
os.makedirs(logs_dir, exist_ok=True)

start_time = time.time()
alert_count = 0

# ===============================
# USER INPUT
# ===============================

tracked_input = input("Enter tracked objects (comma separated): ")
restricted_input = input("Enter restricted objects (comma separated): ")

tracked_objects = [x.strip().lower() for x in tracked_input.split(",") if x.strip()]
restricted_objects = [x.strip().lower() for x in restricted_input.split(",") if x.strip()]

# ===============================
# STATE VARIABLES
# ===============================

detection_history = {}
object_presence = {}
event_log = "System Initialized"
alert_active = False

# ===============================
# MAIN LOOP
# ===============================

print("System Running. Press Q to Quit. A=Toggle Alert Mode, D=Toggle Dashboard")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    current_frame_objects = set()
    alert_triggered_this_frame = False

    results = model(frame, verbose=False)
    detections = results[0]

    # ===============================
    # DETECTION + STABILITY
    # ===============================

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        confidence = float(box.conf[0])

        if class_name in tracked_objects and confidence > CONF_THRESHOLD:

            current_frame_objects.add(class_name)
            detection_history[class_name] = detection_history.get(class_name, 0) + 1

            if detection_history[class_name] >= STABILITY_FRAMES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0)

                # Restricted object logic
                if class_name in restricted_objects:
                    color = (0, 0, 255)
                    alert_triggered_this_frame = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
    print("Detected:", class_name, "Confidence:", confidence)


    # Reset detection history
    for obj in list(detection_history.keys()):
        if obj not in current_frame_objects:
            detection_history[obj] = 0

    # ===============================
    # EVENT MANAGEMENT
    # ===============================

    for obj in tracked_objects:
        was_present = object_presence.get(obj, False)
        is_present = detection_history.get(obj, 0) >= STABILITY_FRAMES

        if is_present and not was_present:
            event_log = f"{obj} appeared"
            if obj in restricted_objects:
                alert_count += 1
                timestamp = int(time.time())
                cv2.imwrite(f"{logs_dir}/alert_{obj}_{timestamp}.jpg", frame)

        if not is_present and was_present:
            event_log = f"{obj} removed"

        object_presence[obj] = is_present

    # ===============================
    # ALERT SYSTEM
    # ===============================

    if alert_triggered_this_frame:
        alert_active = True

        if ALERT_LOUD:
            winsound.Beep(1200, 200)
        else:
            winsound.Beep(800, 100)

    else:
        alert_active = False

    # Red border if alert
    if alert_active:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 10)
        cv2.putText(
            frame,
            "!!! SECURITY ALERT !!!",
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # ===============================
    # TITLE BAR
    # ===============================

    cv2.rectangle(frame, (0, 0), (width, TITLE_BAR_HEIGHT), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "SMART VISION COMMAND CENTER",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ===============================
    # DASHBOARD
    # ===============================

    if SHOW_DASHBOARD:
        cv2.rectangle(
            frame,
            (0, height - DASHBOARD_HEIGHT),
            (width, height),
            (15, 15, 15),
            -1
        )

        runtime = int(time.time() - start_time)

        y_base = height - DASHBOARD_HEIGHT + 25

        cv2.putText(frame, f"Runtime: {runtime}s", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Alert Mode: {'LOUD' if ALERT_LOUD else 'SUBTLE'}", (200, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Alerts Triggered: {alert_count}", (450, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        y_base += 30

        status_text = " | ".join([
            f"{obj}: {'PRESENT' if object_presence.get(obj, False) else 'NOT PRESENT'}"
            for obj in tracked_objects
        ])

        cv2.putText(frame, status_text, (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        y_base += 30

        cv2.putText(frame, f"Last Event: {event_log}", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,255), 2)

    # ===============================
    # DISPLAY
    # ===============================

    cv2.imshow("Smart Vision Command Center", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        ALERT_LOUD = not ALERT_LOUD
    elif key == ord('d'):
        SHOW_DASHBOARD = not SHOW_DASHBOARD

cap.release()
cv2.destroyAllWindows()
