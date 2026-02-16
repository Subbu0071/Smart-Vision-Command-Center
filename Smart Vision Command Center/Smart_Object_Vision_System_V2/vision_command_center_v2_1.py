"""
==============================================================
SMART VISION COMMAND CENTER — V2.2 (Stable Release)

Lead Developer: Subroto Sarkar
AI Architecture & Technical Guidance: OpenAI Assistant

© 2026 Subroto Sarkar. All Rights Reserved.
==============================================================
"""

import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
import winsound

# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.45
STABILITY_WINDOW = 12
MIN_DETECTIONS = 4
DASHBOARD_HEIGHT = 200
TITLE_BAR_HEIGHT = 60

DEFAULT_TRACKED = ["person", "bottle", "cell phone"]
DEFAULT_RESTRICTED = ["cell phone"]

WINDOW_NAME = "Smart Vision Command Center"

# ============================================================
# INITIALIZATION
# ============================================================

model = YOLO("yolov8s.pt")

tracked_objects = DEFAULT_TRACKED.copy()
restricted_objects = DEFAULT_RESTRICTED.copy()

alert_mode_loud = True
show_dashboard = True

detection_memory = {}
object_presence = {}

alert_count = 0
event_log = "System Initialized"
start_time = time.time()

logs_dir = "../logs/screenshots"
os.makedirs(logs_dir, exist_ok=True)

# ============================================================
# STARTUP SCREEN
# ============================================================

def startup_screen():
    screen = np.zeros((600, 1000, 3), dtype=np.uint8)

    while True:
        screen[:] = (8, 10, 18)

        cv2.putText(screen,
                    "SMART VISION COMMAND CENTER",
                    (120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 180),
                    3)

        cv2.putText(screen,
                    "Real-Time AI Security & Analytics Platform",
                    (250, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),
                    1)

        cv2.line(screen, (150, 210), (850, 210), (40, 60, 80), 2)

        cv2.putText(screen, "Tracked Objects:",
                    (200, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2)

        cv2.putText(screen, ", ".join(tracked_objects),
                    (220, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 120),
                    2)

        cv2.putText(screen, "Restricted Objects:",
                    (200, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2)

        cv2.putText(screen, ", ".join(restricted_objects),
                    (220, 410),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 120, 255),
                    2)

        cv2.putText(screen,
                    "Press ENTER to Launch Monitoring System",
                    (260, 500),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.imshow(WINDOW_NAME, screen)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break

    cv2.destroyAllWindows()

# ============================================================
# MAIN EXECUTION
# ============================================================

startup_screen()

cap = cv2.VideoCapture(0)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME,
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

previous_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    for obj in tracked_objects:
        detection_memory.setdefault(obj, [])

    results = model(frame, verbose=False)
    detections = results[0]

    current_detected = {}

    # ========================================================
    # DETECTION + STABILITY
    # ========================================================

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        confidence = float(box.conf[0])

        if class_name in tracked_objects and confidence > CONFIDENCE_THRESHOLD:
            detection_memory[class_name].append(1)
            current_detected[class_name] = (box, confidence)
        else:
            if class_name in detection_memory:
                detection_memory[class_name].append(0)

    for obj in detection_memory:
        detection_memory[obj] = detection_memory[obj][-STABILITY_WINDOW:]

    alert_active = False

    for obj in tracked_objects:
        if sum(detection_memory[obj]) >= MIN_DETECTIONS:
            object_presence[obj] = True

            if obj in current_detected:
                box, conf = current_detected[obj]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0)

                if obj in restricted_objects:
                    color = (0, 0, 255)
                    alert_active = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame,
                            f"{obj} ({conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)
        else:
            object_presence[obj] = False

    runtime = int(time.time() - start_time)

    # ========================================================
    # ALERT SYSTEM
    # ========================================================

    if alert_active:
        alert_count += 1
        timestamp = int(time.time())
        cv2.imwrite(f"{logs_dir}/alert_{timestamp}.jpg", frame)

        winsound.Beep(1400 if alert_mode_loud else 700, 200)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
        event_log = "Restricted Object Detected"

    # ========================================================
    # TITLE BAR
    # ========================================================

    cv2.rectangle(frame, (0, 0), (width, TITLE_BAR_HEIGHT), (12, 15, 30), -1)

    cv2.putText(frame,
                "SMART VISION COMMAND CENTER V2.2",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 180),
                2)

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time

    fps_text = f"FPS: {fps}"
    (tw, th), _ = cv2.getTextSize(fps_text,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, 2)

    box_x1 = width - tw - 40
    box_x2 = width - 20

    cv2.rectangle(frame, (box_x1, 15), (box_x2, 50), (25, 30, 45), -1)
    cv2.putText(frame,
                fps_text,
                (box_x1 + 10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 180),
                2)

    # ========================================================
    # DASHBOARD
    # ========================================================

    if show_dashboard:
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, height - DASHBOARD_HEIGHT),
                      (width, height),
                      (15, 18, 30), -1)

        frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)

        left_x = 40
        right_x = width // 2 + 40
        y_start = height - DASHBOARD_HEIGHT + 40

        # SYSTEM STATUS
        cv2.putText(frame,
                    "SYSTEM STATUS",
                    (left_x, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2)

        y = y_start + 35

        cv2.putText(frame,
                    f"Alert Mode: {'LOUD' if alert_mode_loud else 'SUBTLE'}",
                    (left_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if alert_mode_loud else (0, 200, 255),
                    2)

        y += 30

        cv2.putText(frame,
                    f"Alerts Triggered: {alert_count}",
                    (left_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    2)

        y += 30

        cv2.putText(frame,
                    f"Runtime: {runtime}s",
                    (left_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    2)

        # TRACKED OBJECTS
        cv2.putText(frame,
                    "TRACKED OBJECTS",
                    (right_x, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2)

        y2 = y_start + 35

        for obj in tracked_objects:
            present = object_presence.get(obj, False)
            dot_color = (0, 255, 0) if present else (100, 100, 100)

            cv2.circle(frame, (right_x, y2 - 6), 6, dot_color, -1)

            cv2.putText(frame,
                        obj.capitalize(),
                        (right_x + 20, y2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (220, 220, 220),
                        2)

            status_text = "PRESENT" if present else "NOT PRESENT"

            cv2.putText(frame,
                        status_text,
                        (right_x + 200, y2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if present else (150, 150, 150),
                        1)

            y2 += 30

        # LAST EVENT
        cv2.line(frame,
                 (40, height - 45),
                 (width - 40, height - 45),
                 (40, 50, 70),
                 1)

        cv2.putText(frame,
                    f"Last Event: {event_log}",
                    (40, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 200, 255),
                    2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        alert_mode_loud = not alert_mode_loud
        winsound.Beep(1500 if alert_mode_loud else 600, 150)
    elif key == ord('d'):
        show_dashboard = not show_dashboard

cap.release()
cv2.destroyAllWindows()
