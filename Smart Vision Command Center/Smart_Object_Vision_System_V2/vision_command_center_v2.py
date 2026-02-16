import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
import winsound

# =========================
# CONFIGURATION
# =========================

CONF_THRESHOLD = 0.45
STABILITY_WINDOW = 12
MIN_DETECTIONS = 4
DASHBOARD_HEIGHT = 170
TITLE_HEIGHT = 60

DEFAULT_TRACKED = ["person", "bottle", "cell phone"]
DEFAULT_RESTRICTED = ["cell phone"]

# =========================
# INITIALIZATION
# =========================

model = YOLO("yolov8s.pt")

tracked_objects = DEFAULT_TRACKED.copy()
restricted_objects = DEFAULT_RESTRICTED.copy()

alert_mode_loud = True
show_dashboard = True

detection_memory = {}
object_presence = {}
alert_count = 0
event_log = "System Initialized"

logs_dir = "../logs/screenshots"
os.makedirs(logs_dir, exist_ok=True)

start_time = time.time()

# =========================
# STARTUP SCREEN
# =========================

def draw_startup_screen():
    screen = np.zeros((600, 1000, 3), dtype=np.uint8)
    screen[:] = (8, 10, 18)

    while True:
        screen[:] = (8, 10, 18)

        cv2.putText(screen,
                    "SMART VISION COMMAND CENTER",
                    (150, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 180),
                    3)

        cv2.putText(screen,
                    "Advanced Real-Time AI Monitoring System",
                    (250, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),
                    1)

        cv2.line(screen, (150, 200), (850, 200), (40, 60, 80), 2)

        cv2.putText(screen,
                    "Default Tracking:",
                    (200, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2)

        cv2.putText(screen,
                    ", ".join(tracked_objects),
                    (220, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 120),
                    2)

        cv2.putText(screen,
                    "Restricted Objects:",
                    (200, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2)

        cv2.putText(screen,
                    ", ".join(restricted_objects),
                    (220, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 120, 255),
                    2)

        cv2.putText(screen,
                    "Press ENTER to Launch Monitoring System",
                    (260, 480),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(screen,
                    "Press T to Edit Tracking | R to Edit Restricted",
                    (260, 520),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (130, 130, 130),
                    1)

        cv2.imshow("Smart Vision Command Center", screen)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:
            break
        elif key == ord('t'):
            edit_objects("tracked")
        elif key == ord('r'):
            edit_objects("restricted")

    cv2.destroyAllWindows()

def edit_objects(mode):
    global tracked_objects, restricted_objects
    buffer = ""

    while True:
        screen = np.zeros((350, 900, 3), dtype=np.uint8)
        screen[:] = (15, 18, 30)

        cv2.putText(screen,
                    f"Edit {mode.upper()} Objects (comma separated)",
                    (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.putText(screen,
                    buffer,
                    (80, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2)

        cv2.putText(screen,
                    "ENTER to Apply | ESC to Cancel",
                    (80, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1)

        cv2.imshow("Edit Mode", screen)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 13:
            new_list = [x.strip().lower() for x in buffer.split(",") if x.strip()]
            if mode == "tracked":
                tracked_objects = new_list
            else:
                restricted_objects = new_list
            break
        elif key == 8:
            buffer = buffer[:-1]
        elif 32 <= key <= 126:
            buffer += chr(key)

    cv2.destroyWindow("Edit Mode")

# Launch startup
draw_startup_screen()

# =========================
# MAIN MONITOR LOOP
# =========================

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    for obj in tracked_objects:
        detection_memory.setdefault(obj, [])

    results = model(frame, verbose=False)
    detections = results[0]

    current_frame_detected = {}

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])

        if name in tracked_objects and conf > CONF_THRESHOLD:
            detection_memory[name].append(1)
            current_frame_detected[name] = (box, conf)
        else:
            if name in detection_memory:
                detection_memory[name].append(0)

    for obj in detection_memory:
        detection_memory[obj] = detection_memory[obj][-STABILITY_WINDOW:]

    alert_active = False

    for obj in tracked_objects:
        if sum(detection_memory[obj]) >= MIN_DETECTIONS:
            object_presence[obj] = True

            if obj in current_frame_detected:
                box, conf = current_frame_detected[obj]
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

    if alert_active:
        alert_count += 1
        timestamp = int(time.time())
        cv2.imwrite(f"{logs_dir}/alert_{timestamp}.jpg", frame)

        if alert_mode_loud:
            winsound.Beep(1400, 200)
        else:
            winsound.Beep(700, 150)

        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
        event_log = "Restricted Object Detected"

    # =========================
    # TITLE BAR
    # =========================

    cv2.rectangle(frame, (0, 0), (width, TITLE_HEIGHT), (12, 15, 30), -1)

    cv2.putText(frame,
                "SMART VISION COMMAND CENTER V2",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 180),
                2)

    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
    prev_time = current_time

    cv2.putText(frame,
                f"FPS: {fps}",
                (width - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2)

    # =========================
    # DASHBOARD
    # =========================

    if show_dashboard:
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, height - DASHBOARD_HEIGHT),
                      (width, height),
                      (15, 18, 30), -1)

        frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)

        y = height - DASHBOARD_HEIGHT + 40

        cv2.putText(frame,
                    "SYSTEM STATUS",
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2)

        y += 35

        cv2.putText(frame,
                    f"Alert Mode: {'LOUD' if alert_mode_loud else 'SUBTLE'}",
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if alert_mode_loud else (0, 200, 255),
                    2)

        y += 30

        cv2.putText(frame,
                    f"Alerts Triggered: {alert_count}",
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2)

        y += 30

        cv2.putText(frame,
                    f"Runtime: {runtime}s",
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2)

        y += 40

        cv2.putText(frame,
                    "TRACKED OBJECTS",
                    (450, height - DASHBOARD_HEIGHT + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 180),
                    2)

        y2 = height - DASHBOARD_HEIGHT + 75

        for obj in tracked_objects:
            status = object_presence.get(obj, False)
            color = (0, 255, 0) if status else (100, 100, 100)

            cv2.circle(frame, (470, y2 - 5), 6, color, -1)
            cv2.putText(frame,
                        obj.capitalize(),
                        (490, y2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (220, 220, 220),
                        2)
            y2 += 30

    cv2.imshow("Smart Vision Command Center", frame)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        alert_mode_loud = not alert_mode_loud
        winsound.Beep(1500 if alert_mode_loud else 600, 150)
        event_log = f"Alert Mode: {'LOUD' if alert_mode_loud else 'SUBTLE'}"
    elif key == ord('d'):
        show_dashboard = not show_dashboard

cap.release()
cv2.destroyAllWindows()
