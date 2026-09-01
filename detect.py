import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "pose_landmarker.task"

# "lite" = fastest / least accurate, "full" = balanced (used here),
# "heavy" = most accurate / slowest. Swap the filename below to change.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 960          # wider frame so a full standing body fits
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Pose tracking
MAX_POSES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Don't draw/smooth a landmark unless the model is at least this
# confident it's actually visible (not occluded / off-screen).
MIN_VISIBILITY = 0.5

# Landmark smoothing (higher = smoother, lower = more responsive)
SMOOTHING_ALPHA = 0.4

NUM_LANDMARKS = 33

# ============================================================
# POSE SKELETON CONNECTIONS (standard 33-point MediaPipe layout)
# ============================================================

POSE_CONNECTIONS = np.array([
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),

    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),

    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),

    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),

    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
], dtype=np.int32)

# Bigger joints for the major landmarks people actually care about
KEY_JOINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

WHITE = (255, 255, 255)
DARK = (35, 35, 35)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading pose tracking model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        print(f"Failed to download model: {error}")
        raise SystemExit(1)
    print("Model downloaded.")


# ============================================================
# LANDMARK SMOOTHER (vectorized, size-agnostic)
# ============================================================

class LandmarkSmoother:

    __slots__ = ("alpha", "previous")

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, landmarks, visible_mask):
        if self.previous is None:
            self.previous = landmarks.copy()
            self.previous[~visible_mask] = np.nan
            return self.previous

        # Only blend points that are currently visible; frozen
        # (not overwritten with a guess) where visibility is low,
        # so occluded joints don't jitter toward noise.
        blend = visible_mask
        self.previous[blend] += (
            (landmarks[blend] - self.previous[blend]) * self.alpha
        )
        return self.previous


# ============================================================
# DRAWING
# ============================================================

def draw_skeleton(frame, points_int, visible_mask):
    for start, end in POSE_CONNECTIONS:
        if not (visible_mask[start] and visible_mask[end]):
            continue
        cv2.line(
            frame,
            tuple(points_int[start]),
            tuple(points_int[end]),
            WHITE, 2, cv2.LINE_AA
        )

    for index in range(NUM_LANDMARKS):
        if not visible_mask[index]:
            continue
        x, y = points_int[index]
        is_key = index in KEY_JOINTS
        outer = 7 if is_key else 4
        inner = 3 if is_key else 2
        cv2.circle(frame, (x, y), outer, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), inner, DARK, -1, cv2.LINE_AA)


def draw_status(frame, pose_detected, fps):
    text = "BODY TRACKING" if pose_detected else "NO BODY"
    color = (0, 255, 0) if pose_detected else (180, 180, 180)

    cv2.putText(frame, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q = quit", (18, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)


# ============================================================
# SETUP
# ============================================================

ensure_model()
cv2.setUseOptimized(True)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=MAX_POSES,
    min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

smoother = LandmarkSmoother(SMOOTHING_ALPHA)

start_time = time.perf_counter()
previous_time = start_time
fps = 0.0

print()
print("============================================")
print("           FULL BODY TRACKER")
print("============================================")
print(f"Webcam: {CAMERA_INDEX}   Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print("Step back so your whole body is in frame.")
print("Press Q to quit.")
print("============================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.perf_counter() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_detected = bool(result.pose_landmarks)

        if pose_detected:
            pose = result.pose_landmarks[0]

            raw_points = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
            visibility = np.empty(NUM_LANDMARKS, dtype=np.float32)

            for i, lm in enumerate(pose):
                raw_points[i, 0] = lm.x * width
                raw_points[i, 1] = lm.y * height
                raw_points[i, 2] = lm.z
                visibility[i] = lm.visibility

            visible_mask = visibility >= MIN_VISIBILITY

            smoothed = smoother.update(raw_points, visible_mask)
            points_int = smoothed[:, :2].astype(np.int32)

            draw_skeleton(frame, points_int, visible_mask)
        else:
            smoother.reset()

        current_time = time.perf_counter()
        delta = current_time - previous_time
        previous_time = current_time
        if delta > 0:
            fps = fps * 0.90 + (1.0 / delta) * 0.10

        draw_status(frame, pose_detected, fps)

        cv2.imshow("Body Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
print()
print("Body tracker stopped.")