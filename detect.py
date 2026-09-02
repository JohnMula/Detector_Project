import cv2
import mediapipe as mp
import urllib.request
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Hand tracking
MAX_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Landmark smoothing
#
# Higher = smoother but slightly less responsive
# Lower  = more responsive but more jitter
SMOOTHING_ALPHA = 0.45


# ============================================================
# HAND SKELETON CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def ensure_model():

    if os.path.exists(MODEL_PATH):
        return

    print("Hand model not found.")
    print("Downloading hand tracking model...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

    except Exception as error:

        print(
            f"Failed to download model: {error}"
        )

        raise SystemExit(1)

    print("Model downloaded successfully.")


# ============================================================
# LANDMARK SMOOTHER
# ============================================================

class LandmarkSmoother:

    def __init__(self, alpha=0.45):

        self.alpha = alpha
        self.previous = None

    def reset(self):

        self.previous = None

    def update(self, landmarks):

        # First frame
        if self.previous is None:

            self.previous = [
                tuple(point)
                for point in landmarks
            ]

            return self.previous

        alpha = self.alpha

        smoothed = []

        for old, new in zip(
            self.previous,
            landmarks
        ):

            x = (
                old[0] * (1.0 - alpha)
                +
                new[0] * alpha
            )

            y = (
                old[1] * (1.0 - alpha)
                +
                new[1] * alpha
            )

            z = (
                old[2] * (1.0 - alpha)
                +
                new[2] * alpha
            )

            smoothed.append(
                (x, y, z)
            )

        self.previous = smoothed

        return smoothed


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    points
):

    # --------------------------------------------------------
    # Draw skeleton
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        x1 = int(points[start][0])
        y1 = int(points[start][1])

        x2 = int(points[end][0])
        y2 = int(points[end][1])

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw joints
    # --------------------------------------------------------

    for index, point in enumerate(points):

        x = int(point[0])
        y = int(point[1])

        # Larger wrist and fingertips
        if index in (
            0,
            4,
            8,
            12,
            16,
            20
        ):
            outer_radius = 6
            inner_radius = 3

        else:
            outer_radius = 5
            inner_radius = 2

        # White outer ring
        cv2.circle(
            frame,
            (x, y),
            outer_radius + 2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        # Dark center
        cv2.circle(
            frame,
            (x, y),
            inner_radius,
            (35, 35, 35),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW SIMPLE STATUS
# ============================================================

def draw_status(
    frame,
    hand_detected,
    fps
):

    if hand_detected:

        text = "HAND TRACKING"

        text_color = (
            0,
            255,
            0
        )

    else:

        text = "NO HAND"

        text_color = (
            180,
            180,
            180
        )

    cv2.putText(
        frame,
        text,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Quit instruction
    cv2.putText(
        frame,
        "Q = quit",
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

ensure_model()

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=MAX_HANDS,

    min_hand_detection_confidence=(
        MIN_DETECTION_CONFIDENCE
    ),

    min_hand_presence_confidence=(
        MIN_PRESENCE_CONFIDENCE
    ),

    min_tracking_confidence=(
        MIN_TRACKING_CONFIDENCE
    )
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    raise SystemExit(1)


# Try to use requested camera settings
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


# ------------------------------------------------------------
# Optional performance settings
# ------------------------------------------------------------

# Reduce internal buffering where supported.
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# STATE
# ============================================================

smoother = LandmarkSmoother(
    SMOOTHING_ALPHA
)

timestamp_ms = 0

previous_time = time.perf_counter()

fps = 0.0


# ============================================================
# START
# ============================================================

print()
print("============================================")
print("           HAND TRACKER")
print("============================================")
print()
print("Webcam:", CAMERA_INDEX)
print("Resolution:", CAMERA_WIDTH, "x", CAMERA_HEIGHT)
print("Hands:", MAX_HANDS)
print()
print("Press Q to quit.")
print("============================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read webcam frame."
            )

            break

        # ----------------------------------------------------
        # Mirror
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        # MediaPipe VIDEO mode requires increasing timestamps.
        timestamp_ms += 1


        # ----------------------------------------------------
        # Hand detection
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        hand_detected = False


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand_detected = True

            # We only requested one hand
            hand = result.hand_landmarks[0]

            raw_points = []

            # ------------------------------------------------
            # Convert normalized landmarks into pixel space
            # ------------------------------------------------

            for landmark in hand:

                raw_points.append(
                    (
                        landmark.x * width,
                        landmark.y * height,
                        landmark.z
                    )
                )


            # ------------------------------------------------
            # Smooth landmarks
            # ------------------------------------------------

            points = smoother.update(
                raw_points
            )


            # ------------------------------------------------
            # Draw skeleton
            # ------------------------------------------------

            draw_hand(
                frame,
                points
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            # Reset smoother so the next hand does not
            # interpolate from an old position.
            smoother.reset()


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.perf_counter()

        delta = (
            current_time -
            previous_time
        )

        previous_time = current_time

        if delta > 0:

            instant_fps = 1.0 / delta

            # Smooth FPS display
            fps = (
                fps * 0.90 +
                instant_fps * 0.10
            )


        # ====================================================
        # UI
        # ====================================================

        draw_status(
            frame,
            hand_detected,
            fps
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Hand Tracker",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Hand tracker stopped.")