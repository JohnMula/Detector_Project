import cv2
import mediapipe as mp
import urllib.request
import os
import time
import threading


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ------------------------------------------------------------
# Hand detector
# ------------------------------------------------------------

MAX_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.50
MIN_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------
#
# Lower = more responsive
# Higher = smoother
#
# 0.55 is intentionally fairly responsive.
#

SMOOTHING_ALPHA = 0.55


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
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

    if os.path.isfile(MODEL_PATH):
        return

    print("Hand model not found.")
    print("Downloading...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

    except Exception as error:

        print(
            f"Could not download model: {error}"
        )

        raise SystemExit(1)

    print("Model downloaded.")


# ============================================================
# LANDMARK SMOOTHER
# ============================================================

class LandmarkSmoother:

    def __init__(self, alpha):

        self.alpha = alpha
        self.points = None

    def reset(self):

        self.points = None

    def update(self, landmarks):

        if self.points is None:

            self.points = [
                [
                    point[0],
                    point[1],
                    point[2]
                ]
                for point in landmarks
            ]

            return self.points

        alpha = self.alpha
        inverse = 1.0 - alpha

        for i in range(len(landmarks)):

            self.points[i][0] = (
                self.points[i][0] * inverse
                +
                landmarks[i][0] * alpha
            )

            self.points[i][1] = (
                self.points[i][1] * inverse
                +
                landmarks[i][1] * alpha
            )

            self.points[i][2] = (
                self.points[i][2] * inverse
                +
                landmarks[i][2] * alpha
            )

        return self.points


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, points):

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,

            (
                int(points[start][0]),
                int(points[start][1])
            ),

            (
                int(points[end][0]),
                int(points[end][1])
            ),

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Joints
    # --------------------------------------------------------

    for index, point in enumerate(points):

        x = int(point[0])
        y = int(point[1])

        # Make wrist + fingertips slightly larger
        if index in (
            0,
            4,
            8,
            12,
            16,
            20
        ):

            outer = 6
            inner = 3

        else:

            outer = 5
            inner = 2

        # White outer circle
        cv2.circle(
            frame,
            (x, y),
            outer + 2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        # Dark center
        cv2.circle(
            frame,
            (x, y),
            inner,
            (35, 35, 35),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# CALLBACK RESULT STATE
# ============================================================
#
# MediaPipe LIVE_STREAM calls this function on its own
# processing thread.
#
# We only keep the MOST RECENT result.
#
# This is important:
#
# Old frames are useless for a real-time tracker.
#
# ============================================================

latest_result = None
latest_timestamp = -1

result_lock = threading.Lock()


def result_callback(
    result,
    output_image,
    timestamp_ms
):

    global latest_result
    global latest_timestamp

    with result_lock:

        latest_result = result
        latest_timestamp = timestamp_ms


# ============================================================
# FPS
# ============================================================

class FPSCounter:

    def __init__(self):

        self.last_time = time.perf_counter()

        self.fps = 0.0

    def update(self):

        now = time.perf_counter()

        dt = now - self.last_time

        self.last_time = now

        if dt > 0:

            instant = 1.0 / dt

            # Stable FPS display without affecting tracking
            self.fps = (
                self.fps * 0.90
                +
                instant * 0.10
            )

        return self.fps


# ============================================================
# INITIALIZE
# ============================================================

ensure_model()


# ============================================================
# MEDIAPIPE OPTIONS
# ============================================================

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

    # IMPORTANT:
    # LIVE_STREAM is asynchronous and designed for
    # camera/live input.
    running_mode=RunningMode.LIVE_STREAM,

    num_hands=MAX_HANDS,

    min_hand_detection_confidence=(
        MIN_DETECTION_CONFIDENCE
    ),

    min_hand_presence_confidence=(
        MIN_PRESENCE_CONFIDENCE
    ),

    min_tracking_confidence=(
        MIN_TRACKING_CONFIDENCE
    ),

    result_callback=result_callback
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# Camera configuration
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

# ------------------------------------------------------------
# Reduce buffering
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ------------------------------------------------------------
# Try MJPG where supported.
#
# Many Windows webcams handle MJPG efficiently.
# If the driver ignores it, nothing breaks.
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        "M",
        "J",
        "P",
        "G"
    )
)


# ============================================================
# TRACKING OBJECTS
# ============================================================

smoother = LandmarkSmoother(
    SMOOTHING_ALPHA
)

fps_counter = FPSCounter()

frame_timestamp = 0


# Keep track of which result frame we have rendered.
last_rendered_timestamp = -1


# ============================================================
# START MEDIAPIPE
# ============================================================

print()
print("============================================")
print("        LOW-LATENCY HAND TRACKER")
print("============================================")
print()
print(f"Camera      : {CAMERA_INDEX}")
print(
    f"Resolution  : "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
)
print(
    f"Max hands   : {MAX_HANDS}"
)
print()
print("Live stream mode: ON")
print("Press Q to quit.")
print("============================================")
print()


with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # CAPTURE FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        height, width = frame.shape[:2]


        # ====================================================
        # SEND FRAME TO MEDIAPIPE
        # ====================================================

        # Convert only what MediaPipe needs.
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # LIVE_STREAM requires monotonically increasing
        # timestamps.
        frame_timestamp += 1


        landmarker.detect_async(
            mp_image,
            frame_timestamp
        )


        # ====================================================
        # GET LATEST RESULT
        # ====================================================

        with result_lock:

            result = latest_result
            result_timestamp = latest_timestamp


        hand_detected = False


        # ====================================================
        # DRAW THE MOST RECENT HAND RESULT
        # ====================================================

        if (
            result is not None
            and
            result.hand_landmarks
            and
            result_timestamp != last_rendered_timestamp
        ):

            hand_detected = True

            # First hand only.
            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # Convert normalized coordinates to pixels.
            # ------------------------------------------------

            raw_points = []

            for landmark in hand:

                raw_points.append(
                    (
                        landmark.x * width,
                        landmark.y * height,
                        landmark.z
                    )
                )


            # ------------------------------------------------
            # Smooth the landmarks.
            # ------------------------------------------------

            points = smoother.update(
                raw_points
            )


            # ------------------------------------------------
            # Draw.
            # ------------------------------------------------

            draw_hand(
                frame,
                points
            )


            last_rendered_timestamp = (
                result_timestamp
            )


        elif result is None:

            # No result has arrived yet.
            hand_detected = False


        else:

            # Result hasn't changed since the last frame.
            #
            # We deliberately DON'T rerun calculations.
            # We just display the camera frame.
            hand_detected = bool(
                result.hand_landmarks
            )


        # ====================================================
        # IF HAND DISAPPEARS
        # ====================================================

        if (
            result is not None
            and
            not result.hand_landmarks
        ):

            smoother.reset()


        # ====================================================
        # FPS
        # ====================================================

        fps = fps_counter.update()


        # ====================================================
        # RESULT AGE
        # ====================================================

        # This isn't exact end-to-end camera latency, but it
        # helps identify whether the latest inference result
        # is falling behind the camera.
        #
        result_age = 0.0

        if result_timestamp >= 0:

            result_age = max(
                0,
                frame_timestamp -
                result_timestamp
            )


        # ====================================================
        # STATUS
        # ====================================================

        if hand_detected:

            status = "HAND TRACKING"

            status_color = (
                0,
                255,
                0
            )

        else:

            status = "SEARCHING"

            status_color = (
                180,
                180,
                180
            )


        cv2.putText(
            frame,
            status,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
            cv2.LINE_AA
        )


        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # Result age
        cv2.putText(
            frame,
            f"Result age: {result_age} frames",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Low Latency Hand Tracker",
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