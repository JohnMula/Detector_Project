import cv2
import mediapipe as mp
import urllib.request
import os
import math
import time
from collections import deque


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

# ------------------------------------------------------------
# Hand tracking
# ------------------------------------------------------------

MAX_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.55
MIN_PRESENCE_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55

# ------------------------------------------------------------
# Landmark smoothing
# ------------------------------------------------------------

# 0.0 = no smoothing
# 1.0 = completely frozen
SMOOTHING_ALPHA = 0.35

# ------------------------------------------------------------
# Finger pose thresholds
# ------------------------------------------------------------

# Finger must be this open to become "armed"
FINGER_OPEN_ANGLE = 158.0

# Finger becomes curled below this angle
FINGER_PRESS_ANGLE = 135.0

# Normalized fingertip-to-MCP distance
#
# Larger = extended
# Smaller = curled
FINGER_OPEN_DISTANCE = 2.25
FINGER_PRESS_DISTANCE = 1.80

# ------------------------------------------------------------
# Press movement requirements
# ------------------------------------------------------------

# During a real press, the finger should move.
#
# This is deliberately small because webcam depth is noisy.
MIN_PRESS_MOTION = 0.025

# Optional depth movement:
#
# MediaPipe's relative z becomes smaller as a point moves
# toward the camera.
MIN_FORWARD_Z_MOTION = 0.008

# How many consecutive frames the press must be confirmed
PRESS_CONFIRM_FRAMES = 2

# How many consecutive frames finger must be open before
# we consider it released / ready for the next press
RELEASE_CONFIRM_FRAMES = 3

# Minimum time between registered keys
CLICK_COOLDOWN = 0.22

# ------------------------------------------------------------
# Keyboard
# ------------------------------------------------------------

KEYBOARD_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]

SPECIAL_KEYS = {
    "SPACE": " ",
    "BACK": "BACK",
    "ENTER": "\n",
}


# ============================================================
# DOWNLOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_tasks = mp

BaseOptions = mp_tasks.tasks.BaseOptions
HandLandmarker = mp_tasks.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp_tasks.tasks.vision.RunningMode


# ============================================================
# HAND SKELETON
# ============================================================

CONNECTIONS = [
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
# GLOBAL STATE
# ============================================================

typed_text = ""

pressed_key = None
pressed_key_until = 0.0

last_click_time = 0.0

# Current interaction state
#
# IDLE
# HOVER
# ARMED
# PRESSING
# LOCKED
#
state = "IDLE"

# Confirmation counters
press_frames = 0
release_frames = 0
arm_frames = 0

# Target key is latched when press begins.
# This prevents the finger moving slightly across another
# key during the actual click from producing the wrong key.
locked_key = None

# Smoothed landmarks
smoothed_points = None

# Previous fingertip data
previous_tip = None
previous_tip_z = None
previous_time = None

# Small motion history
tip_motion_history = deque(maxlen=5)


# ============================================================
# MATH
# ============================================================

def distance_2d(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def angle_3_points(a, b, c):
    """
    Angle ABC in degrees.
    """

    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]

    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    magnitude_ba = math.hypot(
        ba_x,
        ba_y
    )

    magnitude_bc = math.hypot(
        bc_x,
        bc_y
    )

    if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
        return 180.0

    dot = (
        ba_x * bc_x +
        ba_y * bc_y
    )

    cosine = dot / (
        magnitude_ba *
        magnitude_bc
    )

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


# ============================================================
# LANDMARK SMOOTHING
# ============================================================

def smooth_landmarks(points):
    """
    Exponential moving average.

    This reduces tiny landmark jumps that can otherwise cause
    accidental press transitions.
    """

    global smoothed_points

    if smoothed_points is None:
        smoothed_points = [
            tuple(point)
            for point in points
        ]

        return smoothed_points

    alpha = SMOOTHING_ALPHA

    output = []

    for old, new in zip(
        smoothed_points,
        points
    ):
        x = (
            old[0] * (1.0 - alpha) +
            new[0] * alpha
        )

        y = (
            old[1] * (1.0 - alpha) +
            new[1] * alpha
        )

        z = (
            old[2] * (1.0 - alpha) +
            new[2] * alpha
        )

        output.append(
            (x, y, z)
        )

    smoothed_points = output

    return output


# ============================================================
# INDEX FINGER FEATURES
# ============================================================

def get_index_features(points):
    """
    Returns:

        pip_angle
        normalized_tip_distance
        hand_scale
    """

    wrist = points[0]
    mcp = points[5]
    pip = points[6]
    dip = points[7]
    tip = points[8]

    pip_angle = angle_3_points(
        mcp,
        pip,
        dip
    )

    hand_scale = distance_2d(
        wrist,
        mcp
    )

    if hand_scale < 1.0:
        hand_scale = 1.0

    tip_mcp_distance = distance_2d(
        mcp,
        tip
    )

    normalized_distance = (
        tip_mcp_distance /
        hand_scale
    )

    return (
        pip_angle,
        normalized_distance,
        hand_scale
    )


# ============================================================
# FINGER STATES
# ============================================================

def is_finger_open(
    pip_angle,
    normalized_distance
):
    """
    Strictly open/extended.

    BOTH conditions are required.

    This is important because one noisy feature alone
    should not arm the keyboard.
    """

    return (
        pip_angle >= FINGER_OPEN_ANGLE
        and
        normalized_distance >= FINGER_OPEN_DISTANCE
    )


def is_finger_pressed_pose(
    pip_angle,
    normalized_distance
):
    """
    Clearly curled / typing position.

    BOTH conditions are required.
    """

    return (
        pip_angle <= FINGER_PRESS_ANGLE
        and
        normalized_distance <= FINGER_PRESS_DISTANCE
    )


# ============================================================
# KEYBOARD
# ============================================================

def build_keyboard(frame_width, frame_height):

    keys = []

    key_w = 54
    key_h = 47
    gap = 8

    keyboard_y = frame_height - 208

    # --------------------------------------------------------
    # Main rows
    # --------------------------------------------------------

    for row_index, row in enumerate(
        KEYBOARD_ROWS
    ):

        total_width = (
            len(row) * key_w +
            (len(row) - 1) * gap
        )

        start_x = (
            frame_width -
            total_width
        ) // 2

        y = keyboard_y + (
            row_index *
            (key_h + gap)
        )

        for column_index, letter in enumerate(
            row
        ):

            x = (
                start_x +
                column_index *
                (key_w + gap)
            )

            keys.append({
                "key": letter,
                "x1": x,
                "y1": y,
                "x2": x + key_w,
                "y2": y + key_h
            })

    # --------------------------------------------------------
    # Bottom row
    # --------------------------------------------------------

    bottom_y = (
        keyboard_y +
        3 * (key_h + gap)
    )

    # Backspace
    back_w = 85

    keys.append({
        "key": "BACK",
        "x1": 55,
        "y1": bottom_y,
        "x2": 55 + back_w,
        "y2": bottom_y + key_h
    })

    # Space
    space_w = 245

    space_x = (
        frame_width -
        space_w
    ) // 2

    keys.append({
        "key": "SPACE",
        "x1": space_x,
        "y1": bottom_y,
        "x2": space_x + space_w,
        "y2": bottom_y + key_h
    })

    # Enter
    enter_w = 85

    keys.append({
        "key": "ENTER",
        "x1": frame_width - 55 - enter_w,
        "y1": bottom_y,
        "x2": frame_width - 55,
        "y2": bottom_y + key_h
    })

    return keys


def get_key_at_position(
    x,
    y,
    keys
):

    for key in keys:

        if (
            key["x1"] <= x <= key["x2"]
            and
            key["y1"] <= y <= key["y2"]
        ):
            return key

    return None


# ============================================================
# TYPE KEY
# ============================================================

def type_key(key):

    global typed_text

    if key == "BACK":

        typed_text = typed_text[:-1]

    elif key == "SPACE":

        typed_text += " "

    elif key == "ENTER":

        typed_text += "\n"

    else:

        typed_text += key


# ============================================================
# DRAW TEXT FIELD
# ============================================================

def draw_text_field(frame):

    height, width, _ = frame.shape

    x1 = 45
    x2 = width - 45

    y1 = 25
    y2 = 108

    # Background
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (245, 245, 245),
        -1,
        cv2.LINE_AA
    )

    # Border
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (80, 80, 80),
        2,
        cv2.LINE_AA
    )

    # Display last part of text
    display_text = typed_text[-45:]

    cv2.putText(
        frame,
        display_text,
        (x1 + 15, y1 + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (25, 25, 25),
        2,
        cv2.LINE_AA
    )

    # Cursor
    text_width = cv2.getTextSize(
        display_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        2
    )[0][0]

    cursor_x = (
        x1 +
        15 +
        text_width +
        3
    )

    cv2.line(
        frame,
        (cursor_x, y1 + 20),
        (cursor_x, y2 - 14),
        (35, 35, 35),
        2
    )


# ============================================================
# DRAW KEYBOARD
# ============================================================

def draw_keyboard(
    frame,
    keys,
    hovered_key=None
):

    current_time = time.monotonic()

    for key in keys:

        x1 = key["x1"]
        y1 = key["y1"]
        x2 = key["x2"]
        y2 = key["y2"]

        name = key["key"]

        # ----------------------------------------------------
        # Key state
        # ----------------------------------------------------

        flash = (
            pressed_key == name
            and
            current_time < pressed_key_until
        )

        hover = (
            hovered_key == name
        )

        # Pressed
        if flash:

            fill = (
                255,
                255,
                255
            )

            text_color = (
                25,
                25,
                25
            )

            border = (
                255,
                255,
                255
            )

        # Hover
        elif hover:

            fill = (
                70,
                70,
                70
            )

            text_color = (
                255,
                255,
                255
            )

            border = (
                210,
                210,
                210
            )

        # Normal
        else:

            fill = (
                32,
                32,
                32
            )

            text_color = (
                245,
                245,
                245
            )

            border = (
                120,
                120,
                120
            )

        # ----------------------------------------------------
        # Key
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            fill,
            -1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            border,
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if name == "BACK":
            label = "<"

        elif name == "ENTER":
            label = "ENTER"

        elif name == "SPACE":
            label = "SPACE"

        else:
            label = name

        font_scale = (
            0.68
            if name not in ("SPACE", "ENTER")
            else 0.52
        )

        thickness = 2

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )[0]

        text_x = (
            x1 +
            (
                x2 -
                x1 -
                text_size[0]
            ) // 2
        )

        text_y = (
            y1 +
            (
                y2 -
                y1 +
                text_size[1]
            ) // 2
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )


# ============================================================
# DRAW HAND SKELETON
# ============================================================

def draw_hand(
    frame,
    points
):

    # --------------------------------------------------------
    # Bones
    # --------------------------------------------------------

    for start, end in CONNECTIONS:

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
    # Joints
    # --------------------------------------------------------

    for index, point in enumerate(
        points
    ):

        x = int(point[0])
        y = int(point[1])

        radius = (
            7
            if index in (0, 8)
            else 5
        )

        cv2.circle(
            frame,
            (x, y),
            radius + 2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            (x, y),
            radius,
            (40, 40, 40),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW STATUS
# ============================================================

def draw_status(
    frame,
    current_state,
    hovered_key,
    pip_angle,
    normalized_distance,
    motion_amount
):

    # State text
    if current_state == "IDLE":

        status = "NO HAND"

    elif current_state == "HOVER":

        status = "HOVER - POINT"

    elif current_state == "ARMED":

        status = "READY - PRESS"

    elif current_state == "PRESSING":

        status = "PRESSING"

    elif current_state == "LOCKED":

        status = "RELEASE TO RESET"

    else:

        status = current_state

    cv2.putText(
        frame,
        status,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Target
    target_text = (
        f"Target: {hovered_key}"
        if hovered_key
        else "Target: -"
    )

    cv2.putText(
        frame,
        target_text,
        (20, 173),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Debug values
    cv2.putText(
        frame,
        f"Angle: {pip_angle:5.1f}",
        (20, 197),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Distance: {normalized_distance:.2f}",
        (20, 219),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Motion: {motion_amount:.3f}",
        (20, 241),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )


# ============================================================
# MEDIAPIPE OPTIONS
# ============================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

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

    raise SystemExit


# Try to use requested resolution
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


print()
print("============================================")
print("       VIRTUAL HAND KEYBOARD")
print("============================================")
print()
print("Hovering over a key DOES NOT type.")
print()
print("1. Extend your index finger.")
print("2. Point at a key.")
print("3. Push/curl the index finger.")
print("4. Release the finger before another key.")
print()
print("Press Q to quit.")
print("============================================")
print()


# ============================================================
# VIDEO LOOP
# ============================================================

timestamp_ms = 0

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ----------------------------------------------------
        # Capture
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to grab frame."
            )

            break

        # Mirror
        frame = cv2.flip(
            frame,
            1
        )

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        keys = build_keyboard(
            width,
            height
        )

        # ----------------------------------------------------
        # Convert frame
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp_tasks.Image(
            image_format=(
                mp_tasks.ImageFormat.SRGB
            ),
            data=rgb
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = max(
            timestamp_ms + 1,
            int(time.monotonic() * 1000)
        )

        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hovered_key = None

        pip_angle = 180.0

        normalized_distance = 9.0

        motion_amount = 0.0

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            # Only use first hand
            hand = result.hand_landmarks[0]

            raw_points = []

            for landmark in hand:

                x = (
                    landmark.x *
                    width
                )

                y = (
                    landmark.y *
                    height
                )

                z = landmark.z

                raw_points.append(
                    (x, y, z)
                )

            # ------------------------------------------------
            # Smooth points
            # ------------------------------------------------

            points = smooth_landmarks(
                raw_points
            )

            # ------------------------------------------------
            # Draw skeleton
            # ------------------------------------------------

            draw_hand(
                frame,
                points
            )

            # ------------------------------------------------
            # Feature calculation
            # ------------------------------------------------

            (
                pip_angle,
                normalized_distance,
                hand_scale
            ) = get_index_features(
                points
            )

            # ------------------------------------------------
            # Finger tip
            # ------------------------------------------------

            tip_x = int(points[8][0])
            tip_y = int(points[8][1])
            tip_z = points[8][2]

            # ------------------------------------------------
            # Find keyboard target
            # ------------------------------------------------

            hovered = get_key_at_position(
                tip_x,
                tip_y,
                keys
            )

            if hovered:

                hovered_key = hovered["key"]

            # ------------------------------------------------
            # Calculate fingertip motion
            # ------------------------------------------------

            current_time = time.monotonic()

            if (
                previous_tip is not None
                and
                previous_time is not None
            ):

                dt = max(
                    current_time -
                    previous_time,
                    0.001
                )

                dx = (
                    points[8][0] -
                    previous_tip[0]
                )

                dy = (
                    points[8][1] -
                    previous_tip[1]
                )

                pixel_motion = math.hypot(
                    dx,
                    dy
                )

                # Normalize movement by hand size.
                #
                # This makes the movement threshold less
                # dependent on how close the hand is to the
                # camera.
                motion_amount = (
                    pixel_motion /
                    max(hand_scale, 1.0)
                )

                tip_motion_history.append(
                    motion_amount
                )

                # Depth change.
                #
                # Negative dz means the fingertip moved
                # toward the camera in MediaPipe's relative
                # coordinate convention.
                z_motion = 0.0

                if previous_tip_z is not None:

                    z_motion = (
                        previous_tip_z -
                        tip_z
                    )

            else:

                z_motion = 0.0

                tip_motion_history.clear()

            previous_tip = (
                points[8][0],
                points[8][1]
            )

            previous_tip_z = tip_z
            previous_time = current_time

            # ------------------------------------------------
            # Finger states
            # ------------------------------------------------

            finger_open = is_finger_open(
                pip_angle,
                normalized_distance
            )

            finger_pressed = (
                is_finger_pressed_pose(
                    pip_angle,
                    normalized_distance
                )
            )

            # Average very recent movement
            if tip_motion_history:

                average_motion = (
                    sum(tip_motion_history) /
                    len(tip_motion_history)
                )

            else:

                average_motion = 0.0

            # Strong movement condition
            moving = (
                average_motion >=
                MIN_PRESS_MOTION
            )

            # Forward depth movement
            forward_motion = (
                z_motion >=
                MIN_FORWARD_Z_MOTION
            )

            # ------------------------------------------------
            # STATE MACHINE
            # ------------------------------------------------

            # =================================================
            # IDLE
            # =================================================

            if state == "IDLE":

                arm_frames = 0
                press_frames = 0
                release_frames = 0
                locked_key = None

                if hovered_key is not None:

                    state = "HOVER"

            # =================================================
            # HOVER
            # =================================================

            elif state == "HOVER":

                # No key = return to idle
                if hovered_key is None:

                    state = "IDLE"

                    arm_frames = 0

                else:

                    # A straight finger is required
                    # before a press can occur.
                    if finger_open:

                        arm_frames += 1

                    else:

                        arm_frames = 0

                    # Arm after a few stable frames
                    if arm_frames >= 2:

                        state = "ARMED"

                        arm_frames = 0

            # =================================================
            # ARMED
            # =================================================

            elif state == "ARMED":

                # If finger goes away from keyboard,
                # cancel this attempt.
                if hovered_key is None:

                    state = "IDLE"

                    press_frames = 0
                    release_frames = 0
                    locked_key = None

                # If finger returns to a stable open
                # position, remain armed.
                elif finger_open:

                    press_frames = 0

                else:

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # We DO NOT click simply because the finger
                    # is curled.
                    #
                    # It has to:
                    #
                    # 1. Become clearly curled
                    # 2. Be moving
                    #
                    # or:
                    #
                    # 1. Become clearly curled
                    # 2. Show forward depth movement
                    #
                    # This is what separates a push from hover.
                    # ------------------------------------------------

                    intentional_motion = (
                        moving
                        or
                        forward_motion
                    )

                    if (
                        finger_pressed
                        and
                        intentional_motion
                    ):

                        press_frames += 1

                    else:

                        # Slowly bending without a push does
                        # not count.
                        press_frames = max(
                            press_frames - 1,
                            0
                        )

                    if (
                        press_frames >=
                        PRESS_CONFIRM_FRAMES
                    ):

                        # ------------------------------------------------
                        # LATCH THE KEY
                        # ------------------------------------------------
                        #
                        # Once the press begins, the target is fixed.
                        # This prevents tiny fingertip movement from
                        # switching to an adjacent key.
                        #
                        locked_key = hovered_key

                        state = "PRESSING"

                        press_frames = 0

            # =================================================
            # PRESSING
            # =================================================

            elif state == "PRESSING":

                # Register exactly ONE key.
                if locked_key is not None:

                    now = time.monotonic()

                    if (
                        now -
                        last_click_time
                    ) >= CLICK_COOLDOWN:

                        type_key(
                            locked_key
                        )

                        pressed_key = (
                            locked_key
                        )

                        pressed_key_until = (
                            now + 0.14
                        )

                        last_click_time = now

                # After clicking, lock until release.
                state = "LOCKED"

                release_frames = 0

            # =================================================
            # LOCKED
            # =================================================

            elif state == "LOCKED":

                # IMPORTANT:
                #
                # A curled finger staying on the key DOES NOT
                # type again.
                #
                # The user must release / straighten the finger.
                if finger_open:

                    release_frames += 1

                else:

                    release_frames = 0

                if (
                    release_frames >=
                    RELEASE_CONFIRM_FRAMES
                ):

                    locked_key = None
                    press_frames = 0

                    # If still pointing at a key,
                    # immediately become armed for next press.
                    if hovered_key is not None:

                        state = "ARMED"

                    else:

                        state = "IDLE"

            # ------------------------------------------------
            # Draw fingertip indicator
            # ------------------------------------------------

            cursor_color = (
                (0, 255, 0)
                if state in (
                    "ARMED",
                    "PRESSING"
                )
                else
                (255, 255, 255)
            )

            cv2.circle(
                frame,
                (
                    tip_x,
                    tip_y
                ),
                12,
                cursor_color,
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            draw_status(
                frame,
                state,
                hovered_key,
                pip_angle,
                normalized_distance,
                motion_amount
            )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            # Completely reset interaction
            state = "IDLE"

            arm_frames = 0
            press_frames = 0
            release_frames = 0

            locked_key = None

            previous_tip = None
            previous_tip_z = None
            previous_time = None

            tip_motion_history.clear()

            # Allow smoothing to rebuild when hand returns
            smoothed_points = None

            draw_status(
                frame,
                "IDLE",
                None,
                180.0,
                9.0,
                0.0
            )

        # ====================================================
        # UI
        # ====================================================

        draw_text_field(
            frame
        )

        draw_keyboard(
            frame,
            keys,
            hovered_key
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "POINT = hover    PUSH/CURL = type",
            (20, height - 229),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Release finger before next key",
            (20, height - 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (185, 185, 185),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        cv2.imshow(
            "Hand Tracking Virtual Keyboard",
            frame
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
        ) == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()