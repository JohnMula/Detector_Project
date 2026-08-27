import cv2
import mediapipe as mp
import urllib.request
import os
import math
import time

# ============================================================
# MEDIA PIPE MODEL
# ============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

mp = mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# HAND LANDMARK CONNECTIONS
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
# KEYBOARD
# ============================================================

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
# GLOBAL STATE
# ============================================================

typed_text = ""

pressed_key = None
pressed_key_until = 0

# Prevent repeated typing while the finger stays curled
click_armed = True

# Number of frames finger must stay curled
CURL_CONFIRM_FRAMES = 3
curl_frames = 0

# Small delay after a click
CLICK_COOLDOWN = 0.35
last_click_time = 0


# ============================================================
# MATH HELPERS
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def angle(a, b, c):
    """
    Angle ABC in degrees.
    """

    ba = (
        a[0] - b[0],
        a[1] - b[1]
    )

    bc = (
        c[0] - b[0],
        c[1] - b[1]
    )

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 180

    cos_value = dot / (magnitude_ba * magnitude_bc)

    cos_value = max(-1, min(1, cos_value))

    return math.degrees(
        math.acos(cos_value)
    )


# ============================================================
# DETECT "TYPING" / CURLING INDEX FINGER
# ============================================================

def index_finger_curled(points):
    """
    Determines whether the index finger is bent enough
    to count as a click.

    Index landmarks:

    5 = MCP
    6 = PIP
    7 = DIP
    8 = TIP
    """

    wrist = points[0]
    mcp = points[5]
    pip = points[6]
    dip = points[7]
    tip = points[8]

    # Angle at PIP.
    # Straight finger ~= 170-180 degrees
    # Bent finger becomes significantly smaller.
    pip_angle = angle(
        mcp,
        pip,
        dip
    )

    # Distance from fingertip to MCP.
    # A bent finger becomes shorter.
    mcp_tip_distance = distance(
        mcp,
        tip
    )

    # Overall hand scale.
    hand_scale = distance(
        wrist,
        mcp
    )

    if hand_scale == 0:
        return False

    normalized_distance = (
        mcp_tip_distance / hand_scale
    )

    # Main curl test
    curled_by_angle = pip_angle < 145

    curled_by_distance = (
        normalized_distance < 2.0
    )

    return (
        curled_by_angle and
        curled_by_distance
    )


# ============================================================
# GET FINGERTIP POSITION
# ============================================================

def get_index_tip(points):
    return points[8]


# ============================================================
# KEYBOARD LAYOUT
# ============================================================

def build_keyboard(frame_width, frame_height):

    keys = []

    key_w = 55
    key_h = 48

    gap = 8

    keyboard_y = frame_height - 215

    # --------------------------------------------
    # Letter rows
    # --------------------------------------------

    for row_index, row in enumerate(KEYBOARD_ROWS):

        total_width = (
            len(row) * key_w +
            (len(row) - 1) * gap
        )

        start_x = (
            frame_width - total_width
        ) // 2

        y = keyboard_y + row_index * (
            key_h + gap
        )

        for col_index, letter in enumerate(row):

            x = (
                start_x +
                col_index * (key_w + gap)
            )

            keys.append({
                "key": letter,
                "x1": x,
                "y1": y,
                "x2": x + key_w,
                "y2": y + key_h
            })

    # --------------------------------------------
    # Bottom row
    # --------------------------------------------

    bottom_y = keyboard_y + 3 * (
        key_h + gap
    )

    # Backspace
    back_w = 85

    keys.append({
        "key": "BACK",
        "x1": 70,
        "y1": bottom_y,
        "x2": 70 + back_w,
        "y2": bottom_y + key_h
    })

    # Space
    space_w = 260

    space_x = (
        frame_width - space_w
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
        "x1": frame_width - 70 - enter_w,
        "y1": bottom_y,
        "x2": frame_width - 70,
        "y2": bottom_y + key_h
    })

    return keys


# ============================================================
# FIND KEY UNDER FINGER
# ============================================================

def get_key_at_position(x, y, keys):

    for key in keys:

        if (
            key["x1"] <= x <= key["x2"]
            and
            key["y1"] <= y <= key["y2"]
        ):
            return key

    return None


# ============================================================
# DRAW KEYBOARD
# ============================================================

def draw_keyboard(frame, keys, hovered_key=None):

    for key in keys:

        x1 = key["x1"]
        y1 = key["y1"]
        x2 = key["x2"]
        y2 = key["y2"]

        is_pressed = (
            pressed_key == key["key"]
            and
            time.time() < pressed_key_until
        )

        # ----------------------------------------
        # Key colors
        # ----------------------------------------

        if is_pressed:
            fill = (255, 255, 255)
            text_color = (30, 30, 30)

        elif hovered_key == key["key"]:
            # Hover only shows the target.
            # It does NOT type.
            fill = (80, 80, 80)
            text_color = (255, 255, 255)

        else:
            fill = (35, 35, 35)
            text_color = (255, 255, 255)

        # ----------------------------------------
        # Key body
        # ----------------------------------------

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
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------
        # Key label
        # ----------------------------------------

        label = key["key"]

        if label == "SPACE":
            label = "SPACE"

        elif label == "BACK":
            label = "←"

        elif label == "ENTER":
            label = "↵"

        font_scale = 0.75

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2
        )[0]

        text_x = (
            x1 +
            (x2 - x1 - text_size[0]) // 2
        )

        text_y = (
            y1 +
            (y2 - y1 + text_size[1]) // 2
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA
        )


# ============================================================
# HANDLE KEY PRESS
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

    h, w, _ = frame.shape

    x1 = 60
    x2 = w - 60

    y1 = 25
    y2 = 115

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

    # Display text
    display_text = typed_text[-45:]

    cv2.putText(
        frame,
        display_text,
        (x1 + 18, y1 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    # Cursor
    cursor_x = (
        x1 +
        18 +
        cv2.getTextSize(
            display_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            2
        )[0][0]
    )

    cv2.line(
        frame,
        (cursor_x, y1 + 20),
        (cursor_x, y2 - 15),
        (30, 30, 30),
        2
    )


# ============================================================
# MEDIA PIPE OPTIONS
# ============================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5,
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print()
print("============================================")
print("   HAND TRACKING VIRTUAL KEYBOARD")
print("============================================")
print()
print("Point your index finger at a key.")
print()
print("Straight finger = aim")
print("Bent finger     = CLICK")
print()
print("Press Q to quit.")
print()
print("============================================")


timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Build keyboard
        keys = build_keyboard(
            width,
            height
        )

        # ----------------------------------------
        # MediaPipe image
        # ----------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        hovered_key = None
        fingertip_position = None
        finger_curled = False

        # ----------------------------------------
        # Process hands
        # ----------------------------------------

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                points = []

                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )

                    points.append(
                        (x, y)
                    )

                # --------------------------------
                # Draw skeleton
                # --------------------------------

                for start, end in CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # --------------------------------
                # Draw joints
                # --------------------------------

                for i, (x, y) in enumerate(points):

                    radius = (
                        7 if i == 0
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

                # --------------------------------
                # Index finger
                # --------------------------------

                fingertip_position = get_index_tip(
                    points
                )

                fingertip_x, fingertip_y = (
                    fingertip_position
                )

                hovered = get_key_at_position(
                    fingertip_x,
                    fingertip_y,
                    keys
                )

                if hovered:

                    hovered_key = hovered["key"]

                # --------------------------------
                # Check curl position
                # --------------------------------

                finger_curled = index_finger_curled(
                    points
                )

                # --------------------------------
                # Cursor / fingertip
                # --------------------------------

                cv2.circle(
                    frame,
                    (fingertip_x, fingertip_y),
                    11,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # --------------------------------
                # Curl status
                # --------------------------------

                status = (
                    "CLICK POSITION"
                    if finger_curled
                    else "AIM"
                )

                status_color = (
                    (0, 255, 0)
                    if finger_curled
                    else (255, 255, 255)
                )

                cv2.putText(
                    frame,
                    status,
                    (25, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA
                )

                # --------------------------------
                # Click detection
                # --------------------------------

                if finger_curled:

                    curl_frames += 1

                else:

                    curl_frames = 0

                    # Release re-arms the click
                    click_armed = True

                now = time.time()

                if (
                    finger_curled
                    and
                    curl_frames >= CURL_CONFIRM_FRAMES
                    and
                    hovered_key is not None
                    and
                    click_armed
                    and
                    (now - last_click_time)
                    > CLICK_COOLDOWN
                ):

                    type_key(
                        hovered_key
                    )

                    pressed_key = hovered_key

                    pressed_key_until = (
                        now + 0.15
                    )

                    last_click_time = now

                    # Prevent repeated clicks
                    click_armed = False

                break

        else:

            curl_frames = 0
            click_armed = True

        # ====================================================
        # DRAW UI
        # ====================================================

        draw_text_field(frame)

        draw_keyboard(
            frame,
            keys,
            hovered_key
        )

        # ----------------------------------------
        # Instructions
        # ----------------------------------------

        cv2.putText(
            frame,
            "Point + curl index finger to type",
            (25, height - 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Hand Tracking Virtual Keyboard",
            frame
        )

        # ----------------------------------------
        # Quit
        # ----------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()