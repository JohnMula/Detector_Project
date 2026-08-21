from ultralytics import YOLO
import cv2

# Load the YOLO26 nano model (smallest/fastest — good for a first test)
# This will auto-download the weights on first run
model = YOLO("yolo26n.pt")

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO detection on the current frame
    results = model(frame, verbose=False)

    # results[0].plot() draws the bounding boxes + labels on the frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO26 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()