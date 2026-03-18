import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize mediapipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Create output folder
os.makedirs("data", exist_ok=True)

gesture_name = input("Enter gesture name (e.g., thumbs_up, heart, peace): ").strip()
file_path = os.path.join("data", f"{gesture_name}.csv")

cap = cv2.VideoCapture(0)
print("Press 's' to start recording, 'q' to quit...")

record = False
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if record:
                data.append(landmarks)

    cv2.imshow("Capture Gesture - Press 's' to Start/Stop, 'q' to Quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        record = not record
        print("Recording..." if record else "Stopped.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
if data:
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print(f"Saved {len(data)} samples to {file_path}")
