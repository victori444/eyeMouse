import cv2
import pyautogui
import mediapipe as mp
import time
import math

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize OpenCV to capture video from the webcam
cap = cv2.VideoCapture(0)

# Wait for camera to initialize
time.sleep(2)

# Keyboard layout and position (simplified)
KEYS = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

KEY_WIDTH = 100  # Width of each key
KEY_HEIGHT = 100  # Height of each key
KEY_MARGIN = 10  # Margin between keys
KEY_POS = (50, 50)  # Top-left position of the keyboard

def draw_virtual_keyboard(frame):
    """Draws a virtual keyboard on the video frame."""
    y_offset = KEY_POS[1]
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
            # Draw a rectangle for each key
            cv2.rectangle(frame, (x_offset, y_offset), (x_offset + KEY_WIDTH, y_offset + KEY_HEIGHT), (255, 255, 255), -1)
            # Write the key's label on the rectangle
            cv2.putText(frame, key, (x_offset + 30, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            x_offset += KEY_WIDTH + KEY_MARGIN
        y_offset += KEY_HEIGHT + KEY_MARGIN

def detect_hand_gesture(hand_landmarks):
    """Detects a pinch gesture between thumb and index fingers."""
    thumb_tip = hand_landmarks[4]  # Thumb tip
    index_tip = hand_landmarks[8]  # Index finger tip

    # Calculate distance between thumb and index finger
    distance_thumb_index = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    PINCH_THRESHOLD = 0.05  # Distance threshold for pinch

    if distance_thumb_index < PINCH_THRESHOLD:
        return "pinch"  # Pinch gesture detected
    return None

def get_key_at_position(x, y):
    """Determine which key is at the given screen position."""
    y_offset = KEY_POS[1]
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
            if x_offset <= x <= x_offset + KEY_WIDTH and y_offset <= y <= y_offset + KEY_HEIGHT:
                return key
            x_offset += KEY_WIDTH + KEY_MARGIN
        y_offset += KEY_HEIGHT + KEY_MARGIN
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results_hands = hands.process(rgb_frame)

    # Draw the virtual keyboard on the frame
    draw_virtual_keyboard(frame)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            hand_landmarks = hand_landmarks.landmark

            # Detect hand gesture (pinch)
            gesture = detect_hand_gesture(hand_landmarks)
            if gesture == "pinch":
                # Get the position of the hand's index finger tip
                index_tip = hand_landmarks[8]
                screen_x = int(index_tip.x * frame.shape[1])  # Scale to screen width
                screen_y = int(index_tip.y * frame.shape[0])  # Scale to screen height

                # Determine which key is being pointed at
                key = get_key_at_position(screen_x, screen_y)
                if key:
                    print(f"Clicked key: {key}")
                    pyautogui.write(key)  # Simulate typing the key
                    time.sleep(0.5)  # Delay to avoid multiple key presses

            # Optionally, draw the hand's position (e.g., index tip) on the frame
            index_tip = hand_landmarks[8]
            screen_x = int(index_tip.x * frame.shape[1])
            screen_y = int(index_tip.y * frame.shape[0])
            cv2.circle(frame, (screen_x, screen_y), 10, (0, 0, 255), -1)

    # Display the frame with the virtual keyboard
    cv2.imshow('Virtual Keyboard', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()