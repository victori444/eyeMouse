import cv2
import pyautogui
import mediapipe as mp
import time
import math

# Initialize MediaPipe for face mesh (for cursor control)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize MediaPipe for hand tracking (for virtual keyboard)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize OpenCV to capture video from the webcam
cap = cv2.VideoCapture(0)

# Wait for camera to initialize
time.sleep(2)

# Keyboard layout and position (with symbols and shift rows)
KEYS = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\'', 'Enter'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'Shift'],
    ['Space']
]

# Reduced key width, height, and margin for smaller keyboard
KEY_WIDTH = 80  # Width of each key (smaller size)
KEY_HEIGHT = 80  # Height of each key (smaller size)
KEY_MARGIN = 8  # Reduced margin between keys
KEY_POS = (20, 50)  # Top-left position of the keyboard (shifted slightly to fit)

# Eye landmarks based on MediaPipe documentation
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Right eye landmarks

# Initialize caps lock state
caps_lock = False

def draw_virtual_keyboard(frame, hovered_key=None):
    """Draws a visually appealing virtual keyboard on the video frame with functionality."""
    overlay = frame.copy()  # Create a copy of the frame for blending
    alpha = 0.6  # Transparency level (0.0 to 1.0)

    y_offset = KEY_POS[1]
    space_rect = None  # To store the space bar's rectangle coordinates for click detection
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
            # Handle special keys (like space, backspace, enter, shift)
            if key == 'Space':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN 
                key_text = "Space"  # Space key prints a space, not the word "Space"
                space_rect = (x_offset, y_offset, key_width, KEY_HEIGHT)  # Define space bar rectangle
            elif key == 'Backspace':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  # Backspace is twice the width
                key_text = "Delete"  # Arrow symbol for Backspace
            elif key == 'Enter':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  # Enter is twice the width
                key_text = "Enter"
            elif key == 'Shift':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  # Shift is twice the width
                key_text = "Shift"
            else:
                key_width = KEY_WIDTH  # Regular key width
                key_text = key.upper() if caps_lock else key.lower()

            # Change key color when hovered over
            if hovered_key == key:
                color = (100, 255, 100)  # Green for hovered key
            else:
                color = (200, 200, 255)  # Light purple background

            # Draw the key background on the overlay
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), color, -1)  # Fill the key

            # Draw the key border (fully opaque)
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), (100, 100, 150), 2, cv2.LINE_AA)

            # Add the text for each key
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_offset + (key_width - text_size[0]) // 2
            text_y = y_offset + (KEY_HEIGHT + text_size[1]) // 2
            cv2.putText(overlay, key_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 100), font_thickness, cv2.LINE_AA)

            # Increment x_offset for the next key
            x_offset += key_width + KEY_MARGIN
        
        # Increment y_offset for the next row
        y_offset += KEY_HEIGHT + KEY_MARGIN

    # Blend the overlay with the original frame (for transparency effect)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return space_rect

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    """Detects winks based on the eye aspect ratio (EAR)."""
    def eye_aspect_ratio(eye_indices):
        """Calculates the Eye Aspect Ratio (EAR) for an eye."""
        # Calculate vertical distances
        vertical1 = ((face_landmarks[eye_indices[1]].x - face_landmarks[eye_indices[5]].x) ** 2 +
                     (face_landmarks[eye_indices[1]].y - face_landmarks[eye_indices[5]].y) ** 2) ** 0.5
        vertical2 = ((face_landmarks[eye_indices[2]].x - face_landmarks[eye_indices[4]].x) ** 2 +
                     (face_landmarks[eye_indices[2]].y - face_landmarks[eye_indices[4]].y) ** 2) ** 0.5
        # Calculate horizontal distance
        horizontal = ((face_landmarks[eye_indices[0]].x - face_landmarks[eye_indices[3]].x) ** 2 +
                      (face_landmarks[eye_indices[0]].y - face_landmarks[eye_indices[3]].y) ** 2) ** 0.5
        # EAR formula
        return (vertical1 + vertical2) / (2.0 * horizontal)

    EAR_THRESHOLD = 0.2  # Threshold to detect a wink (adjust if needed)
    left_ear = eye_aspect_ratio(left_eye_indices)
    right_ear = eye_aspect_ratio(right_eye_indices)

    if left_ear < EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
        return "left"  # Left eye wink
    elif right_ear < EAR_THRESHOLD and left_ear > EAR_THRESHOLD:
        return "right"  # Right eye wink
    return None

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

def track_face_cursor(face_landmarks, frame_width, frame_height):
    """Track the cursor using the nose tip from FaceMesh."""
    NOSE_TIP = 1  # Nose tip (index 1 in MediaPipe)
    nose = face_landmarks[NOSE_TIP]
    screen_x = int(nose.x * frame_width)  # Scale to screen width
    screen_y = int(nose.y * frame_height)  # Scale to screen height
    return screen_x, screen_y

# Add a variable to control if a click has happened or not
last_click_time = time.time()

# Add a flag to control keyboard visibility and functionality
keyboard_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the video frame horizontally (this flips the video feed)
    mirrored_frame = cv2.flip(frame, 1)

    # Convert the mirrored frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks (for cursor control)
    results_face = face_mesh.process(rgb_frame)

    # Process the frame for hand landmarks (for virtual keyboard)
    results_hands = hands.process(rgb_frame)

    # Detect hand landmarks (if hand is visible in the frame)
    hand_visible = False
    if results_hands.multi_hand_landmarks:
        hand_visible = True  # Hand is visible in the frame

    # Toggle the keyboard visibility based on hand presence
    if hand_visible:
        keyboard_active = True  # Show the keyboard
    else:
        keyboard_active = False  # Hide the keyboard

    # Draw the virtual keyboard on the mirrored frame if active
    original_frame = mirrored_frame.copy()  # Now we draw on the mirrored frame
    if keyboard_active:
        draw_virtual_keyboard(original_frame)

    # Detect and track face landmarks (cursor control via nose tip)
    if results_face.multi_face_landmarks:
        for landmarks in results_face.multi_face_landmarks:
            face_landmarks = landmarks.landmark

            # Detect winks (eye aspect ratio) for clicking
            wink = detect_wink(face_landmarks, LEFT_EYE, RIGHT_EYE)
            if wink == "left" or wink == "right":
                pyautogui.click()  # Perform a mouse click

            # Move the cursor with the nose tip (for face tracking)
            cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
            pyautogui.moveTo(cursor_x, cursor_y)

            # Only detect pinching if the keyboard is active
            if keyboard_active and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    hand_landmarks = hand_landmarks.landmark
                    pinch = detect_hand_gesture(hand_landmarks)
                    if pinch == "pinch":
                        index_tip = hand_landmarks[8]
                        screen_x = int(index_tip.x * mirrored_frame.shape[1])
                        screen_y = int(index_tip.y * mirrored_frame.shape[0])

                        key = get_key_at_position(screen_x, screen_y)
                        if key:
                            # Handle special keys (Shift, Space, Enter, Backspace)
                            if key == 'Shift':
                                caps_lock = not caps_lock  # Toggle caps lock state
                            elif key == 'Space':
                                pyautogui.write(' ')  # Output a space character
                            elif key == 'Enter':
                                pyautogui.press('enter')  # Simulate pressing "Enter"
                            elif key == 'Backspace':
                                pyautogui.press('backspace')  # Simulate pressing "Backspace"
                            # Simulate typing the key if at least 0.5 seconds have passed
                            current_time = time.time()
                            if current_time - last_click_time > 0.5:
                                if key not in ['Shift', 'Space', 'Enter', 'Backspace']:  # Don't print special keys
                                    pyautogui.write(key.upper() if caps_lock else key.lower())
                                last_click_time = current_time  # Update last click time

    # Display the frame with the virtual keyboard
    cv2.imshow('Virtual Keyboard Control', original_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
