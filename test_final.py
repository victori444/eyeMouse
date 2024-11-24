import cv2
import pyautogui
import mediapipe as mp
import time
import math
import pygame
from collections import deque
import speech_recognition as sr
import numpy as np

# Initialize MediaPipe for face mesh (for cursor control)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize MediaPipe for hand tracking (for virtual keyboard)T
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Initialize OpenCV to capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for sound playback
pygame.mixer.init()  # Initialize the mixer
sound_effect_click = pygame.mixer.Sound("key_press.wav")
sound_effect_voice = pygame.mixer.Sound("bell_and_water.wav")
sound_effect_drag = pygame.mixer.Sound("pluck.mp3")
sound_effect_mouse = pygame.mixer.Sound("mouse.mp3")
sound_effect_voiceoff = pygame.mixer.Sound("voiceoff.wav")
sound_effect_act = pygame.mixer.Sound("act.mp3")
sound_effect_des = pygame.mixer.Sound("des.mp3")



# Wait for camera to initialize
time.sleep(2)




# Initialize a flag for drag state
drag_enabled = False
last_eyebrow_raised = False  # Tracks the previous eyebrow state
last_eyebrow_toggle_time = 0  # To debounce eyebrow raises
TOGGLE_DEBOUNCE_TIME = 0.5  # Debounce duration in seconds

# Define eyebrow and eye landmarks
LEFT_EYEBROW = [65]  # Updated top of left eyebrow
RIGHT_EYEBROW = [295]  # Updated top of right eyebrow
LEFT_EYE_TOP = [159]  # Verified top of left eye
RIGHT_EYE_TOP = [386]  # Verified top of right eye

# Threshold for eyebrow raise detection
EYEBROW_RAISE_THRESHOLD = 0.05  # Adjust for sensitivity


# Special keys and their corresponding actions

SPECIAL_KEYS = {

    'Space': 'space',  # Space bar

    'Enter': 'enter',  # Enter key

    'Backspace': 'backspace',  # Backspace key

    'Shift': 'shift'  # Shift key

}

def detect_eyebrow_raise(face_landmarks):
    """
    Detects if eyebrows are raised based on the distance between eyebrows and eyes.
    """
    def calculate_distance(eyebrow_index, eye_index):
        return abs(face_landmarks[eyebrow_index].y - face_landmarks[eye_index].y)

    left_distance = calculate_distance(LEFT_EYEBROW[0], LEFT_EYE_TOP[0])
    right_distance = calculate_distance(RIGHT_EYEBROW[0], RIGHT_EYE_TOP[0])

    return left_distance > EYEBROW_RAISE_THRESHOLD or right_distance > EYEBROW_RAISE_THRESHOLD

def detect_eyebrow_toggle(face_landmarks):
    """
    Detects and toggles drag state when eyebrows are raised, ensuring no repeated toggles.
    """
    global drag_enabled, last_eyebrow_raised, last_eyebrow_toggle_time

    # Detect if eyebrows are currently raised
    eyebrow_raised = detect_eyebrow_raise(face_landmarks)

    if eyebrow_raised and not last_eyebrow_raised:  # Transition: Not Raised -> Raised
        current_time = time.time()
        if current_time - last_eyebrow_toggle_time > TOGGLE_DEBOUNCE_TIME:  # Debounce logic
            drag_enabled = not drag_enabled  # Toggle the drag state
            last_eyebrow_toggle_time = current_time
            print("Drag toggled:", "Enabled" if drag_enabled else "Disabled")
            
            # Play appropriate sound based on drag state
            if drag_enabled:
                sound_effect_act.play()  # Play sound for enabling
            else:
                sound_effect_des.play()  # Play sound for disabling

    # Update the last eyebrow state
    last_eyebrow_raised = eyebrow_raised





# Keyboard layout and position (with symbols and shift rows)
KEYS = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\'', 'Enter'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'Shift'],
    ['Space']
]

# Reduced key width, height, and margin for smaller keyboard
KEY_WIDTH = 40  # Width of each key (smaller size)
KEY_HEIGHT = 40  # Height of each key (smaller size)Galaxy
KEY_MARGIN = 4  # Reduced margin between keys
KEY_POS = (20, 50)  # Top-left position of the keyboard (shifted slightly to fit)
# Eye landmarks based on MediaPipe documentationLKKSpace
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

            # Change key color when hovered over,KShiftShift
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

# for wink to last longer for it to click
# Variables to track wink duration
wink_start_time = None  # Timestamp for when a wink starts
wink_threshold_duration = 0.3  # Duration (in seconds) required for a wink to execute a click

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    """Detects winks based on the eye aspect ratio (EAR)."""
    def eye_aspect_ratio(eye_indices):
        """Calculates the Eye Aspect Ratio (EAR) for an eye."""
        # Calculate vertical distances
        vertical1 = ((face_landmarks[eye_indices[1]].x - face_landmarks[eye_indices[5]].x) ** 2 +
                     (face_landmarks[eye_indices[1]].y - face_landmarks[eye_indices[5]].y) ** 2) ** 0.5
        vertical2 = ((face_landmarks[eye_indices[2]].x - face_landmarks[eye_indices[4]].x) ** 2 +
                     (face_landmarks[eye_indices[2]].y - face_landmarks[eye_indices[4]].y) ** 2) ** 0.5
        # Calculate horizontal distanceLLOOBackspace
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

def detect_head_tilt(face_landmarks):
    """
    Detects head tilts based on the nose position.
    """
    NOSE_TIP = 1  # Nose tip
    NOSE_BASE = 168  # Nose base (central)

    # Get the X-coordinates of the nose tip and nose base
    nose_tip_x = face_landmarks[NOSE_TIP].x
    nose_base_x = face_landmarks[NOSE_BASE].x

    # Calculate the horizontal displacement of the nose tip relative to the base
    tilt = nose_tip_x - nose_base_x

    # Define thresholds for tilting
    TILT_THRESHOLD = 0.02  # Adjust for sensitivity
    if tilt > TILT_THRESHOLD:  # Head tilted to the right
        return "up"
    elif tilt < -TILT_THRESHOLD:  # Head tilted to the left
        return "down"
    return None

def detect_mouth_opening(face_landmarks, mouth_indices):

    """

    Detects if the mouth is open by measuring the distance between the top and bottom lips.

    """

    vertical_dist = ((face_landmarks[mouth_indices[2]].x - face_landmarks[mouth_indices[0]].x) ** 2 +

                     (face_landmarks[mouth_indices[2]].y - face_landmarks[mouth_indices[0]].y) ** 2) ** 0.5

    return vertical_dist

def voice_typing():

    """

    Captures voice input and types it using pyautogui.

    """

    with sr.Microphone() as source:

        print("Listening for voice input...")
        sound_effect_voice.play()

        try:

            audio = recognizer.listen(source, timeout=5)

            text = recognizer.recognize_google(audio)

            print(f"Recognized: {text}")

            pyautogui.typewrite(text)  # Type the recognized text

        except sr.WaitTimeoutError:

            print("No speech detected within the timeout.")
            sound_effect_voiceoff.play()

        except sr.UnknownValueError:

            print("Speech was not clear.")
            sound_effect_voiceoff.play()

        except sr.RequestError as e:

            print(f"Could not request results; {e}")

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





def track_face_cursor(face_landmarks, frame_width, frame_height):
    """
    Track the cursor using the nose tip from FaceMesh and map it to the entire screen.
    Applies anatomical offsets to keep eyes in frame while reaching screen edges.
    """
    NOSE_TIP = 1  # Nose tip (index 1 in MediaPipe)
    nose = face_landmarks[NOSE_TIP]
    
    # Get screen width and height
    screen_width, screen_height = pyautogui.size()
    
    # Normalize the position based on the video frame size
    normalized_x = nose.x
    normalized_y = nose.y
    
    # Define a "safe zone" with anatomical offsets
    # Vertical safe zone is shifted down to account for eyes being above nose
    SAFE_ZONE_X_MIN = 0.3
    SAFE_ZONE_X_MAX = 0.7
    SAFE_ZONE_Y_MIN = 0.4  # Shifted down to account for eyes
    SAFE_ZONE_Y_MAX = 0.8  # Shifted down to account for eyes
    
    # Map the safe zone to the full screen using a piecewise linear function
    def map_coordinate(value, safe_min, safe_max):
        if value < safe_min:
            # Map [0, safe_min] to [0, 0.3] of screen
            return (value / safe_min) * 0.3
        elif value > safe_max:
            # Map [safe_max, 1] to [0.7, 1] of screen
            return 0.7 + ((value - safe_max) / (1 - safe_max)) * 0.3
        else:
            # Map [safe_min, safe_max] to [0.3, 0.7] of screen
            return 0.3 + ((value - safe_min) / (safe_max - safe_min)) * 0.4
    
    # Apply the mapping to both coordinates
    mapped_x = map_coordinate(normalized_x, SAFE_ZONE_X_MIN, SAFE_ZONE_X_MAX)
    
    # Apply offset mapping for vertical coordinate
    # This means the nose needs to be lower in the frame to reach top edges
    mapped_y = map_coordinate(normalized_y, SAFE_ZONE_Y_MIN, SAFE_ZONE_Y_MAX)
    
    # Additional vertical offset to account for eyes above nose
    # This effectively shifts the entire vertical range down
    VERTICAL_OFFSET = 0.15  # Adjust this value based on testing
    mapped_y = mapped_y - VERTICAL_OFFSET
    
    # Convert to screen coordinates
    screen_x = int(mapped_x * screen_width)
    screen_y = int(mapped_y * screen_height)
    
    # Ensure cursor stays within screen bounds
    screen_x = min(max(screen_x, 0), screen_width - 1)
    screen_y = min(max(screen_y, 0), screen_height - 1)
    
    return screen_x, screen_y



'''
def detect_head_shake(face_landmarks):
    """Detects a head shake based on the nose movement."""
    NOSE_TIP = 1  # Nose tip (index 1 in MediaPipe)
    NOSE_BASE = 168  # Nose base (central)

    nose_tip_x = face_landmarks[NOSE_TIP].x
    nose_base_x = face_landmarks[NOSE_BASE].x

    shake_threshold = 0.1  # Threshold for detecting a significant shake

    # Detect head shake by checking the horizontal position of the nose
    if abs(nose_tip_x - nose_base_x) > shake_threshold:
        return True
    return False

def detect_thumb_gesture(hand_landmarks):
    """Detects a thumbs up or thumbs down gesture."""
    thumb_tip = hand_landmarks[4]  # Thumb tip
    thumb_ip = hand_landmarks[3]   # Thumb IP joint

    # Calculate distance between thumb tip and thumb IP joint
    distance_thumb = math.sqrt((thumb_tip.x - thumb_ip.x) ** 2 + (thumb_tip.y - thumb_ip.y) ** 2)
    THUMB_THRESHOLD = 0.05  # Distance threshold for thumb gesture

    # Check if the thumb is pointing upwards or downwards based on its position
    if distance_thumb < THUMB_THRESHOLD:
        return "thumbs_down"
    return None
'''






# Add a variable to control if a click has happened or not
last_click_time = time.time()

# Add flags to track when termination gestures are detected
# terminate_program = False

# Add a flag to control keyboard visibility and functionality
# keyboard_active = False


MOUTH = [13, 312, 14, 317]  # Top and bottom points of the mouth for opening detection



# Define a threshold for mouth opening detection

MOUTH_OPEN_THRESHOLD = 0.05  # Adjust this threshold as needed


# Add these constants for window properties
WINDOW_NAME_KEYBOARD = 'Virtual Keyboard Control'
WINDOW_NAME_FACE = 'Facial Control'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Create named windows and set them to stay on top
cv2.namedWindow(WINDOW_NAME_KEYBOARD, cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_NAME_FACE, cv2.WINDOW_NORMAL)

# Set windows to stay on top and not close when clicked
cv2.setWindowProperty(WINDOW_NAME_KEYBOARD, cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty(WINDOW_NAME_FACE, cv2.WND_PROP_TOPMOST, 1)

# Set the window sizes
cv2.resizeWindow(WINDOW_NAME_KEYBOARD, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow(WINDOW_NAME_FACE, WINDOW_WIDTH, WINDOW_HEIGHT)

# Initialize keyboard visibility state
keyboard_visible = False
hand_detection_frames = 0  # Counter for consecutive frames with hand detection
HAND_DETECTION_THRESHOLD = 5  # Number of frames needed to show/hide keyboard










# main loop
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

    # Hand detection logic with smoothing
    if results_hands.multi_hand_landmarks:
        hand_detection_frames += 1
        if hand_detection_frames >= HAND_DETECTION_THRESHOLD and not keyboard_visible:
            if not keyboard_visible:
                keyboard_visible = True
                cv2.namedWindow(WINDOW_NAME_KEYBOARD, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME_KEYBOARD, cv2.WND_PROP_TOPMOST, 1)
    else:
        hand_detection_frames -= 1
        if hand_detection_frames <= 0:
            if keyboard_visible:
                keyboard_visible = False
                cv2.destroyWindow(WINDOW_NAME_KEYBOARD)
        hand_detection_frames = max(0, hand_detection_frames)

    # Create copies for drawing
    keyboard_frame = mirrored_frame.copy()
    face_frame = frame.copy()

    # Only draw keyboard if visible
    if keyboard_visible:
        draw_virtual_keyboard(keyboard_frame)
        cv2.imshow(WINDOW_NAME_KEYBOARD, keyboard_frame)
    
    # Always show the face tracking window
    cv2.imshow(WINDOW_NAME_FACE, face_frame)


    # Detect and track face landmarks (cursor control via nose tip)
    if results_face.multi_face_landmarks:
        for landmarks in results_face.multi_face_landmarks:
            face_landmarks = landmarks.landmark

            # Detect winks (eye aspect ratio) for clicking
            wink = detect_wink(face_landmarks, LEFT_EYE, RIGHT_EYE)
            if wink == "left" or wink == "right":
                if wink_start_time is None:
                    wink_start_time = time.time()  # Start tracking wink duration
                elif time.time() - wink_start_time >= wink_threshold_duration:
                    pyautogui.click()  # Perform a mouse click
                    sound_effect_mouse.play()
                    wink_start_time = None  # Reset the wink timer after a successful click
            else:
                wink_start_time = None  # Reset if no wink is detected

            # Detect mouth opening
            mouth_opening = detect_mouth_opening(face_landmarks, MOUTH)
            if mouth_opening > MOUTH_OPEN_THRESHOLD:
                voice_typing()  # Activate voice typing when mouth is open

            # Detect head tilts for scrolling
            scroll_direction = detect_head_tilt(face_landmarks)
            if scroll_direction == "up":
                pyautogui.scroll(50)  # Scroll up
                time.sleep(0.1)
            elif scroll_direction == "down":
                pyautogui.scroll(-50)  # Scroll down
                time.sleep(0.1)

            # Detect and toggle drag state based on eyebrow raise
            detect_eyebrow_toggle(face_landmarks)

            # Perform drag action if drag is enabled
            if drag_enabled:
                # sound_effect_drag.play()
                # Move the cursor with the nose tip
                cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
                pyautogui.mouseDown()
                pyautogui.moveTo(cursor_x, cursor_y)
            else:
                pyautogui.mouseUp()




            

            '''

            # Detect head shakes for program termination
            if detect_head_shake(face_landmarks):
                print("Head shake detected! Terminating program.")
                terminate_program = True
                break  # Exit the loop if head shake detected

            # Detect hand gestures for program termination (e.g., thumbs down)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    gesture = detect_thumb_gesture(hand_landmarks.landmark)
                    if gesture == "thumbs_down":
                        print("Thumbs down detected! Terminating program.")
                        terminate_program = True
                        break  # Exit the loop if thumbs down detected
                        
            '''

            # Move the cursor with the nose tip (for face tracking)
            cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
            pyautogui.moveTo(cursor_x, cursor_y)

            # Only detect pinching if the keyboard is active
            if keyboard_visible and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    hand_landmarks = hand_landmarks.landmark
                    pinch = detect_hand_gesture(hand_landmarks)
                    if pinch == "pinch":
                        index_tip = hand_landmarks[8]
                        screen_x = int(index_tip.x * mirrored_frame.shape[1])
                        screen_y = int(index_tip.y * mirrored_frame.shape[0])

                        key = get_key_at_position(screen_x, screen_y)
                        if key:
                            # Only trigger key press if at least 0.5 seconds have passed since last click
                            current_time = time.time()
                            if current_time - last_click_time > 0.5:
                                print(f"Clicked key: {key}")
                                
                                if key in SPECIAL_KEYS:

                                    pyautogui.press(SPECIAL_KEYS[key])

                                else:

                                    pyautogui.write(key)  # Regular key typing
                                sound_effect_click.play()  # Play sound when key is pressed
                                last_click_time = current_time  # Update last click time

    # Display the frame with the virtual keyboard
    # cv2.imshow('Virtual Keyboard Control', original_frame)
    cv2.imshow('Facial Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminating program...")
        break

cap.release()
cv2.destroyAllWindows()
