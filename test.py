import cv2
import pyautogui
import mediapipe as mp
import time
import math
import pygame
from collections import deque
import speech_recognition as sr
import numpy as np

# Initialize MediaPipe for face mesh & hand tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize Speech Recognition & OpenCV
recognizer = sr.Recognizer()
cap = cv2.VideoCapture(0)

# Initialize pygame for sound playback
pygame.mixer.init()  # Initialize the mixer
sound_effect_click = pygame.mixer.Sound("key_press.wav")
sound_effect_voice = pygame.mixer.Sound("bell_and_water.wav")
sound_effect_drag = pygame.mixer.Sound("pluck.mp3")
sound_effect_mouse = pygame.mixer.Sound("mouse.mp3")

# Wait for camera to initialize
time.sleep(2)

############################################################
##### DRAG & DROP FUNCTIONALITY WITH EYEBROW MOVEMENT ######
############################################################

# Flag for drag stat
drag_enabled = False
last_eyebrow_toggle_time = 0
TOGGLE_DEBOUNCE_TIME = 0.5

# Eyebrow and eye landmarks
LEFT_EYEBROW = [65]  
RIGHT_EYEBROW = [295]  
LEFT_EYE_TOP = [159] 
RIGHT_EYE_TOP = [386]  
EYEBROW_RAISE_THRESHOLD = 0.05

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

    # Update the last eyebrow state
    last_eyebrow_raised = eyebrow_raised


############################################################
#####       VIRTUAL KEYBOARD WITH HAND TRACKING       ######
############################################################

KEYS = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\'', 'Enter'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'Shift'],
    ['Space']
]

SPECIAL_KEYS = {
    'Space': 'space',  # Space bar
    'Enter': 'enter',  # Enter key
    'Backspace': 'backspace',  # Backspace key
    'Shift': 'shift'  # Shift key
}

KEY_WIDTH = 40  
KEY_HEIGHT = 40  
KEY_MARGIN = 4
KEY_POS = (20, 50) 
# Eye landmarks based on MediaPipe documentationLKKSpace
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
caps_lock = False

def draw_virtual_keyboard(frame, hovered_key=None):
    """
    Draw virtual keyboard over video frame
    """
    overlay = frame.copy()  # copy of frame for blending
    alpha = 0.6  # transparency

    y_offset = KEY_POS[1]
    space_rect = None 
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
            if key == 'Space':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN 
                key_text = "Space"  
                space_rect = (x_offset, y_offset, key_width, KEY_HEIGHT)  
            elif key == 'Backspace':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  
                key_text = "Delete" 
            elif key == 'Enter':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  
                key_text = "Enter"
            elif key == 'Shift':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN  
                key_text = "Shift"
            else:
                key_width = KEY_WIDTH  
                key_text = key.upper() if caps_lock else key.lower()

            # background & border
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), (200, 200, 255), -1)
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), (100, 100, 150), 2, cv2.LINE_AA)

            # key text
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_offset + (key_width - text_size[0]) // 2
            text_y = y_offset + (KEY_HEIGHT + text_size[1]) // 2
            cv2.putText(overlay, key_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 100), font_thickness, cv2.LINE_AA)

            x_offset += key_width + KEY_MARGIN # offset between keys horizontally
        
        # offset for next row
        y_offset += KEY_HEIGHT + KEY_MARGIN

    # transparency by blending overlay w original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return space_rect

def get_key_at_position(x, y):
    """
    Determine which key is at the given screen position
    """
    y_offset = KEY_POS[1]
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
            if x_offset <= x <= x_offset + KEY_WIDTH and y_offset <= y <= y_offset + KEY_HEIGHT:
                return key
            x_offset += KEY_WIDTH + KEY_MARGIN
        y_offset += KEY_HEIGHT + KEY_MARGIN
    return None

def detect_hand_gesture(hand_landmarks):
    """
    Detects pinch gesture to select key 
    """
    thumb_tip = hand_landmarks[4] 
    index_tip = hand_landmarks[8]  

    # distance between thumb and index finger
    distance_thumb_index = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    PINCH_THRESHOLD = 0.05 

    if distance_thumb_index < PINCH_THRESHOLD:
        return "pinch"  
    return None

############################################################
#####    TRACK, CLICK & SCROLL WITH FACE DETECTION    ######
############################################################

# for wink to last longer for it to click
wink_start_time = None  # Timestamp for when a wink starts
wink_threshold_duration = 0.3  # Duration (in seconds) required for a wink to execute a click

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    """
    Detect wink based on eye aspect ratio (EAR)
    """
    def eye_aspect_ratio(eye_indices):
        # Calculates the Eye Aspect Ratio (EAR) for an eye
        vertical1 = ((face_landmarks[eye_indices[1]].x - face_landmarks[eye_indices[5]].x) ** 2 +
                     (face_landmarks[eye_indices[1]].y - face_landmarks[eye_indices[5]].y) ** 2) ** 0.5
        vertical2 = ((face_landmarks[eye_indices[2]].x - face_landmarks[eye_indices[4]].x) ** 2 +
                     (face_landmarks[eye_indices[2]].y - face_landmarks[eye_indices[4]].y) ** 2) ** 0.5
        horizontal = ((face_landmarks[eye_indices[0]].x - face_landmarks[eye_indices[3]].x) ** 2 +
                      (face_landmarks[eye_indices[0]].y - face_landmarks[eye_indices[3]].y) ** 2) ** 0.5
        # EAR
        return (vertical1 + vertical2) / (2.0 * horizontal)

    EAR_THRESHOLD = 0.2  # Threshold to detect a wink
    left_ear = eye_aspect_ratio(left_eye_indices)
    right_ear = eye_aspect_ratio(right_eye_indices)

    if left_ear < EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
        return "left"  # Left eye wink
    elif right_ear < EAR_THRESHOLD and left_ear > EAR_THRESHOLD:
        return "right"  # Right eye wink
    return None


def detect_head_tilt(face_landmarks):
    """
    Detect head tilt based on nose position
    """
    NOSE_TIP = 1
    NOSE_BASE = 168

    # X-coordinates of the nose tip & base
    nose_tip_x = face_landmarks[NOSE_TIP].x
    nose_base_x = face_landmarks[NOSE_BASE].x

    # horizontal displacement of the nose tip relative to the base
    tilt = nose_tip_x - nose_base_x
    # determine if scroll up/down
    TILT_THRESHOLD = 0.02  # Adjust for sensitivity
    if tilt > TILT_THRESHOLD:  # Head tilted to the right
        return "up"
    elif tilt < -TILT_THRESHOLD:  # Head tilted to the left
        return "down"
    return None

def track_face_cursor(face_landmarks, frame_width, frame_height):
    """
    Track cursor using nose tip from FaceMesh and map it to the entire screen
    """
    NOSE_TIP = 1 
    nose = face_landmarks[NOSE_TIP]

    # screen width and height
    screen_width, screen_height = pyautogui.size()  # Get the screen resolution

    # normalize the position based on the video frame size
    normalized_x = nose.x * frame_width
    normalized_y = nose.y * frame_height

    # scaling factor - fixes cursor being too sensitive
    scale_factor = 1.5 

    # cursor position relative to the screen
    screen_x = int((normalized_x / frame_width) * screen_width * scale_factor)
    screen_y = int((normalized_y / frame_height) * screen_height * scale_factor)
    screen_x = min(max(screen_x, 0), screen_width - 1)
    screen_y = min(max(screen_y, 0), screen_height - 1)

    return screen_x, screen_y


############################################################
#####            VOICE RECOGNITION TO TEXT            ######
############################################################

def detect_mouth_opening(face_landmarks, mouth_indices):
    """
    Detects if mouth is open (distance between the top and bottom lips)
    """
    vertical_dist = ((face_landmarks[mouth_indices[2]].x - face_landmarks[mouth_indices[0]].x) ** 2 +
                     (face_landmarks[mouth_indices[2]].y - face_landmarks[mouth_indices[0]].y) ** 2) ** 0.5
    return vertical_dist

def voice_typing():
    """
    Capture voice input and types it using pyautogui
    """
    with sr.Microphone() as source:
        print("Listening for voice input...")
        sound_effect_voice.play()

        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            pyautogui.typewrite(text) 
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout.")
        except sr.UnknownValueError:
            print("Speech was not clear.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")


############################################################
#####                    MAIN LOOP                    ######
############################################################

# Mouth open coords & open threshold
MOUTH = [13, 312, 14, 317]
MOUTH_OPEN_THRESHOLD = 0.05 

# window properties
WINDOW_NAME_KEYBOARD = 'Virtual Keyboard Control'
WINDOW_NAME_FACE = 'Facial Control'

# named windows to stay on top
cv2.namedWindow(WINDOW_NAME_KEYBOARD, cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_NAME_FACE, cv2.WINDOW_NORMAL)

# windows to stay on top and not close when clicked
cv2.setWindowProperty(WINDOW_NAME_KEYBOARD, cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty(WINDOW_NAME_FACE, cv2.WND_PROP_TOPMOST, 1)

# Initialize keyboard visibility state
keyboard_visible = False
hand_detection_frames = 0  # Counter for consecutive frames with hand detection
HAND_DETECTION_THRESHOLD = 3  # Number of frames needed to show/hide keyboard
HAND_LOST_THRESHOLD = -2

# var to control if a click has happened or not
last_click_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror video & convert to rgb for mediapipe processing
    mirrored_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial & hand landmarks
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    # Hand detection logic 
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

    # Ensure the counter doesn't drop below the loss threshold
    hand_detection_frames = max(HAND_LOST_THRESHOLD, hand_detection_frames)

    # copies for drawing
    keyboard_frame = mirrored_frame.copy()
    face_frame = frame.copy()

    # draw keyboard if visible
    if keyboard_visible:
        draw_virtual_keyboard(keyboard_frame)
        cv2.imshow(WINDOW_NAME_KEYBOARD, keyboard_frame)
    
    ##### FACE TRACKING #####
    cv2.imshow(WINDOW_NAME_FACE, face_frame)
    if results_face.multi_face_landmarks:
        for landmarks in results_face.multi_face_landmarks:
            face_landmarks = landmarks.landmark

            # (1) Detect winks for click
            wink = detect_wink(face_landmarks, LEFT_EYE, RIGHT_EYE)
            if wink == "left" or wink == "right":
                if wink_start_time is None:
                    wink_start_time = time.time()  # track wink duration
                elif time.time() - wink_start_time >= wink_threshold_duration:
                    pyautogui.click()
                    sound_effect_mouse.play()
                    wink_start_time = None 
            else:
                wink_start_time = None 

            # (2) Detect mouth open for speech recog
            mouth_opening = detect_mouth_opening(face_landmarks, MOUTH)
            if mouth_opening > MOUTH_OPEN_THRESHOLD:
                voice_typing()

            # (3) Detect head tilts for scrolling
            scroll_direction = detect_head_tilt(face_landmarks)
            if scroll_direction == "up":
                pyautogui.scroll(50)
                time.sleep(0.1)
            elif scroll_direction == "down":
                pyautogui.scroll(-50)
                time.sleep(0.1)


            

            # (4) Move cursor with the nose tip
            cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
            pyautogui.moveTo(cursor_x, cursor_y)
            

            # (5) detect pinching to select key (if keyboard active)
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
                            current_time = time.time()
                            if current_time - last_click_time > 0.5:
                                print(f"Clicked key: {key}")

                                if key in SPECIAL_KEYS:
                                    pyautogui.press(SPECIAL_KEYS[key])
                                else:
                                    pyautogui.write(key) 

                                sound_effect_click.play() 
                                last_click_time = current_time

    # Display frame with virtual keyboard
    cv2.imshow('Facial Control', mirrored_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminating program...")
        break

cap.release()
cv2.destroyAllWindows()
