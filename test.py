import cv2
import pyautogui
import mediapipe as mp
import time
import math
import pygame
from collections import deque
import speech_recognition as sr
import numpy as np
import screeninfo

# for cursor control
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# for virtual keyboard
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

# sounds
pygame.mixer.init()
sound_effect_click = pygame.mixer.Sound("sounds\key_press.wav")
sound_effect_voice = pygame.mixer.Sound("sounds\bell_and_water.wav")
sound_effect_drag = pygame.mixer.Sound("sounds\pluck.mp3")
sound_effect_mouse = pygame.mixer.Sound("sounds\mouse.mp3")
sound_effect_voiceoff = pygame.mixer.Sound("sounds\voiceoff.wav")
sound_effect_act = pygame.mixer.Sound("sounds\act.mp3")
sound_effect_des = pygame.mixer.Sound("sounds\des.mp3")



# Wait for camera to initialize
time.sleep(2)

drag_enabled = False
last_eyebrow_raised = False  # Tracks the previous eyebrow state
last_eyebrow_toggle_time = 0  # To debounce eyebrow raises
TOGGLE_DEBOUNCE_TIME = 0.5  # Debounce duration in seconds

LEFT_EYEBROW = [65]  # Updated top of left eyebrow
RIGHT_EYEBROW = [295]  # Updated top of right eyebrow
LEFT_EYE_TOP = [159]  # Verified top of left eye
RIGHT_EYE_TOP = [386]  # Verified top of right eye

EYEBROW_RAISE_THRESHOLD = 0.05  # Adjust for sensitivity

SPECIAL_KEYS = {

    'Space': 'space',  # Space bar

    'Enter': 'enter',  # Enter key

    'Backspace': 'backspace',  # Backspace key

    'Shift': 'shift'  # Shift key

}

def detect_eyebrow_raise(face_landmarks):

    def calculate_distance(eyebrow_index, eye_index):
        return abs(face_landmarks[eyebrow_index].y - face_landmarks[eye_index].y)

    left_distance = calculate_distance(LEFT_EYEBROW[0], LEFT_EYE_TOP[0])
    right_distance = calculate_distance(RIGHT_EYEBROW[0], RIGHT_EYE_TOP[0])

    return left_distance > EYEBROW_RAISE_THRESHOLD or right_distance > EYEBROW_RAISE_THRESHOLD

def detect_eyebrow_toggle(face_landmarks):

    global drag_enabled, last_eyebrow_raised, last_eyebrow_toggle_time

    eyebrow_raised = detect_eyebrow_raise(face_landmarks)

    if eyebrow_raised and not last_eyebrow_raised:  # Transition: Not Raised -> Raised
        current_time = time.time()
        if current_time - last_eyebrow_toggle_time > TOGGLE_DEBOUNCE_TIME:  # Debounce logic
            drag_enabled = not drag_enabled  # Toggle the drag state
            last_eyebrow_toggle_time = current_time
            print("Drag toggled:", "Enabled" if drag_enabled else "Disabled")
            
        
            if drag_enabled:
                sound_effect_act.play()  
            else:
                sound_effect_des.play()  

    # Update the last eyebrow state
    last_eyebrow_raised = eyebrow_raised

KEYS = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\'', 'Enter'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'Shift'],
    ['Space']
]

KEY_WIDTH = 40 
KEY_HEIGHT = 40  
KEY_MARGIN = 4 
KEY_POS = (20, 50)  

LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Right eye landmarks


caps_lock = False

def draw_virtual_keyboard(frame, hovered_key=None):
    
    overlay = frame.copy()  
    alpha = 0.6  # Transparency level


    y_offset = KEY_POS[1]
    space_rect = None  
    for row in KEYS:
        x_offset = KEY_POS[0]
        for key in row:
           
            if key == 'Space':
                key_width = KEY_WIDTH * 2 + KEY_MARGIN 
                key_text = "Space"  # Space key prints a space, not the word "Space"
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

            
            if hovered_key == key:
                color = (100, 255, 100)  
            else:
                color = (200, 200, 255) 

            
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), color, -1)  # Fill the key

            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + key_width, y_offset + KEY_HEIGHT), (100, 100, 150), 2, cv2.LINE_AA)

            # Add the text for each key
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_offset + (key_width - text_size[0]) // 2
            text_y = y_offset + (KEY_HEIGHT + text_size[1]) // 2
            cv2.putText(overlay, key_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 100), font_thickness, cv2.LINE_AA)

            
            x_offset += key_width + KEY_MARGIN
        
        y_offset += KEY_HEIGHT + KEY_MARGIN

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return space_rect

def get_key_at_position(x, y):
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
wink_start_time = None  
wink_threshold_duration = 0.3  # Duration (in seconds) required for a wink to execute a click

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    
    def eye_aspect_ratio(eye_indices):
        
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
        return "left" 
    elif right_ear < EAR_THRESHOLD and left_ear > EAR_THRESHOLD:
        return "right"  
    return None

def detect_head_tilt(face_landmarks):

    NOSE_TIP = 1  # Nose tip
    NOSE_BASE = 168  # Nose base (central)

    nose_tip_x = face_landmarks[NOSE_TIP].x
    nose_base_x = face_landmarks[NOSE_BASE].x

    tilt = nose_tip_x - nose_base_x

    TILT_THRESHOLD = 0.02  # Adjust for sensitivity
    if tilt > TILT_THRESHOLD:  # Head tilted to the right
        return "up"
    elif tilt < -TILT_THRESHOLD:  # Head tilted to the left
        return "down"
    return None

def detect_mouth_opening(face_landmarks, mouth_indices):

    vertical_dist = ((face_landmarks[mouth_indices[2]].x - face_landmarks[mouth_indices[0]].x) ** 2 +

                     (face_landmarks[mouth_indices[2]].y - face_landmarks[mouth_indices[0]].y) ** 2) ** 0.5

    return vertical_dist

def voice_typing():

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

    thumb_tip = hand_landmarks[4]  # Thumb tip
    index_tip = hand_landmarks[8]  # Index finger tip

    # Calculate distance between thumb and index finger
    distance_thumb_index = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    PINCH_THRESHOLD = 0.05  # Distance threshold for pinch

    if distance_thumb_index < PINCH_THRESHOLD:
        return "pinch" 
    return None


def track_face_cursor(face_landmarks, frame_width, frame_height):

    NOSE_TIP = 1  
    nose = face_landmarks[NOSE_TIP]

    screen_width, screen_height = pyautogui.size()

    normalized_x = nose.x
    normalized_y = nose.y
    
    SAFE_ZONE_X_MIN = 0.3
    SAFE_ZONE_X_MAX = 0.7
    SAFE_ZONE_Y_MIN = 0.4  
    SAFE_ZONE_Y_MAX = 0.8  

    def map_coordinate(value, safe_min, safe_max):
        if value < safe_min:
            return (value / safe_min) * 0.3
        elif value > safe_max:
            return 0.7 + ((value - safe_max) / (1 - safe_max)) * 0.3
        else:
            return 0.3 + ((value - safe_min) / (safe_max - safe_min)) * 0.4

    mapped_x = map_coordinate(normalized_x, SAFE_ZONE_X_MIN, SAFE_ZONE_X_MAX)

    # This means the nose needs to be lower in the frame to reach top edges
    mapped_y = map_coordinate(normalized_y, SAFE_ZONE_Y_MIN, SAFE_ZONE_Y_MAX)

    VERTICAL_OFFSET = 0.15  
    mapped_y = mapped_y - VERTICAL_OFFSET

    screen_x = int(mapped_x * screen_width)
    screen_y = int(mapped_y * screen_height)
 
    screen_x = min(max(screen_x, 0), screen_width - 1)
    screen_y = min(max(screen_y, 0), screen_height - 1)
    
    return screen_x, screen_y


# Add a variable to control if a click has happened or not
last_click_time = time.time()


MOUTH = [13, 312, 14, 317]  

MOUTH_OPEN_THRESHOLD = 0.05  # Adjust this threshold as needed

WINDOW_NAME_KEYBOARD = 'Virtual Keyboard Control'
WINDOW_NAME_FACE = 'Facial Control'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

cv2.namedWindow(WINDOW_NAME_KEYBOARD, cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_NAME_FACE, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(WINDOW_NAME_KEYBOARD, cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty(WINDOW_NAME_FACE, cv2.WND_PROP_TOPMOST, 1)

cv2.resizeWindow(WINDOW_NAME_KEYBOARD, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow(WINDOW_NAME_FACE, WINDOW_WIDTH, WINDOW_HEIGHT)

screen = screeninfo.get_monitors()[0]

cv2.moveWindow(WINDOW_NAME_KEYBOARD, 0, screen.height - WINDOW_HEIGHT)

keyboard_visible = False
hand_detection_frames = 0 
HAND_DETECTION_THRESHOLD = 5  




# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mirrored_frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)

    results_face = face_mesh.process(rgb_frame)

    results_hands = hands.process(rgb_frame)

    if results_hands.multi_hand_landmarks:
        hand_detection_frames += 1
        if hand_detection_frames >= HAND_DETECTION_THRESHOLD and not keyboard_visible:
            if not keyboard_visible:
                keyboard_visible = True
                cv2.namedWindow(WINDOW_NAME_KEYBOARD, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME_KEYBOARD, cv2.WND_PROP_TOPMOST, 1)
                cv2.resizeWindow(WINDOW_NAME_KEYBOARD, WINDOW_WIDTH, WINDOW_HEIGHT)
                cv2.moveWindow(WINDOW_NAME_KEYBOARD, 0, screen.height - WINDOW_HEIGHT + 50)
    else:
        hand_detection_frames -= 1
        if hand_detection_frames <= 0:
            if keyboard_visible:
                keyboard_visible = False
                cv2.destroyWindow(WINDOW_NAME_KEYBOARD)
        hand_detection_frames = max(0, hand_detection_frames)

    keyboard_frame = mirrored_frame.copy()
    face_frame = frame.copy()

    if keyboard_visible:
        draw_virtual_keyboard(keyboard_frame)
        cv2.imshow(WINDOW_NAME_KEYBOARD, keyboard_frame)
    
    cv2.imshow(WINDOW_NAME_FACE, face_frame)

    if results_face.multi_face_landmarks:
        for landmarks in results_face.multi_face_landmarks:
            face_landmarks = landmarks.landmark

            wink = detect_wink(face_landmarks, LEFT_EYE, RIGHT_EYE)
            if wink == "left" or wink == "right":
                if wink_start_time is None:
                    wink_start_time = time.time()  
                elif time.time() - wink_start_time >= wink_threshold_duration:
                    pyautogui.click()  # Perform a mouse click
                    sound_effect_mouse.play()
                    wink_start_time = None  # Reset the wink timer after a successful click
            else:
                wink_start_time = None  # Reset if no wink is detected

            mouth_opening = detect_mouth_opening(face_landmarks, MOUTH)
            if mouth_opening > MOUTH_OPEN_THRESHOLD:
                voice_typing()  

            scroll_direction = detect_head_tilt(face_landmarks)
            if scroll_direction == "up":
                pyautogui.scroll(50)  # Scroll up
                time.sleep(0.1)
            elif scroll_direction == "down":
                pyautogui.scroll(-50)  # Scroll down
                time.sleep(0.1)

            detect_eyebrow_toggle(face_landmarks)

            if drag_enabled:
                cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
                pyautogui.mouseDown()
                pyautogui.moveTo(cursor_x, cursor_y)
            else:
                pyautogui.mouseUp()

            cursor_x, cursor_y = track_face_cursor(face_landmarks, mirrored_frame.shape[1], mirrored_frame.shape[0])
            pyautogui.moveTo(cursor_x, cursor_y)

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

    cv2.imshow('Facial Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminating program...")
        break

cap.release()
cv2.destroyAllWindows()
