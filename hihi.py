import cv2
import pyautogui
import mediapipe as mp
import time

# Initialize MediaPipe for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize OpenCV to capture video from the webcam
cap = cv2.VideoCapture(0)

# Wait for camera to initialize
time.sleep(2)

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    """
    Detects winks based on eye aspect ratio (EAR).
    """
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
    
    EAR_THRESHOLD = 0.2  # Adjust this threshold as needed
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

# Eye landmarks based on MediaPipe documentation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Check if face landmarks are detected
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            face_landmarks = landmarks.landmark

            # Detect winks
            wink = detect_wink(face_landmarks, LEFT_EYE, RIGHT_EYE)
            if wink == "left" or wink == "right":
                pyautogui.click()  # Perform a mouse click
                time.sleep(0.5)  # Add a delay to avoid multiple clicks

            # Detect head tilts for scrolling
            scroll_direction = detect_head_tilt(face_landmarks)
            if scroll_direction == "up":
                pyautogui.scroll(20)  # Scroll up
                time.sleep(0.1)
            elif scroll_direction == "down":
                pyautogui.scroll(-20)  # Scroll down
                time.sleep(0.1)

            # Optional: Map nose to move the cursor (from the original code)
            nose = face_landmarks[1]  # Nose tip
            screen_x = int((1-nose.x) * pyautogui.size().width)
            screen_y = int(nose.y * pyautogui.size().height)
            pyautogui.moveTo(screen_x, screen_y)

    # Display the frame (optional, for debugging)
    cv2.imshow('Facial Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()