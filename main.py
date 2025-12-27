import cv2
import mediapipe as mp
import numpy as np
import math

MEME_IMAGE_PATH = 'thinking_meme.jpeg'  
POSE_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
ELBOW_ANGLE_THRESHOLD = 80 
MOUTH_TOUCH_THRESHOLD = 0.06 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=POSE_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)

def calculate_angle(a, b, c):
    """Calculates the angle at point b given three points a, b, and c."""
    a = np.array(a) 
    b = np.array(b)
    c = np.array(c) 

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_distance(a, b):
    """Calculates Euclidean distance between two points a and b."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

cap = cv2.VideoCapture(0)

meme_img_original = cv2.imread(MEME_IMAGE_PATH)
if meme_img_original is None:
    print(f"Error: Could not load image form {MEME_IMAGE_PATH}. Please check file name.")
    exit()

meme_resized = None

print("Starting Meme Mirror... Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_height, frame_width, _ = image.shape

    if meme_resized is None:
        meme_resized = cv2.resize(meme_img_original, (frame_width, frame_height))

    skeleton_image = np.zeros(image.shape, dtype=np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)

    pose_detected = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                 landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        wrist_to_mouth_dist = calculate_distance(r_wrist, mouth)

        if elbow_angle < ELBOW_ANGLE_THRESHOLD and wrist_to_mouth_dist < MOUTH_TOUCH_THRESHOLD:
            pose_detected = True
        else:
            mp_drawing.draw_landmarks(
                skeleton_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
            cv2.putText(skeleton_image, f"Angle: {int(elbow_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(skeleton_image, f"Dist: {round(wrist_to_mouth_dist, 3)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if pose_detected:
        final_output = meme_resized
    else:
        final_output = skeleton_image

    cv2.imshow('The Meme Mirror', final_output)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()