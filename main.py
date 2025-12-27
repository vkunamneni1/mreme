import cv2
import mediapipe as mp
import numpy as np

MEME_IMAGE_PATH = 'thinking_meme.jpg'

DEBUG_MODE = True

ELBOW_ANGLE_THRESHOLD = 90
MOUTH_TOUCH_THRESHOLD = 0.15

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)

def calculate_angle(a, b, c):
    """Calculates angle at joint b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

cap = cv2.VideoCapture(0)
meme_img_original = cv2.imread(MEME_IMAGE_PATH)
meme_resized = None

if meme_img_original is None:
    print(f"Error: Could not load {MEME_IMAGE_PATH}")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)

    if meme_resized is None:
        h, w, _ = image.shape
        meme_resized = cv2.resize(meme_img_original, (w, h))

    if DEBUG_MODE:
        output_image = image.copy()
    else:
        output_image = np.zeros(image.shape, dtype=np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    pose_match = False

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        r_angle = calculate_angle(
            [lms[12].x, lms[12].y],
            [lms[14].x, lms[14].y],
            [lms[16].x, lms[16].y]
        )
        r_dist = calculate_distance([lms[16].x, lms[16].y], [lms[10].x, lms[10].y])

        l_angle = calculate_angle(
            [lms[11].x, lms[11].y],
            [lms[13].x, lms[13].y],
            [lms[15].x, lms[15].y]
        )
        l_dist = calculate_distance([lms[15].x, lms[15].y], [lms[9].x, lms[9].y])

        cv2.putText(output_image, f"R_Ang:{int(r_angle)} R_Dist:{r_dist:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_image, f"L_Ang:{int(l_angle)} L_Dist:{l_dist:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        right_match = (r_angle < ELBOW_ANGLE_THRESHOLD and r_dist < MOUTH_TOUCH_THRESHOLD)
        left_match  = (l_angle < ELBOW_ANGLE_THRESHOLD and l_dist < MOUTH_TOUCH_THRESHOLD)

        if right_match or left_match:
            pose_match = True

        if not pose_match:
            mp_drawing.draw_landmarks(
                output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec
            )

    final_display = meme_resized if pose_match else output_image
    cv2.imshow('Meme Mirror', final_display)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()