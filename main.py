import cv2
import mediapipe as mp
import numpy as np

MEME_LIBRARY = {
    "thinking": "memes/thinking.jpg",
    "pointing": "memes/pointing.jpg",
    "shocked":  "memes/shocked.jpg",
    "staring":  "memes/staring.jpg"
}

SHOW_CAMERA_FEED = True

TOUCH_THRESHOLD = 0.08
MOUTH_OPEN_THRESHOLD = 0.25
POINTING_HEIGHT_OFFSET = 0.05

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    smooth_landmarks=True,
    model_complexity=1,
    refine_face_landmarks=True
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=4)
hand_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=3)

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_hand_finger_tip(hand_landmarks, finger_id=8):
    if hand_landmarks is None:
        return None
    lm = hand_landmarks.landmark[finger_id]
    return [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]

def get_mouth_aspect_ratio(face_landmarks):
    if face_landmarks is None:
        return 0
    lms = face_landmarks.landmark
    upper_lip = [lms[13].x, lms[13].y]
    lower_lip = [lms[14].x, lms[14].y]
    left_corner = [lms[78].x, lms[78].y]
    right_corner = [lms[308].x, lms[308].y]
    
    vertical = calculate_distance(upper_lip, lower_lip)
    horizontal = calculate_distance(left_corner, right_corner)
    
    if horizontal == 0:
        return 0
    return vertical / horizontal

meme_images = {}
for name, path in MEME_LIBRARY.items():
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load {path}. Check filename!")
        exit()
    meme_images[name] = img

cap = cv2.VideoCapture(0)
pose_history = []
HISTORY_LENGTH = 3

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    resized_memes = {}
    for name, img in meme_images.items():
        resized = cv2.resize(img, (w, h // 2))
        resized_memes[name] = {
            "normal": resized,
            "flipped": cv2.flip(resized, 1)
        }

    output_image = image.copy() if SHOW_CAMERA_FEED else np.zeros(image.shape, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    current_pose = "staring"
    is_flipped = False

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark

        nose = [lms[0].x, lms[0].y]
        left_eye = [lms[2].x, lms[2].y]
        right_eye = [lms[5].x, lms[5].y]
        mouth_left = [lms[9].x, lms[9].y]
        mouth_right = [lms[10].x, lms[10].y]
        shoulders_y = (lms[11].y + lms[12].y) / 2

        r_index = get_hand_finger_tip(results.right_hand_landmarks, 8)
        l_index = get_hand_finger_tip(results.left_hand_landmarks, 8)

        mouth_ratio = get_mouth_aspect_ratio(results.face_landmarks)
        cv2.putText(output_image, f"Mouth: {mouth_ratio:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            upper_lip_pt = (int(face_lms[13].x * w), int(face_lms[13].y * h))
            lower_lip_pt = (int(face_lms[14].x * w), int(face_lms[14].y * h))
            cv2.circle(output_image, upper_lip_pt, 5, (0, 255, 0), -1)
            cv2.circle(output_image, lower_lip_pt, 5, (0, 0, 255), -1)
            cv2.line(output_image, upper_lip_pt, lower_lip_pt, (255, 255, 0), 2)

        if r_index:
            cx, cy = int(r_index[0]*w), int(r_index[1]*h)
            cv2.circle(output_image, (cx, cy), 12, (0, 255, 0), -1)
            cv2.putText(output_image, "R", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if l_index:
            cx, cy = int(l_index[0]*w), int(l_index[1]*h)
            cv2.circle(output_image, (cx, cy), 12, (255, 0, 255), -1)
            cv2.putText(output_image, "L", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        mouth_cx = int(((mouth_left[0] + mouth_right[0]) / 2) * w)
        mouth_cy = int(((mouth_left[1] + mouth_right[1]) / 2) * h)
        cv2.circle(output_image, (mouth_cx, mouth_cy), 10, (255, 0, 0), -1)

        thinking_triggered = False
        if r_index:
            r_dist = calculate_distance(r_index[:2], mouth_right)
            cv2.putText(output_image, f"R_Dist: {r_dist:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if r_dist < TOUCH_THRESHOLD and r_index[1] < shoulders_y:
                current_pose = "thinking"
                is_flipped = True
                thinking_triggered = True

        if l_index and not thinking_triggered:
            l_dist = calculate_distance(l_index[:2], mouth_left)
            cv2.putText(output_image, f"L_Dist: {l_dist:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            if l_dist < TOUCH_THRESHOLD and l_index[1] < shoulders_y:
                current_pose = "thinking"
                is_flipped = False
                thinking_triggered = True

        if not thinking_triggered:
            nose_y = nose[1]
            
            r_pointing = False
            l_pointing = False
            
            if r_index:
                finger_height = r_index[1]
                cv2.putText(output_image, f"R_Height: {finger_height:.2f} Nose: {nose_y:.2f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if finger_height < nose_y - POINTING_HEIGHT_OFFSET:
                    r_pointing = True
            
            if l_index:
                finger_height = l_index[1]
                cv2.putText(output_image, f"L_Height: {finger_height:.2f} Nose: {nose_y:.2f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                if finger_height < nose_y - POINTING_HEIGHT_OFFSET:
                    l_pointing = True
            
            if r_pointing:
                current_pose = "pointing"
                is_flipped = True
            elif l_pointing:
                current_pose = "pointing"
                is_flipped = False

        if current_pose == "staring" and mouth_ratio > MOUTH_OPEN_THRESHOLD:
            current_pose = "shocked"

        if current_pose == "staring":
            mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, drawing_spec, drawing_spec)
            mp_drawing.draw_landmarks(output_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec, hand_spec)
            mp_drawing.draw_landmarks(output_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec, hand_spec)

    pose_history.append(current_pose)
    if len(pose_history) > HISTORY_LENGTH:
        pose_history.pop(0)

    stable_pose = current_pose
    if len(pose_history) >= 2:
        if pose_history.count(pose_history[-1]) >= 2:
            stable_pose = pose_history[-1]

    version = "flipped" if is_flipped else "normal"
    meme_display = resized_memes[stable_pose][version]
    
    camera_half = cv2.resize(output_image, (w, h // 2))
    final_display = np.vstack([camera_half, meme_display])

    cv2.imshow('Meme Mirror', final_display)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()