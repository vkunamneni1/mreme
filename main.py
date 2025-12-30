#!/usr/bin/env python3

import os
import math
import time
import random
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


WINDOW_NAME = "Meme Mirror"
TOUCH_THRESHOLD = 0.08
MOUTH_OPEN_THRESHOLD = 0.5
POINTING_OFFSET = 0.05

MEME_PATHS = {
    "thinking": "memes/thinking.jpg",
    "pointing": "memes/pointing.jpg",
    "shocked": "memes/shocked.jpg",
    "staring": "memes/staring.jpg",
}

POSE_COLORS = {
    "staring": (90, 90, 90),
    "thinking": (0, 180, 255),
    "pointing": (0, 255, 120),
    "shocked": (80, 80, 255),
}

ROAST_MESSAGES = [
    "ur not even trying",
    "my grandma poses better",
    "is that a pose or a cry for help",
    "the meme is embarrassed for u",
    "AI judging u rn",
    "touch grass immediately",
    "ur skeleton is disappointed",
    "webcam quality matches ur effort",
    "pose like u mean it coward",
    "even the pixels are cringing",
    "delete this from ur memory",
    "ur aura is beige",
    "main character? more like NPC",
    "the algorithm has seen enough",
    "ur giving nothing",
]


class AppState:
    def __init__(self):
        self.debug_mode = False
        self.fullscreen = False
        self.meme_only = False
        self.paused = False
        self.side_by_side = False
        self.pip_mode = False
        self.zoom_face = False
        self.recording = False
        self.watermark = False
        self.vertical_mode = False
        self.face_crop = False
        self.smooth_transition = True
        self.chaos_mode = False
        self.show_help = False

        self.streak = 0
        self.transition_alpha = 1.0
        self.current_meme = None
        self.manual_pose_index = -1
        self.camera_index = 0
        self.previous_meme = None
        self.blend_alpha = 1.0
        self.flash_intensity = 0
        self.video_writer = None
        self.gif_buffer = deque(maxlen=90)
        self.pose_history = []
        self.last_detected_pose = "staring"
        self.frame_timestamp = time.time()
        self.session_start = time.time()
        self.pose_counts = {"thinking": 0, "pointing": 0, "shocked": 0, "staring": 0}
        self.cringe_score = 0
        self.ego_deaths = 0
        self.last_roast = ""
        self.roast_timer = 0
        self.frame_count = 0
        self.ui_pulse = 0


def calculate_distance(point_a, point_b):
    return np.linalg.norm(np.array(point_a) - np.array(point_b))


def get_finger_position(hand_landmarks, finger_index=8):
    if hand_landmarks is None:
        return None
    landmark = hand_landmarks.landmark[finger_index]
    return [landmark.x, landmark.y, getattr(landmark, "visibility", 1.0)]


def calculate_mouth_openness(face_landmarks):
    if face_landmarks is None:
        return 0
    landmarks = face_landmarks.landmark
    upper_lip = [landmarks[13].x, landmarks[13].y]
    lower_lip = [landmarks[14].x, landmarks[14].y]
    left_corner = [landmarks[78].x, landmarks[78].y]
    right_corner = [landmarks[308].x, landmarks[308].y]
    vertical_distance = calculate_distance(upper_lip, lower_lip)
    horizontal_distance = calculate_distance(left_corner, right_corner)
    if horizontal_distance == 0:
        return 0
    return vertical_distance / horizontal_distance


def apply_glitch_effect(text):
    glitch_chars = "@#$%&*!?"
    result = ""
    for char in text:
        if random.random() > 0.15:
            result += char
        else:
            result += random.choice(glitch_chars)
    return result


def apply_corruption_effect(frame, intensity=0.1):
    if random.random() >= intensity:
        return frame
    height, width = frame.shape[:2]
    start_y = random.randint(0, height - 20)
    end_y = random.randint(start_y, height)
    if start_y < end_y:
        horizontal_shift = random.randint(-30, 30)
        frame[start_y:end_y] = np.roll(frame[start_y:end_y], horizontal_shift, axis=1)
    return frame


def draw_corner_brackets(image, x1, y1, x2, y2, color, corner_length=8, thickness=2):
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)


def draw_scanline_overlay(image, line_spacing=4, overlay_alpha=0.03):
    overlay = image.copy()
    height = image.shape[0]
    for y_position in range(0, height, line_spacing):
        cv2.line(overlay, (0, y_position), (image.shape[1], y_position), (0, 0, 0), 1)
    cv2.addWeighted(overlay, overlay_alpha, image, 1 - overlay_alpha, 0, image)


def get_face_bounding_box(face_landmarks, frame_width, frame_height, padding=0.3):
    if face_landmarks is None:
        return None
    x_coords = [lm.x * frame_width for lm in face_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in face_landmarks.landmark]
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)
    face_width = max_x - min_x
    face_height = max_y - min_y
    pad_x = face_width * padding
    pad_y = face_height * padding
    return (
        int(max(0, min_x - pad_x)),
        int(max(0, min_y - pad_y)),
        int(min(frame_width, max_x + pad_x)),
        int(min(frame_height, max_y + pad_y)),
    )


def blend_face_onto_meme(frame, meme_image, face_bbox, target_region):
    if face_bbox is None:
        return meme_image

    face_x1, face_y1, face_x2, face_y2 = face_bbox
    target_x1, target_y1, target_x2, target_y2 = target_region

    face_crop = frame[face_y1:face_y2, face_x1:face_x2]
    if face_crop.size == 0:
        return meme_image

    target_width = target_x2 - target_x1
    target_height = target_y2 - target_y1
    if target_width <= 0 or target_height <= 0:
        return meme_image

    resized_face = cv2.resize(face_crop, (target_width, target_height))
    result = meme_image.copy()

    ellipse_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    center = (target_width // 2, target_height // 2)
    axes = (target_width // 2 - 5, target_height // 2 - 5)
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)
    ellipse_mask = cv2.GaussianBlur(ellipse_mask, (21, 21), 0)

    mask_3channel = cv2.merge([ellipse_mask] * 3) / 255.0
    region_of_interest = result[target_y1:target_y2, target_x1:target_x2]
    blended = (resized_face * mask_3channel + region_of_interest * (1 - mask_3channel))
    result[target_y1:target_y2, target_x1:target_x2] = blended.astype(np.uint8)

    return result


def load_meme_images():
    images = {}
    for pose_name, file_path in MEME_PATHS.items():
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Cannot load meme image at {file_path}")
            exit(1)
        images[pose_name] = image
    return images


def setup_directories():
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)


def create_pose_detector():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
        model_complexity=1,
        refine_face_landmarks=True,
    )


def show_splash_screen(meme_images):
    screen_width, screen_height = 800, 600
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for frame_index in range(45):
        canvas[:] = (8, 8, 8)

        for y in range(0, screen_height, 4):
            fade_amount = 0.015
            row = canvas[y : y + 1, :]
            canvas[y : y + 1, :] = np.clip(row.astype(float) * (1 - fade_amount), 0, 255).astype(np.uint8)

        pulse_value = (math.sin(frame_index * 0.25) + 1) / 2
        red_intensity = int(180 + 75 * pulse_value)

        text = "WARNING"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        text_x = screen_width // 2 - text_w // 2
        cv2.putText(canvas, text, (text_x, screen_height // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, red_intensity), 2)

        subtitle = "flashing lights / photosensitive seizure"
        (sub_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, subtitle, (screen_width // 2 - sub_w // 2, screen_height // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        box_color = (0, 0, red_intensity)
        draw_corner_brackets(canvas, screen_width // 2 - 240, screen_height // 2 - 75,
                             screen_width // 2 + 240, screen_height // 2 + 45,
                             box_color, corner_length=20, thickness=2)

        hint = "[ Q ] quit    [ SPACE ] continue"
        (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, hint, (screen_width // 2 - hint_w // 2, screen_height // 2 + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" ") and frame_index > 8:
            break

    for frame_index in range(80):
        canvas[:] = (5, 5, 5)
        draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

        if frame_index < 25:
            progress = frame_index / 25
            text_scale = 0.3 + progress * 1.2
            text_alpha = min(1, progress * 2)
            text = "MEME MIRROR"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
            brightness = int(255 * text_alpha)
            cv2.putText(canvas, text, (screen_width // 2 - text_w // 2, screen_height // 2 + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (brightness, brightness, brightness), 2)

        elif frame_index < 50:
            phase = (frame_index - 25) / 25
            glitch_offset = int(4 * math.sin(phase * math.pi * 4)) if frame_index < 40 else 0

            (meme_w, _), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)
            (mirror_w, _), _ = cv2.getTextSize("MIRROR", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)

            meme_x = screen_width // 2 - meme_w // 2
            mirror_x = screen_width // 2 - mirror_w // 2

            if frame_index < 38:
                cv2.putText(canvas, "MEME", (meme_x + glitch_offset, screen_height // 2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 80, 80), 3)
                cv2.putText(canvas, "MEME", (meme_x - glitch_offset, screen_height // 2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80, 255, 255), 3)
            cv2.putText(canvas, "MEME", (meme_x, screen_height // 2 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

            if frame_index < 38:
                cv2.putText(canvas, "MIRROR", (mirror_x + glitch_offset, screen_height // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 80, 80), 3)
                cv2.putText(canvas, "MIRROR", (mirror_x - glitch_offset, screen_height // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80, 255, 255), 3)
            cv2.putText(canvas, "MIRROR", (mirror_x, screen_height // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

            draw_corner_brackets(canvas, screen_width // 2 - 180, screen_height // 2 - 70,
                                 screen_width // 2 + 180, screen_height // 2 + 90,
                                 (255, 255, 255), corner_length=15, thickness=2)

        elif frame_index < 70:
            progress = (frame_index - 50) / 20
            bar_y = screen_height // 2 + 30
            bar_x = screen_width // 2 - 150
            bar_width = int(progress * 300)

            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 3), (255, 255, 255), -1)
            draw_corner_brackets(canvas, bar_x - 5, bar_y - 5, bar_x + 305, bar_y + 8,
                                 (80, 80, 80), corner_length=4, thickness=1)

            loading_messages = ["loading vibes", "calibrating cringe", "summoning memes",
                                "processing ego", "init chaos"]
            loading_text = random.choice(loading_messages)
            (load_w, _), _ = cv2.getTextSize(loading_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(canvas, loading_text, (screen_width // 2 - load_w // 2, bar_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

            percent_text = f"{int(progress * 100)}%"
            cv2.putText(canvas, percent_text, (bar_x + 308, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

        else:
            flash_value = 255 if frame_index % 2 == 0 else 0
            canvas[:] = (flash_value, flash_value, flash_value)
            go_text = random.choice(["GO", "→", "POSE"])
            (go_w, go_h), _ = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
            text_color = (0, 0, 0) if flash_value else (255, 255, 255)
            cv2.putText(canvas, go_text, (screen_width // 2 - go_w // 2, screen_height // 2 + go_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 4)

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(45) & 0xFF == ord("q"):
            return False

    tutorial_slides = [
        ("STARING", "just look at camera", "staring", (90, 90, 90)),
        ("THINKING", "finger on chin", "thinking", (0, 180, 255)),
        ("POINTING", "point up above head", "pointing", (0, 255, 120)),
        ("SHOCKED", "open mouth wide", "shocked", (80, 80, 255)),
    ]

    for slide_index, (title, description, pose_key, color) in enumerate(tutorial_slides):
        canvas[:] = (10, 10, 10)
        draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(canvas, title, (screen_width // 2 - title_w // 2, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        (desc_w, _), _ = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(canvas, description, (screen_width // 2 - desc_w // 2, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        if pose_key in meme_images:
            thumbnail = cv2.resize(meme_images[pose_key], (300, 170))
            thumb_x = screen_width // 2 - 150
            thumb_y = 160
            canvas[thumb_y : thumb_y + 170, thumb_x : thumb_x + 300] = thumbnail
            draw_corner_brackets(canvas, thumb_x - 2, thumb_y - 2, thumb_x + 302, thumb_y + 172,
                                 color, corner_length=12, thickness=2)

        page_text = f"{slide_index + 1}/5"
        cv2.putText(canvas, page_text, (screen_width - 60, screen_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        cv2.putText(canvas, "SPACE to continue", (screen_width // 2 - 80, screen_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                break

    canvas[:] = (10, 10, 10)
    draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

    (controls_w, _), _ = cv2.getTextSize("CONTROLS", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.putText(canvas, "CONTROLS", (screen_width // 2 - controls_w // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

    control_list = [
        "H - show/hide help menu",
        "A - toggle face crop",
        "V - vertical mode (9:16)",
        "W - watermark on/off",
        "S - screenshot",
        "R - start/stop recording",
        "F - fullscreen",
        "Q - quit",
    ]
    for i, control in enumerate(control_list):
        cv2.putText(canvas, control, (screen_width // 2 - 120, 110 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.putText(canvas, "5/5", (screen_width - 60, screen_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(canvas, "SPACE to start", (screen_width // 2 - 70, screen_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.imshow(WINDOW_NAME, canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            break

    canvas[:] = (0, 0, 0)
    (ready_w, ready_h), _ = cv2.getTextSize("READY", cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
    cv2.putText(canvas, "READY", (screen_width // 2 - ready_w // 2, screen_height // 2 + ready_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
    cv2.imshow(WINDOW_NAME, canvas)
    cv2.waitKey(500)

    return True


def detect_current_pose(pose_results, frame_width, frame_height):
    detected_pose = "staring"
    should_flip = False

    if pose_results.pose_landmarks is None:
        return detected_pose, should_flip

    landmarks = pose_results.pose_landmarks.landmark
    nose_position = [landmarks[0].x, landmarks[0].y]
    left_mouth = [landmarks[9].x, landmarks[9].y]
    right_mouth = [landmarks[10].x, landmarks[10].y]
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

    right_finger = get_finger_position(pose_results.right_hand_landmarks)
    left_finger = get_finger_position(pose_results.left_hand_landmarks)
    mouth_ratio = calculate_mouth_openness(pose_results.face_landmarks)

    is_thinking = False

    if right_finger is not None:
        distance_to_mouth = calculate_distance(right_finger[:2], right_mouth)
        if distance_to_mouth < TOUCH_THRESHOLD and right_finger[1] < shoulder_y:
            detected_pose = "thinking"
            should_flip = True
            is_thinking = True

    if left_finger is not None and not is_thinking:
        distance_to_mouth = calculate_distance(left_finger[:2], left_mouth)
        if distance_to_mouth < TOUCH_THRESHOLD and left_finger[1] < shoulder_y:
            detected_pose = "thinking"
            should_flip = False
            is_thinking = True

    if not is_thinking:
        if right_finger is not None and right_finger[1] < nose_position[1] - POINTING_OFFSET:
            detected_pose = "pointing"
            should_flip = False
        elif left_finger is not None and left_finger[1] < nose_position[1] - POINTING_OFFSET:
            detected_pose = "pointing"
            should_flip = True

    if detected_pose == "staring" and mouth_ratio > MOUTH_OPEN_THRESHOLD:
        detected_pose = "shocked"

    return detected_pose, should_flip


def draw_debug_overlay(canvas, pose_results, frame_width, frame_height, mouth_ratio):
    cv2.putText(canvas, f"mouth:{mouth_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if pose_results.face_landmarks is not None:
        face_lm = pose_results.face_landmarks.landmark
        upper_lip_pos = (int(face_lm[13].x * frame_width), int(face_lm[13].y * frame_height))
        lower_lip_pos = (int(face_lm[14].x * frame_width), int(face_lm[14].y * frame_height))
        cv2.circle(canvas, upper_lip_pos, 5, (0, 255, 0), -1)
        cv2.circle(canvas, lower_lip_pos, 5, (0, 0, 255), -1)

    right_finger = get_finger_position(pose_results.right_hand_landmarks)
    left_finger = get_finger_position(pose_results.left_hand_landmarks)

    if right_finger is not None:
        pos = (int(right_finger[0] * frame_width), int(right_finger[1] * frame_height))
        cv2.circle(canvas, pos, 10, (0, 255, 0), -1)

    if left_finger is not None:
        pos = (int(left_finger[0] * frame_width), int(left_finger[1] * frame_height))
        cv2.circle(canvas, pos, 10, (255, 0, 255), -1)


def draw_skeleton_overlay(canvas, pose_results, drawing_utils):
    body_spec = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2)
    hand_spec = drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

    drawing_utils.draw_landmarks(
        canvas, pose_results.pose_landmarks,
        mp.solutions.holistic.POSE_CONNECTIONS, body_spec, body_spec
    )
    drawing_utils.draw_landmarks(
        canvas, pose_results.right_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS, hand_spec, hand_spec
    )
    drawing_utils.draw_landmarks(
        canvas, pose_results.left_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS, hand_spec, hand_spec
    )


def draw_header_ui(canvas, state, frame_width, current_pose):
    header_height = 50
    cv2.rectangle(canvas, (0, 0), (frame_width, header_height), (10, 10, 10), -1)
    cv2.line(canvas, (0, header_height), (frame_width, header_height), (40, 40, 40), 1)

    title = "MEME MIRROR"
    if state.chaos_mode:
        title = apply_glitch_effect(title)

    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    if state.chaos_mode:
        brightness = random.randint(180, 255)
    else:
        brightness = int(200 + 55 * state.ui_pulse)
    title_color = (brightness, brightness, brightness)

    title_x = frame_width // 2 - title_w // 2
    cv2.putText(canvas, title, (title_x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)

    cv2.line(canvas, (title_x - 20, 38), (title_x - 5, 38), (60, 60, 60), 1)
    cv2.line(canvas, (title_x + title_w + 5, 38), (title_x + title_w + 20, 38), (60, 60, 60), 1)

    draw_corner_brackets(canvas, 8, 10, 38, 40, (100, 100, 100), corner_length=6, thickness=1)
    cv2.line(canvas, (15, 17), (31, 33), (150, 150, 150), 1)
    cv2.line(canvas, (31, 17), (15, 33), (150, 150, 150), 1)

    pose_text = current_pose.upper()
    if state.chaos_mode:
        pose_text = apply_glitch_effect(pose_text)

    (pose_w, _), _ = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pose_x = frame_width - pose_w - 18

    pose_color = POSE_COLORS[current_pose]
    glow = int(40 * state.ui_pulse)
    bg_color = (pose_color[0] // 4 + glow, pose_color[1] // 4 + glow, pose_color[2] // 4 + glow)

    cv2.rectangle(canvas, (pose_x - 12, 12), (frame_width - 8, 40), bg_color, -1)
    draw_corner_brackets(canvas, pose_x - 12, 12, frame_width - 8, 40,
                         pose_color, corner_length=5, thickness=1)
    cv2.putText(canvas, pose_text, (pose_x - 2, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return header_height


def draw_footer_ui(canvas, state, frame_width, frame_height, current_pose):
    bar_progress = (state.pose_counts[current_pose] % 100) / 100
    bar_width = int(bar_progress * (frame_width - 16))
    cv2.rectangle(canvas, (8, frame_height - 4), (8 + bar_width, frame_height),
                  POSE_COLORS[current_pose], -1)
    cv2.rectangle(canvas, (8, frame_height - 4), (frame_width - 8, frame_height), (40, 40, 40), 1)

    draw_corner_brackets(canvas, 2, 2, frame_width - 2, frame_height - 2,
                         (50, 50, 50), corner_length=15, thickness=1)


def draw_streak_indicator(canvas, state, frame_width, header_height):
    if state.streak <= 0:
        return

    streak_text = f"x{state.streak}"
    (text_w, _), _ = cv2.getTextSize(streak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    streak_intensity = min(1.0, state.streak / 10)
    streak_color = (0, int(255 * streak_intensity), int(255 * (1 - streak_intensity * 0.5)))

    box_x1 = frame_width - text_w - 22
    box_y1 = header_height + 5
    box_x2 = frame_width - 8
    box_y2 = header_height + 30

    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
    draw_corner_brackets(canvas, box_x1, box_y1, box_x2, box_y2,
                         streak_color, corner_length=4, thickness=1)
    cv2.putText(canvas, streak_text, (frame_width - text_w - 15, header_height + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, streak_color, 1)


def draw_roast_message(canvas, state, frame_width, frame_height):
    time_since_roast = time.time() - state.roast_timer
    if time_since_roast >= 3 or not state.last_roast:
        return

    fade_factor = 1.0 - time_since_roast / 3
    (text_w, _), _ = cv2.getTextSize(state.last_roast, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)

    text_x = frame_width // 2 - text_w // 2
    text_y = frame_height // 2

    box_color = (int(200 * fade_factor), int(50 * fade_factor), int(50 * fade_factor))
    text_brightness = int(255 * fade_factor)

    cv2.rectangle(canvas, (text_x - 15, text_y - 22), (text_x + text_w + 15, text_y + 8),
                  (10, 10, 10), -1)
    draw_corner_brackets(canvas, text_x - 15, text_y - 22, text_x + text_w + 15, text_y + 8,
                         box_color, corner_length=6, thickness=1)
    cv2.putText(canvas, state.last_roast, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (text_brightness,) * 3, 1)


def compose_output_frame(camera_view, meme_view, state, frame_width, frame_height):
    if state.vertical_mode:
        vertical_width = min(frame_width, 540)
        vertical_height = int(vertical_width * 16 / 9)
        camera_resized = cv2.resize(camera_view, (vertical_width, vertical_height // 2))
        meme_resized = cv2.resize(meme_view, (vertical_width, vertical_height // 2))
        return np.vstack([camera_resized, meme_resized])

    if state.meme_only:
        return cv2.resize(meme_view, (frame_width, frame_height))

    if state.side_by_side:
        camera_half = cv2.resize(camera_view, (frame_width // 2, frame_height))
        meme_half = cv2.resize(meme_view, (frame_width // 2, frame_height))
        return np.hstack([camera_half, meme_half])

    if state.pip_mode:
        output = cv2.resize(meme_view, (frame_width, frame_height))
        pip_width = frame_width // 4
        pip_height = frame_height // 4
        pip_view = cv2.resize(camera_view, (pip_width, pip_height))
        pip_x = frame_width - pip_width - 10
        output[10 : 10 + pip_height, pip_x : pip_x + pip_width] = pip_view
        cv2.rectangle(output, (pip_x, 10), (frame_width - 10, 10 + pip_height), (255, 255, 255), 2)
        return output

    camera_half_height = cv2.resize(camera_view, (frame_width, frame_height // 2))
    return np.vstack([camera_half_height, meme_view])


def draw_status_indicators(output, state):
    output_height, output_width = output.shape[:2]

    fps_value = 1 / (time.time() - state.frame_timestamp + 0.001)
    fps_text = f"{int(fps_value)}"
    cv2.putText(output, fps_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(output, "fps", (10 + len(fps_text) * 9, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (35, 35, 35), 1)

    indicator_y = 28

    if state.manual_pose_index >= 0:
        cv2.putText(output, "MANUAL", (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 140, 50), 1)
        indicator_y += 16

    if state.chaos_mode:
        chaos_text = apply_glitch_effect("CHAOS") if random.random() > 0.5 else "CHAOS"
        cv2.putText(output, chaos_text, (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 50, 200), 1)
        indicator_y += 16

    if state.ego_deaths > 0:
        ego_text = f"{state.ego_deaths} ego deaths"
        cv2.putText(output, ego_text, (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 40, 100), 1)

    session_minutes = int((time.time() - state.session_start) / 60)
    if session_minutes >= 5:
        shame_messages = ["wasting time", "go outside", "intervention needed"]
        shame_index = min(2, (session_minutes - 5) // 5)
        shame_text = f"{shame_messages[shame_index]} ({session_minutes}m)"
        cv2.putText(output, shame_text, (output_width - 130, output_height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 40, 40), 1)


def draw_mode_badges(output, state):
    output_height, output_width = output.shape[:2]
    badge_y = 45

    if state.vertical_mode:
        cv2.rectangle(output, (output_width - 85, badge_y - 14), (output_width - 8, badge_y + 4),
                      (80, 40, 120), -1)
        cv2.putText(output, "VERTICAL", (output_width - 80, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 255), 1)
        badge_y += 22

    if state.face_crop:
        cv2.rectangle(output, (output_width - 95, badge_y - 14), (output_width - 8, badge_y + 4),
                      (40, 100, 80), -1)
        cv2.putText(output, "FACE CROP", (output_width - 90, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 200), 1)
        badge_y += 22

    if state.watermark:
        cv2.rectangle(output, (output_width - 55, badge_y - 14), (output_width - 8, badge_y + 4),
                      (60, 60, 60), -1)
        cv2.putText(output, "WM", (output_width - 45, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        badge_y += 22

    if state.smooth_transition:
        cv2.putText(output, "~smooth", (output_width - 65, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)


def draw_watermark(output, state):
    if not state.watermark:
        return

    output_height, output_width = output.shape[:2]
    watermark_text = "MEME MIRROR"
    (text_w, _), _ = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    text_x = output_width - text_w - 15
    text_y = output_height - 35

    cv2.putText(output, watermark_text, (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(output, watermark_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_recording_indicator(output, state):
    if not state.recording:
        return

    output_height, output_width = output.shape[:2]
    pulse_value = int(180 + 75 * math.sin(state.frame_count * 0.15))

    cv2.circle(output, (output_width - 20, 20), 6, (0, 0, pulse_value), -1)
    cv2.putText(output, "rec", (output_width - 50, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, pulse_value), 1)


def draw_help_button(output, state):
    output_height, output_width = output.shape[:2]

    cv2.rectangle(output, (12, output_height - 24), (55, output_height - 6), (0, 0, 0), -1)
    cv2.rectangle(output, (12, output_height - 24), (55, output_height - 6), (0, 200, 255), 1)
    cv2.putText(output, "HELP", (17, output_height - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)


def draw_help_menu(output, state):
    if not state.show_help:
        return

    output_height, output_width = output.shape[:2]

    commands = [
        "Q - quit", "H - toggle help", "A - face crop", "V - vertical mode",
        "W - watermark", "N - smooth transition", "S - screenshot", "R - record",
        "G - save gif", "F - fullscreen", "P - pause", "M - meme only",
        "B - side by side", "I - pip mode", "Z - zoom face",
        "X - chaos mode", "D - debug", "C - switch camera",
    ]

    menu_width = 180
    menu_height = len(commands) * 18 + 20
    menu_x = 12
    menu_y = output_height - 30 - menu_height

    cv2.rectangle(output, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                  (0, 0, 0), -1)
    cv2.rectangle(output, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                  (0, 200, 255), 1)

    for i, command in enumerate(commands):
        cv2.putText(output, command, (menu_x + 10, menu_y + 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def handle_keypress(key, state, output, capture, pose_list):
    if key == ord("q"):
        return False

    if key == ord("h"):
        state.show_help = not state.show_help
    elif key == ord("d"):
        state.debug_mode = not state.debug_mode
    elif key == ord("s"):
        filename = f"screenshots/meme_{int(time.time())}.png"
        cv2.imwrite(filename, output)
        state.flash_intensity = 255
    elif key == ord("f"):
        state.fullscreen = not state.fullscreen
    elif key == ord("p"):
        state.paused = not state.paused
    elif key == ord("m"):
        state.meme_only = not state.meme_only
    elif key == ord("b"):
        state.side_by_side = not state.side_by_side
        state.pip_mode = False
    elif key == ord("i"):
        state.pip_mode = not state.pip_mode
        state.side_by_side = False
    elif key == ord("z"):
        state.zoom_face = not state.zoom_face
    elif key == ord("x"):
        state.chaos_mode = not state.chaos_mode
    elif key == ord("w"):
        state.watermark = not state.watermark
    elif key == ord("v"):
        state.vertical_mode = not state.vertical_mode
    elif key == ord("n"):
        state.smooth_transition = not state.smooth_transition
    elif key == ord("a"):
        state.face_crop = not state.face_crop
    elif key == ord("c"):
        state.camera_index = (state.camera_index + 1) % 3
        capture.release()
        capture.open(state.camera_index)
    elif key == ord("r"):
        state.recording = not state.recording
        if state.recording:
            output_height, output_width = output.shape[:2]
            filename = f"recordings/meme_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            state.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (output_width, output_height))
        elif state.video_writer is not None:
            state.video_writer.release()
            state.video_writer = None
    elif key == ord("g"):
        for i, frame in enumerate(state.gif_buffer):
            cv2.imwrite(f"screenshots/gif_{i:03d}.png", frame)
        state.flash_intensity = 255
    elif key == ord("t"):
        state.last_roast = random.choice(ROAST_MESSAGES)
        state.roast_timer = time.time()
    elif key in [81, 2]:
        if state.manual_pose_index >= 0:
            state.manual_pose_index = (state.manual_pose_index - 1) % len(pose_list)
        else:
            state.manual_pose_index = len(pose_list) - 1
    elif key in [83, 3]:
        state.manual_pose_index = (state.manual_pose_index + 1) % len(pose_list)
    elif key in [82, 0]:
        state.manual_pose_index = -1

    return True


def main():
    setup_directories()
    meme_images = load_meme_images()
    pose_list = list(MEME_PATHS.keys())

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Cannot open camera")
        return

    if not show_splash_screen(meme_images):
        capture.release()
        cv2.destroyAllWindows()
        return

    pose_detector = create_pose_detector()
    drawing_utils = mp.solutions.drawing_utils
    state = AppState()

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        if state.paused:
            cv2.imshow(WINDOW_NAME, state.last_output if hasattr(state, "last_output") else frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("p"):
                state.paused = False
            if key == ord("q"):
                break
            continue

        sized_memes = {}
        for pose_name, image in meme_images.items():
            resized = cv2.resize(image, (frame_width, frame_height // 2))
            sized_memes[pose_name] = {
                "normal": resized,
                "flipped": cv2.flip(resized, 1),
            }

        canvas = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(rgb_frame)

        detected_pose, should_flip = detect_current_pose(pose_results, frame_width, frame_height)

        if state.debug_mode and pose_results.pose_landmarks is not None:
            mouth_ratio = calculate_mouth_openness(pose_results.face_landmarks)
            draw_debug_overlay(canvas, pose_results, frame_width, frame_height, mouth_ratio)

        if detected_pose == "staring" and pose_results.pose_landmarks is not None:
            draw_skeleton_overlay(canvas, pose_results, drawing_utils)

        state.pose_history.append(detected_pose)
        if len(state.pose_history) > 5:
            state.pose_history.pop(0)

        stable_pose = detected_pose
        if len(state.pose_history) >= 2:
            if state.pose_history.count(state.pose_history[-1]) >= 2:
                stable_pose = state.pose_history[-1]

        state.pose_counts[stable_pose] += 1
        total_poses = sum(state.pose_counts.values())

        if stable_pose == "staring" and total_poses > 100:
            state.cringe_score += 0.1
        elif stable_pose != "staring":
            state.cringe_score = max(0, state.cringe_score - 0.5)

        if state.cringe_score > 50 and random.random() < 0.02:
            state.ego_deaths += 1
            state.cringe_score = 0
            state.last_roast = random.choice(ROAST_MESSAGES)
            state.roast_timer = time.time()

        if state.manual_pose_index >= 0:
            display_pose = pose_list[state.manual_pose_index]
        else:
            display_pose = stable_pose

        meme_variant = "flipped" if should_flip else "normal"
        current_meme = sized_memes[display_pose][meme_variant]

        if state.chaos_mode:
            current_meme = apply_corruption_effect(current_meme.copy(), 0.3)
            if random.random() < 0.1:
                current_meme = cv2.flip(current_meme, random.choice([-1, 0, 1]))

        if stable_pose != "staring":
            if stable_pose == state.last_detected_pose:
                state.streak += 1
            else:
                state.streak = 1
        else:
            state.streak = 0
        state.last_detected_pose = stable_pose

        if state.smooth_transition:
            if state.current_meme is None or state.current_meme.shape != current_meme.shape:
                state.current_meme = current_meme.copy()
                state.transition_alpha = 1.0
            elif not np.array_equal(state.current_meme, sized_memes[display_pose][meme_variant]):
                state.transition_alpha = max(0.0, state.transition_alpha - 0.08)
                if state.transition_alpha <= 0:
                    state.current_meme = sized_memes[display_pose][meme_variant].copy()
                    state.transition_alpha = 1.0
                else:
                    current_meme = cv2.addWeighted(
                        state.current_meme, state.transition_alpha,
                        current_meme, 1 - state.transition_alpha, 0
                    )
        else:
            if state.previous_meme is not None and state.previous_meme.shape == current_meme.shape:
                if state.blend_alpha < 1.0:
                    state.blend_alpha = min(1.0, state.blend_alpha + 0.12)
                    current_meme = cv2.addWeighted(
                        state.previous_meme, 1 - state.blend_alpha,
                        current_meme, state.blend_alpha, 0
                    )
                elif not np.array_equal(state.previous_meme, current_meme):
                    state.blend_alpha = 0.0
                    state.flash_intensity = 180
        state.previous_meme = sized_memes[display_pose][meme_variant].copy()

        if state.face_crop and pose_results.face_landmarks is not None:
            face_bbox = get_face_bounding_box(pose_results.face_landmarks, frame_width, frame_height)
            meme_height, meme_width = current_meme.shape[:2]
            face_target_regions = {
                "thinking": (int(meme_width * 0.25), int(meme_height * 0.08),
                             int(meme_width * 0.75), int(meme_height * 0.58)),
                "pointing": (int(meme_width * 0.2), int(meme_height * 0.08),
                             int(meme_width * 0.65), int(meme_height * 0.55)),
                "shocked": (int(meme_width * 0.25), int(meme_height * 0.05),
                            int(meme_width * 0.75), int(meme_height * 0.55)),
                "staring": (int(meme_width * 0.25), int(meme_height * 0.08),
                            int(meme_width * 0.75), int(meme_height * 0.58)),
            }
            if display_pose in face_target_regions:
                current_meme = blend_face_onto_meme(
                    frame, current_meme, face_bbox, face_target_regions[display_pose]
                )

        if state.flash_intensity > 0:
            flash_overlay = np.full(canvas.shape, state.flash_intensity, dtype=np.uint8)
            canvas = cv2.addWeighted(canvas, 1, flash_overlay, 0.3, 0)
            state.flash_intensity = max(0, state.flash_intensity - 40)

        if state.chaos_mode:
            canvas = apply_corruption_effect(canvas, 0.2)

        state.frame_count += 1
        state.ui_pulse = math.sin(state.frame_count * 0.05) * 0.5 + 0.5

        camera_view = canvas.copy()

        if state.zoom_face and pose_results.pose_landmarks is not None:
            nose = pose_results.pose_landmarks.landmark[0]
            center_x = int(nose.x * frame_width)
            center_y = int(nose.y * frame_height)
            zoom_size = min(frame_width, frame_height) // 2

            crop_x1 = max(0, center_x - zoom_size // 2)
            crop_y1 = max(0, center_y - zoom_size // 2)
            crop_x2 = min(frame_width, crop_x1 + zoom_size)
            crop_y2 = min(frame_height, crop_y1 + zoom_size)

            if crop_x2 - crop_x1 > 50 and crop_y2 - crop_y1 > 50:
                camera_view = cv2.resize(canvas[crop_y1:crop_y2, crop_x1:crop_x2],
                                         (frame_width, frame_height))

        header_height = draw_header_ui(camera_view, state, frame_width, display_pose)
        draw_footer_ui(camera_view, state, frame_width, frame_height, display_pose)
        draw_streak_indicator(camera_view, state, frame_width, header_height)
        draw_roast_message(camera_view, state, frame_width, frame_height)

        output = compose_output_frame(camera_view, current_meme, state, frame_width, frame_height)

        state.frame_timestamp = time.time()

        if not state.meme_only:
            draw_scanline_overlay(output, line_spacing=3, overlay_alpha=0.02)

        draw_status_indicators(output, state)
        draw_mode_badges(output, state)
        draw_watermark(output, state)

        output_height, output_width = output.shape[:2]
        draw_corner_brackets(output, 4, output_height - 28, output_width - 4, output_height - 4,
                             (40, 40, 40), corner_length=8, thickness=1)

        draw_recording_indicator(output, state)
        draw_help_button(output, state)
        draw_help_menu(output, state)

        if state.recording and state.video_writer is not None:
            state.video_writer.write(output)

        state.gif_buffer.append(output.copy())
        state.last_output = output

        if state.fullscreen:
            cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(5) & 0xFF
        if not handle_keypress(key, state, output, capture, pose_list):
            break

    capture.release()
    if state.video_writer is not None:
        state.video_writer.release()
    cv2.destroyAllWindows()
    pose_detector.close()


if __name__ == "__main__":
    main()