import cv2, mediapipe as mp, numpy as np, time, random, os, math
from collections import deque

debug = fullscreen = meme_only = paused = side_by_side = pip_mode = zoom = False
recording = False
streak = 0
manual_idx = -1
cam_idx = 0
prev_meme = None
alpha = 1.0
flash = 0
writer = None
gif_buf = deque(maxlen=90)
history = []
last_pose = "staring"
t = time.time()
session_start = time.time()
pose_counts = {"thinking": 0, "pointing": 0, "shocked": 0, "staring": 0}
cringe_score = 0
ego_deaths = 0
existential_mode = False
chaos_mode = False
last_roast = ""
roast_timer = 0
frame_count = 0
ui_pulse = 0

roasts = [
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

existential_thoughts = [
    "why are we performing for machines",
    "is this what we evolved for",
    "the meme looks back at you",
    "you are the content now",
    "surveillance capitalism but make it fun",
    "your face is training data",
    "pose for the void",
    "the skeleton was inside you all along",
    "meaning is a social construct",
    "we live in a simulation of a simulation",
]

memes = {
    "thinking": "memes/thinking.jpg",
    "pointing": "memes/pointing.jpg",
    "shocked": "memes/shocked.jpg",
    "staring": "memes/staring.jpg"
}
colors = {"staring": (90,90,90), "thinking": (0,180,255), "pointing": (0,255,120), "shocked": (80,80,255)}
poses = list(memes.keys())

holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.6, min_tracking_confidence=0.6,
    smooth_landmarks=True, model_complexity=1, refine_face_landmarks=True
)
draw = mp.solutions.drawing_utils
body_style = draw.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=2)
hand_style = draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)

def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))

def finger_pos(hand, idx=8):
    if not hand: return None
    p = hand.landmark[idx]
    return [p.x, p.y, getattr(p, 'visibility', 1.0)]

def mouth_open(face):
    if not face: return 0
    lm = face.landmark
    vert = dist([lm[13].x, lm[13].y], [lm[14].x, lm[14].y])
    horiz = dist([lm[78].x, lm[78].y], [lm[308].x, lm[308].y])
    return vert / horiz if horiz else 0

def glitch_text(txt):
    return ''.join(c if random.random() > 0.15 else random.choice("@#$%&*!?") for c in txt)

def corrupt_frame(frame, intensity=0.1):
    if random.random() < intensity:
        h, w = frame.shape[:2]
        y1, y2 = random.randint(0, h-20), random.randint(0, h)
        if y1 < y2:
            shift = random.randint(-30, 30)
            frame[y1:y2] = np.roll(frame[y1:y2], shift, axis=1)
    return frame

def draw_minimal_box(img, x1, y1, x2, y2, color, corner_len=8, thickness=2):
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

def draw_scanlines(img, spacing=4, alpha=0.03):
    overlay = img.copy()
    h = img.shape[0]
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (img.shape[1], y), (0,0,0), 1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def breathing_pulse(base, amplitude=20, speed=2):
    return int(base + amplitude * math.sin(time.time() * speed))

imgs = {}
for k, v in memes.items():
    im = cv2.imread(v)
    if im is None: print(f"cant load {v}"); exit()
    imgs[k] = im

os.makedirs("screenshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
cap = cv2.VideoCapture(cam_idx)

def splash_screen():
    W, H = 800, 600
    splash = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.namedWindow('Meme Mirror', cv2.WINDOW_NORMAL)
    
    for i in range(45):
        splash[:] = (8, 8, 8)
        
        for y in range(0, H, 4):
            alpha = 0.015
            splash[y:y+1, :] = np.clip(splash[y:y+1, :].astype(float) * (1 - alpha), 0, 255).astype(np.uint8)
        
        pulse = (math.sin(i * 0.25) + 1) / 2
        red = int(180 + 75 * pulse)
        
        (ww, wh), _ = cv2.getTextSize("WARNING", cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        cv2.putText(splash, "WARNING", (W//2 - ww//2, H//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, red), 2)
        
        sub = "flashing lights / photosensitive seizure"
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(splash, sub, (W//2 - sw//2, H//2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        
        draw_minimal_box(splash, W//2 - 240, H//2 - 75, W//2 + 240, H//2 + 45, (0, 0, red), corner_len=20, thickness=2)
        
        hint = "[ Q ] quit    [ SPACE ] continue"
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(splash, hint, (W//2 - hw//2, H//2 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        
        cv2.imshow('Meme Mirror', splash)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'): return False
        if key == ord(' ') and i > 8: break
    
    for i in range(80):
        splash[:] = (5, 5, 5)
        draw_scanlines(splash, spacing=3, alpha=0.02)
        
        if i < 25:
            t = i / 25
            scale = 0.3 + t * 1.2
            alpha = min(1, t * 2)
            (tw, th), _ = cv2.getTextSize("MEME MIRROR", cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            col = int(255 * alpha)
            cv2.putText(splash, "MEME MIRROR", (W//2 - tw//2, H//2 + th//2), cv2.FONT_HERSHEY_SIMPLEX, scale, (col,col,col), 2)
        
        elif i < 50:
            phase = (i - 25) / 25
            offset = int(4 * math.sin(phase * math.pi * 4)) if i < 40 else 0
            
            (mw, mh), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)
            (rw, rh), _ = cv2.getTextSize("MIRROR", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)
            
            if i < 38:
                cv2.putText(splash, "MEME", (W//2 - mw//2 + offset, H//2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,80,80), 3)
                cv2.putText(splash, "MEME", (W//2 - mw//2 - offset, H//2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80,255,255), 3)
            cv2.putText(splash, "MEME", (W//2 - mw//2, H//2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,255,255), 3)
            
            if i < 38:
                cv2.putText(splash, "MIRROR", (W//2 - rw//2 + offset, H//2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,80,80), 3)
                cv2.putText(splash, "MIRROR", (W//2 - rw//2 - offset, H//2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80,255,255), 3)
            cv2.putText(splash, "MIRROR", (W//2 - rw//2, H//2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,255,255), 3)
            
            draw_minimal_box(splash, W//2 - 180, H//2 - 70, W//2 + 180, H//2 + 90, (255,255,255), corner_len=15, thickness=2)
        
        elif i < 70:
            pct = (i - 50) / 20
            
            bar_y = H//2 + 30
            bar_x = W//2 - 150
            bar_w = int(pct * 300)
            
            cv2.rectangle(splash, (bar_x, bar_y), (bar_x + bar_w, bar_y + 3), (255,255,255), -1)
            draw_minimal_box(splash, bar_x - 5, bar_y - 5, bar_x + 305, bar_y + 8, (80,80,80), corner_len=4, thickness=1)
            
            loading_txt = random.choice(["loading vibes", "calibrating cringe", "summoning memes", "processing ego", "init chaos"])
            (lw, lh), _ = cv2.getTextSize(loading_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(splash, loading_txt, (W//2 - lw//2, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)
            
            pct_txt = f"{int(pct * 100)}%"
            cv2.putText(splash, pct_txt, (bar_x + 308, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1)
        
        else:
            flash_v = 255 if (i % 2 == 0) else 0
            splash[:] = (flash_v, flash_v, flash_v)
            txt = random.choice(["GO", "→", "POSE"])
            (gw, gh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
            cv2.putText(splash, txt, (W//2 - gw//2, H//2 + gh//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0) if flash_v else (255,255,255), 4)
        
        cv2.imshow('Meme Mirror', splash)
        if cv2.waitKey(45) & 0xFF == ord('q'): return False
    
    return True

if not splash_screen():
    cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    if paused and 'out' in dir():
        cv2.imshow('Meme Mirror', out)
        k = cv2.waitKey(5) & 0xFF
        if k == ord('p'): paused = False
        if k == ord('q'): break
        continue
    
    sized = {}
    for name, im in imgs.items():
        r = cv2.resize(im, (w, h//2))
        sized[name] = {"normal": r, "flipped": cv2.flip(r, 1)}
    
    canvas = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    pose = "staring"
    flip = False

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        nose = [lm[0].x, lm[0].y]
        mouth_l, mouth_r = [lm[9].x, lm[9].y], [lm[10].x, lm[10].y]
        shoulder_y = (lm[11].y + lm[12].y) / 2
        
        r_finger = finger_pos(res.right_hand_landmarks)
        l_finger = finger_pos(res.left_hand_landmarks)
        mouth_ratio = mouth_open(res.face_landmarks)
        
        if debug:
            cv2.putText(canvas, f"mouth:{mouth_ratio:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if res.face_landmarks:
                fl = res.face_landmarks.landmark
                cv2.circle(canvas, (int(fl[13].x*w), int(fl[13].y*h)), 5, (0,255,0), -1)
                cv2.circle(canvas, (int(fl[14].x*w), int(fl[14].y*h)), 5, (0,0,255), -1)
            if r_finger: cv2.circle(canvas, (int(r_finger[0]*w), int(r_finger[1]*h)), 10, (0,255,0), -1)
            if l_finger: cv2.circle(canvas, (int(l_finger[0]*w), int(l_finger[1]*h)), 10, (255,0,255), -1)
        
        got_thinking = False
        if r_finger and dist(r_finger[:2], mouth_r) < 0.08 and r_finger[1] < shoulder_y:
            pose, flip, got_thinking = "thinking", True, True
        if l_finger and not got_thinking and dist(l_finger[:2], mouth_l) < 0.08 and l_finger[1] < shoulder_y:
            pose, flip, got_thinking = "thinking", False, True
        
        if not got_thinking:
            if r_finger and r_finger[1] < nose[1] - 0.05:
                pose, flip = "pointing", False
            elif l_finger and l_finger[1] < nose[1] - 0.05:
                pose, flip = "pointing", True
        
        if pose == "staring" and mouth_ratio > 0.5:
            pose = "shocked"
        
        if pose == "staring":
            draw.draw_landmarks(canvas, res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, body_style, body_style)
            draw.draw_landmarks(canvas, res.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, hand_style, hand_style)
            draw.draw_landmarks(canvas, res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, hand_style, hand_style)

    history.append(pose)
    if len(history) > 5: history.pop(0)

    stable = pose
    if len(history) >= 2 and history.count(history[-1]) >= 2:
        stable = history[-1]
    
    pose_counts[stable] += 1
    total_poses = sum(pose_counts.values())
    
    if stable == "staring" and total_poses > 100:
        cringe_score += 0.1
    elif stable != "staring":
        cringe_score = max(0, cringe_score - 0.5)
    
    if cringe_score > 50 and random.random() < 0.02:
        ego_deaths += 1
        cringe_score = 0
        last_roast = random.choice(roasts)
        roast_timer = time.time()

    show_pose = stable if manual_idx < 0 else poses[manual_idx]
    ver = "flipped" if flip else "normal"
    meme = sized[show_pose][ver]
    
    if chaos_mode:
        meme = corrupt_frame(meme.copy(), 0.3)
        if random.random() < 0.1:
            meme = cv2.flip(meme, random.choice([-1, 0, 1]))
    
    if stable != "staring":
        streak = streak + 1 if stable == last_pose else 1
    else:
        streak = 0
    last_pose = stable
    
    if prev_meme is not None and prev_meme.shape == meme.shape:
        if alpha < 1.0:
            alpha = min(1.0, alpha + 0.12)
            meme = cv2.addWeighted(prev_meme, 1-alpha, meme, alpha, 0)
        elif not np.array_equal(prev_meme, meme):
            alpha = 0.0
            flash = 180
    prev_meme = sized[show_pose][ver].copy()
    
    if flash > 0:
        canvas = cv2.addWeighted(canvas, 1, np.full(canvas.shape, flash, dtype=np.uint8), 0.3, 0)
        flash = max(0, flash - 40)
    
    if chaos_mode:
        canvas = corrupt_frame(canvas, 0.2)
    
    frame_count += 1
    ui_pulse = math.sin(frame_count * 0.05) * 0.5 + 0.5
    
    cam = canvas.copy()
    if zoom and res.pose_landmarks:
        px, py = int(res.pose_landmarks.landmark[0].x * w), int(res.pose_landmarks.landmark[0].y * h)
        sz = min(w, h) // 2
        x1, y1 = max(0, px - sz//2), max(0, py - sz//2)
        x2, y2 = min(w, x1 + sz), min(h, y1 + sz)
        if x2-x1 > 50 and y2-y1 > 50:
            cam = cv2.resize(canvas[y1:y2, x1:x2], (w, h))
    
    header_h = 50
    cv2.rectangle(cam, (0, 0), (w, header_h), (10,10,10), -1)
    cv2.line(cam, (0, header_h), (w, header_h), (40,40,40), 1)
    
    session_time = int(time.time() - session_start)
    
    title = "MEME MIRROR"
    if chaos_mode: title = glitch_text(title)
    elif existential_mode: title = "M E M E   M I R R O R"
    
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pulse_bright = int(200 + 55 * ui_pulse) if not chaos_mode else random.randint(180, 255)
    title_color = (pulse_bright,) * 3
    cv2.putText(cam, title, (w//2 - tw//2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)
    
    cv2.line(cam, (w//2 - tw//2 - 20, 38), (w//2 - tw//2 - 5, 38), (60,60,60), 1)
    cv2.line(cam, (w//2 + tw//2 + 5, 38), (w//2 + tw//2 + 20, 38), (60,60,60), 1)
    
    draw_minimal_box(cam, 8, 10, 38, 40, (100,100,100), corner_len=6, thickness=1)
    cv2.line(cam, (15, 17), (31, 33), (150,150,150), 1)
    cv2.line(cam, (31, 17), (15, 33), (150,150,150), 1)
    
    pose_txt = show_pose.upper()
    if chaos_mode: pose_txt = glitch_text(pose_txt)
    (pw, ph), _ = cv2.getTextSize(pose_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pose_x = w - pw - 18
    
    c = colors[show_pose]
    glow = int(40 * ui_pulse)
    cv2.rectangle(cam, (pose_x - 12, 12), (w - 8, 40), (c[0]//4+glow, c[1]//4+glow, c[2]//4+glow), -1)
    draw_minimal_box(cam, pose_x - 12, 12, w - 8, 40, c, corner_len=5, thickness=1)
    cv2.putText(cam, pose_txt, (pose_x - 2, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    
    bar_width = int((pose_counts[show_pose] % 100) / 100 * (w - 16))
    cv2.rectangle(cam, (8, h - 4), (8 + bar_width, h), colors[show_pose], -1)
    cv2.rectangle(cam, (8, h - 4), (w - 8, h), (40,40,40), 1)
    
    draw_minimal_box(cam, 2, 2, w-2, h-2, (50,50,50), corner_len=15, thickness=1)
    
    if streak > 0:
        streak_txt = f"x{streak}"
        (sw, sh), _ = cv2.getTextSize(streak_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        streak_alpha = min(1.0, streak / 10)
        streak_color = (0, int(255 * streak_alpha), int(255 * (1 - streak_alpha * 0.5)))
        cv2.rectangle(cam, (w - sw - 22, header_h + 5), (w - 8, header_h + 30), (15,15,15), -1)
        draw_minimal_box(cam, w - sw - 22, header_h + 5, w - 8, header_h + 30, streak_color, corner_len=4, thickness=1)
        cv2.putText(cam, streak_txt, (w - sw - 15, header_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, streak_color, 1)
    
    if time.time() - roast_timer < 3 and last_roast:
        roast_fade = 1.0 - (time.time() - roast_timer) / 3
        (rw, rh), _ = cv2.getTextSize(last_roast, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        rx, ry = w//2 - rw//2, h//2
        cv2.rectangle(cam, (rx - 15, ry - 22), (rx + rw + 15, ry + 8), (10,10,10), -1)
        draw_minimal_box(cam, rx - 15, ry - 22, rx + rw + 15, ry + 8, (int(200*roast_fade), int(50*roast_fade), int(50*roast_fade)), corner_len=6, thickness=1)
        cv2.putText(cam, last_roast, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (int(255*roast_fade),)*3, 1)
    
    if existential_mode and random.random() < 0.015:
        thought = random.choice(existential_thoughts)
        (ew, eh), _ = cv2.getTextSize(thought, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ex = random.randint(10, max(11, w - ew - 10))
        ey = random.randint(header_h + 20, h - 30)
        cv2.putText(cam, thought, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
    
    half = cv2.resize(cam, (w, h//2))
    
    if meme_only:
        out = cv2.resize(meme, (w, h))
    elif side_by_side:
        out = np.hstack([cv2.resize(cam, (w//2, h)), cv2.resize(meme, (w//2, h))])
    elif pip_mode:
        out = cv2.resize(meme, (w, h))
        psz = (w//4, h//4)
        out[10:10+psz[1], w-psz[0]-10:w-10] = cv2.resize(cam, psz)
        cv2.rectangle(out, (w-psz[0]-10, 10), (w-10, 10+psz[1]), (255,255,255), 2)
    else:
        out = np.vstack([half, meme])
    
    fps = 1 / (time.time() - t + 0.001)
    t = time.time()
    
    if not meme_only:
        draw_scanlines(out, spacing=3, alpha=0.02)
    
    fps_txt = f"{int(fps)}"
    cv2.putText(out, fps_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)
    cv2.putText(out, "fps", (10 + len(fps_txt)*9, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (35,35,35), 1)
    
    stats_y = 28
    if manual_idx >= 0:
        cv2.putText(out, "MANUAL", (10, stats_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,140,50), 1)
        stats_y += 16
    
    if chaos_mode:
        ch_txt = glitch_text("CHAOS") if random.random() > 0.5 else "CHAOS"
        cv2.putText(out, ch_txt, (10, stats_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,50,200), 1)
        stats_y += 16
    
    if existential_mode:
        cv2.putText(out, "V O I D", (10, stats_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)
        stats_y += 16
    
    if ego_deaths > 0:
        ego_txt = f"{ego_deaths} ego deaths"
        cv2.putText(out, ego_txt, (10, stats_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,40,100), 1)
    
    session_mins = int((time.time() - session_start) / 60)
    if session_mins >= 5:
        shame = ["wasting time", "go outside", "intervention needed"][min(2, (session_mins - 5) // 5)]
        cv2.putText(out, f"{shame} ({session_mins}m)", (ow - 130, oh - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,40,40), 1)
    
    if recording:
        rec_pulse = int(180 + 75 * math.sin(frame_count * 0.15))
        cv2.circle(out, (out.shape[1]-20, 20), 6, (0,0,rec_pulse), -1)
        cv2.putText(out, "rec", (out.shape[1]-50, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,rec_pulse), 1)
        if writer: writer.write(out)
    
    gif_buf.append(out.copy())
    
    oh, ow = out.shape[:2]
    
    draw_minimal_box(out, 4, oh - 28, ow - 4, oh - 4, (40,40,40), corner_len=8, thickness=1)
    
    hints = ["q:quit", "vibes", "no rules", "figure it out", "ur watched", "smile", 
        "art probably", "e:void", "x:chaos", "touch grass", "the void gazes"]
    hint = hints[int(time.time() * 0.3) % len(hints)]
    if chaos_mode: hint = glitch_text(hint)
    cv2.putText(out, hint, (12, oh - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50,50,50), 1)
    
    cv2.putText(out, "[q]", (ow - 30, oh - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1)
    
    if fullscreen:
        cv2.namedWindow('Meme Mirror', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Meme Mirror', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow('Meme Mirror', cv2.WINDOW_NORMAL)
    
    cv2.imshow('Meme Mirror', out)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): break
    elif key == ord('d'): debug = not debug
    elif key == ord('s'):
        cv2.imwrite(f"screenshots/meme_{int(time.time())}.png", out)
        flash = 255
    elif key == ord('f'): fullscreen = not fullscreen
    elif key == ord('p'): paused = not paused
    elif key == ord('m'): meme_only = not meme_only
    elif key == ord('b'): side_by_side, pip_mode = not side_by_side, False
    elif key == ord('i'): pip_mode, side_by_side = not pip_mode, False
    elif key == ord('z'): zoom = not zoom
    elif key == ord('x'): chaos_mode = not chaos_mode
    elif key == ord('e'): existential_mode = not existential_mode
    elif key == ord('c'):
        cam_idx = (cam_idx + 1) % 3
        cap.release()
        cap = cv2.VideoCapture(cam_idx)
    elif key == ord('r'):
        recording = not recording
        if recording:
            writer = cv2.VideoWriter(f"recordings/meme_{int(time.time())}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (out.shape[1], out.shape[0]))
        elif writer:
            writer.release()
            writer = None
    elif key == ord('g'):
        for i, f in enumerate(gif_buf): cv2.imwrite(f"screenshots/gif_{i:03d}.png", f)
        flash = 255
    elif key == ord('t'):
        last_roast = random.choice(roasts)
        roast_timer = time.time()
    elif key in [81, 2]: manual_idx = (manual_idx - 1) % len(poses) if manual_idx >= 0 else len(poses) - 1
    elif key in [83, 3]: manual_idx = (manual_idx + 1) % len(poses)
    elif key in [82, 0]: manual_idx = -1

session_time = int(time.time() - session_start)
print(f"\n{'='*40}")
print(f"SESSION COMPLETE - MEME MIRROR")
print(f"{'='*40}")
print(f"Time wasted: {session_time//60}m {session_time%60}s")
print(f"Ego deaths: {ego_deaths}")
print(f"Poses struck:")
for p, c in pose_counts.items():
    if c > 0:
        print(f"  {p}: {c}")
print(f"Final cringe score: {cringe_score:.1f}")
if session_time > 600:
    print(f"\nWARNING: You spent over 10 minutes on this.")
    print(f"Consider: touching grass, calling a friend, existing offline")
print(f"{'='*40}\n")

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
holistic.close()