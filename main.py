import cv2, mediapipe as mp, numpy as np, time, random, os
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

memes = {
    "thinking": "memes/thinking.jpg",
    "pointing": "memes/pointing.jpg",
    "shocked": "memes/shocked.jpg",
    "staring": "memes/staring.jpg"
}
colors = {"staring": (128,128,128), "thinking": (0,165,255), "pointing": (0,255,0), "shocked": (0,0,255)}
poses = list(memes.keys())

holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.6, min_tracking_confidence=0.6,
    smooth_landmarks=True, model_complexity=1, refine_face_landmarks=True
)
draw = mp.solutions.drawing_utils
body_style = draw.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=4)
hand_style = draw.DrawingSpec(color=(0,255,255), thickness=4, circle_radius=3)

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
    
    for i in range(80):
        splash[:] = (0, 0, 0)
        
        pulse = abs((i % 20) - 10) / 10
        red = int(255 * pulse)
        
        (ww, wh), _ = cv2.getTextSize("WARNING", cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)
        cv2.putText(splash, "WARNING", (W//2 - ww//2, H//2 - 50), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, red), 5)
        
        sub = "FLASHING LIGHTS / SEIZURE"
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(splash, sub, (W//2 - sw//2, H//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(splash, (W//2 - 280, H//2 - 130), (W//2 + 280, H//2 + 50), (0, 0, red), 3)
        
        hint = "Q = quit  /  SPACE = continue"
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(splash, hint, (W//2 - hw//2, H//2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        bar_w = int((i / 80) * 300)
        cv2.rectangle(splash, (W//2 - 150, H//2 + 160), (W//2 - 150 + bar_w, H//2 + 175), (60, 60, 60), -1)
        cv2.rectangle(splash, (W//2 - 150, H//2 + 160), (W//2 + 150, H//2 + 175), (60, 60, 60), 1)
        
        cv2.imshow('Meme Mirror', splash)
        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'): return False
        if key == ord(' ') and i > 15: break
    
    phrases = [
        "this isnt a filter",
        "ur cringe is loading",
        "no refunds",
        "pose or die", 
        "not our fault",
        "touch grass after",
        "legal said no",
        "powered by regret",
    ]
    
    for i in range(90):
        splash[:] = (0, 0, 0)
        
        if i < 25:
            scale = 0.5 + (i / 25) * 3.5
            thick = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_DUPLEX, scale, thick)
            cv2.putText(splash, "MEME", (W//2 - tw//2, H//2 + th//2), cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), thick)
        
        elif i < 45:
            offset = int(3 * np.sin(i * 0.5))
            (mw, mh), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_DUPLEX, 3.5, 8)
            (rw, rh), _ = cv2.getTextSize("MIRROR", cv2.FONT_HERSHEY_DUPLEX, 3.5, 8)
            mx, rx = W//2 - mw//2, W//2 - rw//2
            cv2.putText(splash, "MEME", (mx + offset, H//2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255,0,0), 8)
            cv2.putText(splash, "MEME", (mx - offset, H//2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0,255,255), 8)
            cv2.putText(splash, "MEME", (mx, H//2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255,255,255), 8)
            cv2.putText(splash, "MIRROR", (rx + offset, H//2 + 130), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255,0,0), 8)
            cv2.putText(splash, "MIRROR", (rx - offset, H//2 + 130), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0,255,255), 8)
            cv2.putText(splash, "MIRROR", (rx, H//2 + 130), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255,255,255), 8)
        
        elif i < 60:
            bg = 255 if (i % 4 < 2) else 0
            fg = 0 if bg else 255
            splash[:] = (bg, bg, bg)
            (mw, mh), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_DUPLEX, 3.5, 10)
            (rw, rh), _ = cv2.getTextSize("MIRROR", cv2.FONT_HERSHEY_DUPLEX, 3.5, 10)
            cv2.putText(splash, "MEME", (W//2 - mw//2, H//2 + 30), cv2.FONT_HERSHEY_DUPLEX, 3.5, (fg,fg,fg), 10)
            cv2.putText(splash, "MIRROR", (W//2 - rw//2, H//2 + 130), cv2.FONT_HERSHEY_DUPLEX, 3.5, (fg,fg,fg), 10)
            cv2.rectangle(splash, (W//2 - 250, H//2 - 40), (W//2 + 250, H//2 + 180), (fg,fg,fg), 5)
        
        elif i < 80:
            phrase = phrases[(i-60) % len(phrases)]
            (pw, ph), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(splash, phrase, (W//2 - pw//2, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            
            pct = (i - 60) / 20
            bar_w = int(pct * 500)
            cv2.rectangle(splash, (W//2 - 250, H//2 + 80), (W//2 - 250 + bar_w, H//2 + 105), (255,255,255), -1)
            cv2.rectangle(splash, (W//2 - 250, H//2 + 80), (W//2 + 250, H//2 + 105), (255,255,255), 2)
            cv2.putText(splash, f"{int(pct*100)}%", (W//2 - 20, H//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0) if pct > 0.4 else (255,255,255), 2)
        
        else:
            flash_v = 255 if (i % 2) else 0
            splash[:] = (flash_v, flash_v, flash_v)
            (gw, gh), _ = cv2.getTextSize("GO", cv2.FONT_HERSHEY_DUPLEX, 5, 15)
            cv2.putText(splash, "GO", (W//2 - gw//2, H//2 + gh//2), cv2.FONT_HERSHEY_DUPLEX, 5, (0,0,0) if flash_v else (255,255,255), 15)
        
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

    show_pose = stable if manual_idx < 0 else poses[manual_idx]
    ver = "flipped" if flip else "normal"
    meme = sized[show_pose][ver]
    
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
    
    cam = canvas.copy()
    if zoom and res.pose_landmarks:
        px, py = int(res.pose_landmarks.landmark[0].x * w), int(res.pose_landmarks.landmark[0].y * h)
        sz = min(w, h) // 2
        x1, y1 = max(0, px - sz//2), max(0, py - sz//2)
        x2, y2 = min(w, x1 + sz), min(h, y1 + sz)
        if x2-x1 > 50 and y2-y1 > 50:
            cam = cv2.resize(canvas[y1:y2, x1:x2], (w, h))
    
    cv2.rectangle(cam, (0, 0), (w, 55), (0,0,0), -1)
    title = "MEME MIRROR"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    cv2.putText(cam, title, (w//2 - tw//2, 28 + th//2), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255), 2)
    
    cv2.rectangle(cam, (10, 12), (42, 44), (0,0,0), -1)
    cv2.rectangle(cam, (10, 12), (42, 44), (255,255,255), 2)
    cv2.line(cam, (16, 18), (36, 38), (255,255,255), 2)
    cv2.line(cam, (36, 18), (16, 38), (255,255,255), 2)
    
    pose_txt = show_pose.upper()
    (pw, ph), _ = cv2.getTextSize(pose_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pose_x = w - pw - 25
    cv2.rectangle(cam, (pose_x - 8, 12), (w - 10, 44), colors[show_pose], -1)
    cv2.rectangle(cam, (pose_x - 8, 12), (w - 10, 44), (255,255,255), 2)
    cv2.putText(cam, pose_txt, (pose_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    cv2.rectangle(cam, (0, h-10), (w, h), colors[show_pose], -1)
    cv2.rectangle(cam, (1, 1), (w-1, h-1), (255,255,255), 2)
    
    if streak > 0:
        streak_txt = f"x{streak}"
        (sw, sh), _ = cv2.getTextSize(streak_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(cam, (w - sw - 20, 52), (w - 10, 82), (0,0,0), -1)
        cv2.rectangle(cam, (w - sw - 20, 52), (w - 10, 82), (0,255,255), 2)
        cv2.putText(cam, streak_txt, (w - sw - 15, 74), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
    
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
    fps_txt = f"{int(fps)}fps"
    (fw, fh), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (5, 5), (fw + 15, 30), (0,0,0), -1)
    cv2.putText(out, fps_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    if manual_idx >= 0:
        (mw, mh), _ = cv2.getTextSize("MANUAL", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (5, 33), (mw + 15, 58), (255,165,0), -1)
        cv2.putText(out, "MANUAL", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    if recording:
        cv2.circle(out, (out.shape[1]-30, 30), 10, (0,0,255), -1)
        cv2.putText(out, "REC", (out.shape[1]-70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        if writer: writer.write(out)
    
    gif_buf.append(out.copy())
    
    oh, ow = out.shape[:2]
    
    quit_txt = "QUIT"
    (qw, qh), _ = cv2.getTextSize(quit_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    qx, qy = ow//2 - qw//2 - 8, oh - 35
    cv2.rectangle(out, (qx, qy), (qx + qw + 16, oh - 10), (0,0,0), -1)
    cv2.rectangle(out, (qx, qy), (qx + qw + 16, oh - 10), (255,255,255), 2)
    cv2.putText(out, quit_txt, (qx + 8, oh - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    
    hints = ["Q to quit", "vibes only", "no rules", "figure it out", "keys r cool", "gl hf"]
    hint = hints[int(time.time()) % len(hints)]
    cv2.putText(out, hint, (10, oh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,60,60), 1)
    
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
    elif key in [81, 2]: manual_idx = (manual_idx - 1) % len(poses) if manual_idx >= 0 else len(poses) - 1
    elif key in [83, 3]: manual_idx = (manual_idx + 1) % len(poses)
    elif key in [82, 0]: manual_idx = -1

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
holistic.close()