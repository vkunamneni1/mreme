import cv2, mediapipe as mp, numpy as np, time, random, os
from collections import deque

debug = fullscreen = meme_only = paused = side_by_side = pip_mode = zoom = False
challenge = reaction = combo = recording = False
challenge_target = reaction_target = None
challenge_time = reaction_time = 0
challenge_score = combo_score = combo_idx = streak = 0
best_reaction = float('inf')
combo_seq = ["thinking", "pointing", "shocked"]
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
        
        if pose == "staring" and mouth_ratio > 0.25:
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
            alpha = min(1.0, alpha + 0.15)
            meme = cv2.addWeighted(prev_meme, 1-alpha, meme, alpha, 0)
        elif not np.array_equal(prev_meme, meme):
            alpha = 0.0
            flash = 200
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
    
    cv2.putText(cam, "MEME MIRROR", (w//2 - 100, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.putText(cam, show_pose.upper(), (w-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[show_pose], 2)
    cv2.rectangle(cam, (0, h-8), (w, h), colors[show_pose], -1)
    cv2.rectangle(cam, (3, 3), (w-3, h-3), (255,255,255), 2)
    
    if streak > 0:
        cv2.putText(cam, f"x{streak}", (w-80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    if challenge:
        if challenge_target is None:
            challenge_target = random.choice(["thinking", "pointing", "shocked"])
            challenge_time = time.time()
        left = 5 - (time.time() - challenge_time)
        cv2.putText(cam, f"DO: {challenge_target.upper()}", (w//2-80, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,255), 3)
        cv2.putText(cam, f"{left:.1f}s", (w//2-30, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(cam, f"Score: {challenge_score}", (w//2-50, h//2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if stable == challenge_target:
            challenge_score += 1
            challenge_target = None
            flash = 255
        elif left <= 0:
            challenge_target = None
    
    if reaction:
        if reaction_target is None:
            reaction_target = random.choice(["thinking", "pointing", "shocked"])
            reaction_time = time.time()
        cv2.putText(cam, f"QUICK! {reaction_target.upper()}", (w//2-100, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)
        if stable == reaction_target:
            rt = time.time() - reaction_time
            if rt < best_reaction: best_reaction = rt
            cv2.putText(cam, f"{rt:.2f}s", (w//2-40, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            reaction_target = None
        if best_reaction < float('inf'):
            cv2.putText(cam, f"Best: {best_reaction:.2f}s", (w//2-50, h//2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    if combo:
        tgt = combo_seq[combo_idx]
        cv2.putText(cam, f"{' > '.join(combo_seq)}", (10, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(cam, f"DO: {tgt.upper()}", (w//2-60, h//2+20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 2)
        cv2.putText(cam, f"Combos: {combo_score}", (w//2-50, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if stable == tgt:
            combo_idx += 1
            if combo_idx >= len(combo_seq):
                combo_score += 1
                combo_idx = 0
                flash = 255
    
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
    cv2.putText(out, f"{int(fps)}fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    mode = ""
    if challenge: mode = "[CHALLENGE]"
    elif reaction: mode = "[REACTION]"
    elif combo: mode = "[COMBO]"
    elif manual_idx >= 0: mode = "[MANUAL]"
    if mode: cv2.putText(out, mode, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
    
    if recording:
        cv2.circle(out, (out.shape[1]-30, 30), 10, (0,0,255), -1)
        cv2.putText(out, "REC", (out.shape[1]-70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        if writer: writer.write(out)
    
    gif_buf.append(out.copy())
    
    cv2.putText(out, "Q=Quit D=Debug S=Screenshot F=Full P=Pause M=Meme B=Side I=PiP Z=Zoom", (10, out.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    cv2.putText(out, "1-3=Modes R=Record G=GIF C=Camera Arrows=Manual", (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    
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
    elif key == ord('1'):
        challenge, reaction, combo = not challenge, False, False
        challenge_target, challenge_score = None, 0
    elif key == ord('2'):
        reaction, challenge, combo = not reaction, False, False
        reaction_target, best_reaction = None, float('inf')
    elif key == ord('3'):
        combo, challenge, reaction = not combo, False, False
        combo_idx, combo_score = 0, 0
    elif key in [81, 2]: manual_idx = (manual_idx - 1) % len(poses) if manual_idx >= 0 else len(poses) - 1
    elif key in [83, 3]: manual_idx = (manual_idx + 1) % len(poses)
    elif key in [82, 0]: manual_idx = -1

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
holistic.close()