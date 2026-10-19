# Mreme / Meme Mirror

A real-time pose detection application built with Python, OpenCV, and MediaPipe that displays specific monkey memes based on your body poses. The app uses computer vision to instantly detect and match four distinct states: Staring (neutral), Thinking (hand on mouth), Pointing (finger up), and Shocked (open mouth), displaying the corresponding monkey meme template in real-time.

<p align="center">
  <img src="memes/staring.jpg" width="180" />
  <img src="memes/thinking.jpg" width="180" />
  <img src="memes/pointing.jpg" width="180" />
  <img src="memes/shocked.jpg" width="180" />
</p>

## Features

- Real-time pose detection using the MediaPipe Holistic Landmarker
- Automatic monkey meme matching based on detected poses
- Multiple display modes (split view, side-by-side, picture-in-picture)
- Screenshot and video recording capabilities
- Face overlay onto monkey meme templates
- Vertical mode for social media export
- Virtual camera output, so you can use the memes as your webcam in Teams, Zoom, or Meet

## Supported Poses

| Pose | How to Trigger |
|------|----------------|
| Staring | Default state - just look at the camera |
| Thinking | Touch your finger to your chin |
| Pointing | Point your finger above your head |
| Shocked | Open your mouth wide |

## Requirements

- Python 3.11–3.14
- Computer with a webcam
- Libraries in requirements.txt
- Internet access on first run, to download the MediaPipe Holistic Landmarker model (about 14 MB) into `models/`

## Setup and running program

First, clone the repo
```bash
git clone https://github.com/vkunamneni1/mreme.git
cd mreme
```

Create a virtual environment (any Python from 3.11 to 3.14)
```bash
# Windows
py -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install necessary dependencies
```bash
pip install -r requirements.txt
```

Run program
```bash
python main.py
```

On the first run the app downloads the pose model to `models/holistic_landmarker.task`. If the download fails, get it from the URL the app prints and save it at that path.

### Use it as your webcam in Teams (or Zoom, Meet)

The app sends the meme to a virtual camera through [pyvirtualcam](https://github.com/letmaik/pyvirtualcam). On Windows this uses the OBS Virtual Camera, so install [OBS Studio](https://obsproject.com) once (OBS itself does not need to be running). On macOS, install OBS 30 or newer, then start and stop its virtual camera once in OBS. On Linux, set up [v4l2loopback](https://github.com/letmaik/pyvirtualcam#linux-v4l2loopback).

1. Run `python main.py`. The virtual camera is on by default (the window title says "virtual camera on"). Press `O` to turn it off and on, or run `python main.py --no-virtual-cam` to start with it off.
2. In Teams, go to Settings > Devices > Camera and pick **OBS Virtual Camera**. If it isn't listed, fully quit Teams (right-click its tray icon > Quit) and start it again.

The app needs your real webcam, so don't pick the real webcam in Teams at the same time. The app starts in meme-only view, and Teams gets the meme fitted into a 16:9 frame with black bars. Press `M` for the split view with your camera, or `B` for side by side. Teams doesn't get the app's on-screen controls (help, fps, badges). If the Teams preview stays black, press `O` twice to restart the feed. Teams may mirror your own preview, so text can look backwards to you. Other people see it the right way round.

If the app reports that it cannot read frames from the camera, close any other app that is using the webcam (Teams, Zoom, browser tabs, OBS) and try again. Press `C` in the app to switch to another camera.

## Controls

| Key | Function |
|-----|----------|
| Q | Quit |
| H | Show/hide help overlay |
| S | Take screenshot |
| R | Start/stop video recording |
| G | Export GIF frames |
| F | Toggle fullscreen |
| P | Pause |
| M | Meme-only view (default) / split view |
| B | Side-by-side view |
| I | Picture-in-picture mode |
| V | Vertical mode (9:16 aspect ratio) |
| W | Toggle watermark |
| A | Toggle face overlay on meme |
| Z | Toggle face zoom |
| N | Toggle smooth transitions |
| C | Switch camera |
| O | Virtual camera on/off (for Teams, Zoom, Meet) |
| ←/→ | Manually cycle through memes |
| ↑ | Return to auto-detection |
