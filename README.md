# ⚽ Toe Tap Counter

![Flutter](https://img.shields.io/badge/Flutter-02569B?logo=flutter&logoColor=white)
![Dart](https://img.shields.io/badge/Dart-0175C2?logo=dart&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-FF6F00?logo=tensorflow&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO26-111F68)
![Platform](https://img.shields.io/badge/Platform-Android-3DDC84?logo=android&logoColor=white)

A Flutter app that counts toe taps on a football in real time, using the phone camera and two on-device YOLO models (pose and object detection). All inference runs locally on the device, in a background isolate.

~~~text
Camera ─► YOLO Pose (person, knees, ankles) ─┐
                                             ├─► Foot ↔ ball contact logic ─► Tap counter
Camera ─► YOLO Detect (football) ────────────┘
~~~

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Models](#models)
- [Getting started](#getting-started)
- [How to use](#how-to-use)
- [How tap counting works](#how-tap-counting-works)
- [Performance](#performance)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)

## Features

- Live camera preview with a pose and ball overlay
- Person and leg keypoint detection (knees and ankles) with YOLO Pose
- Football detection with YOLO object detection
- Estimated toe point for each foot (COCO has no toe keypoint)
- Contact-based tap counting: one approach = one tap
- Duplicate-count prevention (per-foot re-arming and cooldowns)
- Independent left and right foot tracking, so alternating taps count
- Start / Stop and Reset controls
- On-screen diagnostics: person confidence, inference time, fps, ball status, foot/ball proximity
- Preprocessing and inference in a background isolate, so the UI stays responsive
- Graceful handling of temporary detection loss (the ball position is remembered briefly when the foot hides it)

## Tech stack

| Area | Technology |
|------|------------|
| App | Flutter / Dart (Android, portrait) |
| Camera | `camera` plugin (YUV420 stream) |
| Inference | `tflite_flutter` (TensorFlow Lite) |
| Models | Ultralytics YOLO26n-pose and YOLO26n, exported to TFLite |

## Models

Both models must be exported at the **same input size**. The app uses 320×320.

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `assets/yolo26n-pose.tflite` | Person and 17 COCO keypoints | `[1, 3, 320, 320]` | `[1, 56, 2100]` |
| `assets/football.tflite` | COCO detector; the app reads class 32 (`sports ball`) | `[1, 3, 320, 320]` | `[1, 84, 2100]` |

`2100 = (320/8)² + (320/16)² + (320/32)²` candidate boxes. The input layout (NHWC or NCHW) is detected at startup, so either works.

The football model is a general COCO detector filtered to the "sports ball" class. A football-specific trained model can be used instead: set `kBallClass` in `main.dart` to the ball's class index (0 for a single-class model).

## Getting started

### Prerequisites

- Flutter SDK (Dart `^3.13.4`, as set in `pubspec.yaml`)
- An Android device with a camera
- Python 3.11+ with `ultralytics` (only needed to export the models)

### 1. Export the models

~~~bash
pip install -U ultralytics
python export_models.py
~~~

This exports both models at 320×320 and copies them into `assets/`. On Windows, if the TFLite export fails on a newer Python version, use Python 3.11 in a virtual environment or run the script in Google Colab.

### 2. Check `pubspec.yaml`

~~~yaml
flutter:
  assets:
    - assets/yolo26n-pose.tflite
    - assets/football.tflite
~~~

### 3. Run

~~~bash
flutter pub get
flutter run --release
~~~

Use **release mode**. Debug builds are much slower and make the preprocessing loop look worse than it is.

### 4. Check the startup log

You should see lines like:

~~~text
POSE  in=[1, 3, 320, 320] out=[1, 56, 2100]
BALL  in=[1, 3, 320, 320] out=[1, 84, 2100]
timing prep=..ms pose=..ms ball=..ms total=..ms
~~~

If a shape differs, the app shows an initialization error with the expected shape. Re-export with `end2end=False` and the same `imgsz` as `kInputSize` in `main.dart`.

## How to use

1. Stand so your legs and the football are both visible, with the phone in portrait.
2. Wait until the status reads **Ready. Press Start.**
3. Press **Start** and tap the ball.
4. Press **Stop** to pause counting, or **Reset** to set the counter to 0.

Status messages: `Looking for person...`, `Show both feet`, `Show the football`, `Counting...`.

## How tap counting works

For each foot, the distance to the ball centre is the smaller of the ankle distance and the estimated toe-point distance. Each foot runs this cycle independently:

~~~text
ARMED      foot approaches the ball (distance shrinking)
   │       distance < touchFactor × contactR
   ▼
CONTACT detected ─► COUNT +1 ─► DISARMED
   │
DISARMED   foot rests on / hovers near the ball: nothing counts
   │       foot moves away (leaves the zone, or backs off from its closest point)
   ▼
ARMED      again, waiting for the next approach
~~~

- The count happens at the moment of contact, not when the foot leaves.
- A foot that stays close to the ball never counts twice.
- A 250 ms per-foot cooldown and a 150 ms global cooldown guard against jitter and both feet counting the same contact.
- After Start, Reset, or losing the ball, a foot already on the ball must move away before it can count.
- The contact zone scales with the ball radius and shin length, so it works at different distances from the camera.

### Tuning

Constants are at the top of `_ToeTapPageState` in `main.dart`. The `foot/ball` value in the info panel shows what the logic sees: it should drop below about 0.9 at each tap and rise above about 1.1 between taps.

| Constant | Default | Effect |
|----------|---------|--------|
| `touchFactor` | 0.90 | Contact distance in contact-zone units. Higher counts more easily |
| `rearmFactor` | 1.10 | Distance at which a foot counts as having left the ball |
| `retreatFactor` | 0.40 | Backing off by this much re-arms a foot that hovers over the ball |
| `footExtend` | 0.30 | How far past the ankle the toe point is placed (fraction of shin) |
| `footCooldownMs` | 250 | Minimum time between counts for one foot |
| `cooldownMs` | 150 | Minimum time between any two counts |
| `ballHoldMs` | 1200 | How long the last ball position is kept if the ball is hidden |

Missed taps: raise `touchFactor` or lower `retreatFactor`. Extra counts: lower `touchFactor` or raise `retreatFactor`.

## Performance

- Preprocessing (rotation, letterbox, YUV→RGB) and both models run in one background isolate. Frames that arrive while it is busy are dropped, so it always works on the newest frame.
- The camera frame is converted to a tensor once and shared by both models.
- A 320×320 input has about 4× fewer pixels than 640×640. For small or distant balls, try 416: set `IMGSZ` in `export_models.py` and `kInputSize` in `main.dart`, then re-export.
- The `timing` log lines and the on-screen ms/fps show where time goes.

## Project structure

~~~text
toe_tap/
├── lib/
│   └── main.dart            # app, isolate worker, decoding, tap logic, UI
├── assets/
│   ├── yolo26n-pose.tflite
│   └── football.tflite
├── export_models.py         # exports both models at 320×320 into assets/
└── pubspec.yaml
~~~

## Limitations

- **No true toe keypoint.** COCO pose stops at the ankle, so the toe is estimated by extending the knee→ankle line.
- **General ball model.** The default football detector is COCO's "sports ball". It can miss blurry, fast or distant balls. Fine-tuning on football images would improve it.
- **Single person, single ball.** Only the highest-scoring person and ball are tracked.
- **Frame rate limits fast taps.** At low fps a quick tap can fall between frames and be missed.
- **Portrait only.** The overlay math assumes portrait orientation.
- Tested on Android only.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the pose and detection models
- [tflite_flutter](https://pub.dev/packages/tflite_flutter) for TFLite inference in Flutter
- [camera](https://pub.dev/packages/camera) for the live camera stream