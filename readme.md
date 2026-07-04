# AI Mouse Controller

An AI-powered virtual mouse controller that allows users to control the computer cursor using **head movements, eye gaze, facial gestures, and blink detection**. The project combines **MediaPipe Face Mesh**, **OpenCV**, **Machine Learning**, and **CustomTkinter** to provide a hands-free Human-Computer Interaction (HCI) system.

---
## ⬇ Download

Download the latest Windows executable here:

**[Download Eye Cursor AI](https://github.com/Ahmed-043/vision_control/releases/download/v1.0.0/Eye_Cursor_Ai.exe)**

---

## Features

### 🖱 Cursor Control Modes

- **Direct Mode**
  - Maps head movement directly to screen coordinates.
  - Smooth and intuitive cursor movement.

- **Joystick Mode**
  - Uses relative head movement similar to a joystick.
  - Adjustable speed and movement tolerance.

- **Eye Tracking Mode**
  - Uses iris tracking to estimate gaze direction.
  - Machine-learning based gaze prediction using Ridge Regression.
  - Full-screen calibration with multiple calibration points.
  - Optional gaze indicator overlay.

---

## Additional Features

- Real-time facial landmark detection using MediaPipe Face Mesh.
- Eye tracking with calibration and machine learning.
- Cursor smoothing to reduce jitter.
- Automatic head calibration.
- Left-eye blink → Left mouse click.
- Right-eye blink → Right mouse click.
- Mouth-open gesture to start/stop tracking (Head Tracking modes).
- Multiple camera support.
- Adjustable:
  - Cursor Speed
  - Movement Smoothing
  - Movement Tolerance
- Camera preview toggle.
- Live facial landmark visualization.
- Eye tracking overlay showing predicted gaze position.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- PyAutoGUI
- CustomTkinter
- Keyboard
- Threading

---

## Project Structure

```
AI-Mouse-Controller/
│
├── main.py          # Main application
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-Mouse-Controller.git
cd AI-Mouse-Controller
```

### 2. Create a virtual environment

### Windows

Installation
1. Open the `Libraries` folder.
2. Run `python-3.8.0-amd64.exe` and install Python 3.8 (64-bit).
3. Create a virtual environment using Python 3.8:


```bash
py -3.8 -m venv .venv
```

Activate it

```bash
.venv\Scripts\activate
```

Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

or install manually

```bash
pip install opencv-python pyautogui numpy customtkinter keyboard scikit-learn mediapipe==0.8.10
```

---

## Running the Application

```bash
python main.py
```

---

## Controls

| Action                  | Key       |
|-------------------------|-----------|
| Start / Stop Tracking   | **Q**     |
| Calibrate               | **Space** |
| Toggle Eye Tracking Dot | **W**     |
| Increase Smoothing      | **→**     |
| Decrease Smoothing      | **←**     |
| Increase Speed          | **↑**     |
| Decrease Speed          | **↓**     |

---

## Cursor Modes

### Direct Mode

Uses the position of the user's head to move the cursor directly across the screen.

---

### Joystick Mode

Uses head displacement from a calibrated center position.

Ideal for users who prefer relative cursor movement.

---

### Eye Tracking Mode

Uses iris landmarks detected by MediaPipe.

The system first performs calibration by asking the user to look at multiple screen positions.

A Ridge Regression model learns the relationship between eye features and screen coordinates, enabling gaze-based cursor movement.

---

## Mouse Click Detection

Blink gestures are used as mouse clicks.

| Gesture         | Action          |
|-----------------|-----------------|
| Left Eye Blink  | Left Click      |
| Right Eye Blink | Right Click     |
| Open Mouth      | Toggle Tracking |

---

## Calibration

### Head Calibration

Automatically records the neutral head position and compensates for camera alignment.

---

### Eye Calibration

The user follows a yellow dot displayed at multiple screen locations.

The collected eye feature vectors are used to train a Ridge Regression model that predicts gaze position.

---

## Adjustable Parameters

- Cursor Speed
- Cursor Smoothing
- Movement Tolerance
- Camera Selection
- Camera View Toggle

---

## How It Works

1. The webcam captures live video frames.
2. MediaPipe Face Mesh extracts facial landmarks.
3. Depending on the selected mode:
   - Head landmarks control cursor movement.
   - Eye landmarks predict gaze location.
4. Cursor movement is smoothed using exponential smoothing.
5. Blink detection performs mouse clicks.
6. PyAutoGUI sends mouse commands to the operating system.

---

## Requirements

- Python 3.7 to 3.9
- mediapipe 0.8.10
- Webcam
- Windows/Linux

---
## Note
This project requires MediaPipe version 0.8.10. This version is compatible only with Python 3.7 to 3.9. Using a different version of MediaPipe or Python may result in compatibility issues or runtime errors.
---

## Future Improvements

- Scroll control using facial gestures.
- Drag-and-drop support.
- Double-click gesture.
- Eye tracking without calibration.
- Deep learning based gaze estimation.
- Multi-monitor support.
- Gesture customization.

---

## Applications

- Hands-free computer interaction
- Accessibility for physically impaired users
- Human-Computer Interaction research
- Smart workstation control
- Educational AI projects
- It is just a proof of concept, not a production code

---

## License

This project is intended for educational and research purposes.

---

## Author

**Muhammad Ahmed Mughal**

Department of Computer Science (Undergraduate)
