import threading
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import customtkinter as ctk
import keyboard
from sklearn.linear_model import Ridge
from collections import deque
import tkinter as tk

# --- GLOBALS ---
is_running = False
show_cam = True
mode = "Direct"  # Direct, Joystick, or Eye
mouth_was_open = False
nose_middle = (0.0, 0.0)
calibrated = False
smooth_x, smooth_y = 0.0, 0.0
last_landmarks = None

# Internal alpha: High slider (0.8) -> Low alpha (0.2) -> High Smoothness
alpha = 0.2

# Eye tracking globals
eye_features = []
eye_targets = []
eye_model = Ridge(alpha=0.001)
eye_is_trained = False
eye_smoothing = deque(maxlen=5)
eye_calibration_points = []
eye_current_point_idx = 0
eye_calibrating = False
eye_dot_x, eye_dot_y = 0, 0
eye_cal_window = None
eye_cal_canvas = None
eye_cal_target = None
eye_cal_gaze_dot = None
eye_overlay_window = None
eye_overlay_canvas = None
eye_overlay_dot = None

# MediaPipe & Camera Setup
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

current_cam_idx = 0
cam = cv2.VideoCapture(current_cam_idx)


# --- CORE LOGIC ---

def get_frame():
    success, frame = cam.read()
    if not success: return None, None
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    return frame, results


def check_mouth_open(landmarks):
    mouth_dist = abs(landmarks[13].y - landmarks[14].y)
    face_height = abs(landmarks[10].y - landmarks[152].y)
    if face_height == 0: return False
    return (mouth_dist / face_height) > 0.08


def apply_smoothing(target_x, target_y):
    global smooth_x, smooth_y
    smooth_x = (alpha * target_x) + ((1 - alpha) * smooth_x)
    smooth_y = (alpha * target_y) + ((1 - alpha) * smooth_y)
    return smooth_x, smooth_y


def move_mouse(landmarks, speed, tolerance):
    ref_center_x = (landmarks[205].x + landmarks[425].x) / 2
    ref_center_y = (landmarks[205].y + landmarks[425].y) / 2
    corrected_x = (landmarks[1].x + nose_middle[0] / 100)
    corrected_y = (landmarks[1].y + nose_middle[1] / 100)
    x_val = (corrected_x - ref_center_x) * 10
    y_val = (corrected_y - ref_center_y) * 10
    if abs(x_val) < (tolerance / 10): x_val = 0
    if abs(y_val) < (tolerance / 10): y_val = 0
    sm_x, sm_y = apply_smoothing(x_val, y_val)
    pyautogui.moveRel(speed * sm_x / 5, speed * sm_y / 5, duration=0)


def move_mouse_direct(landmarks, sensitivity=10):
    ref_center_x = (landmarks[205].x + landmarks[425].x) / 2
    ref_center_y = (landmarks[205].y + landmarks[425].y) / 2
    corrected_x = (landmarks[1].x + nose_middle[0] / 100)
    corrected_y = (landmarks[1].y + nose_middle[1] / 100)
    dx = (corrected_x - ref_center_x)
    dy = (corrected_y - ref_center_y)
    raw_x = (screen_w / 2) + (dx * screen_w * sensitivity)
    raw_y = (screen_h / 2) + (dy * screen_h * sensitivity)
    sm_x, sm_y = apply_smoothing(raw_x, raw_y)
    sm_x = max(0, min(screen_w, sm_x))
    sm_y = max(0, min(screen_h, sm_y))
    pyautogui.moveTo(sm_x, sm_y, duration=0)


# --- EYE TRACKING LOGIC ---

def get_eye_data(landmarks):
    """Extract eye tracking features from landmarks"""
    # Left Eye
    l_iris = np.array([landmarks[468].x, landmarks[468].y])
    l_inner = np.array([landmarks[133].x, landmarks[133].y])
    l_outer = np.array([landmarks[33].x, landmarks[33].y])

    # Right Eye
    r_iris = np.array([landmarks[473].x, landmarks[473].y])
    r_inner = np.array([landmarks[362].x, landmarks[362].y])
    r_outer = np.array([landmarks[263].x, landmarks[263].y])

    # Calculate feature vectors with amplification
    feat = [
        (l_iris[0] - l_inner[0]) * 100, (l_iris[1] - l_inner[1]) * 100,
        (r_iris[0] - r_inner[0]) * 100, (r_iris[1] - r_inner[1]) * 100,
        (l_iris[0] - l_outer[0]) * 100, (r_iris[0] - r_outer[0]) * 100
    ]
    return feat


def start_eye_calibration():
    """Initialize eye tracking calibration with fullscreen overlay"""
    global eye_calibration_points, eye_current_point_idx, eye_calibrating, eye_features, eye_targets, eye_is_trained
    global eye_cal_window, eye_cal_canvas, eye_cal_target, eye_cal_gaze_dot

    # Reset calibration data
    eye_features = []
    eye_targets = []
    eye_is_trained = False
    eye_current_point_idx = 0
    eye_calibrating = True

    # Define calibration points (13 points across the screen)
    eye_calibration_points = [
        (50, 50), (screen_w // 2, 50), (screen_w - 50, 50),
        (50, screen_h // 2), (screen_w // 2, screen_h // 2), (screen_w - 50, screen_h // 2),
        (50, screen_h - 50), (screen_w // 2, screen_h - 50), (screen_w - 50, screen_h - 50),
        (screen_w // 4, screen_h // 4), (3 * screen_w // 4, screen_h // 4),
        (screen_w // 4, 3 * screen_h // 4), (3 * screen_w // 4, 3 * screen_h // 4)
    ]

    # Create fullscreen calibration window
    eye_cal_window = tk.Toplevel()
    eye_cal_window.attributes("-fullscreen", True, "-topmost", True)
    eye_cal_window.config(bg='white')
    eye_cal_window.wm_attributes("-transparentcolor", "white")

    eye_cal_canvas = tk.Canvas(eye_cal_window, width=screen_w, height=screen_h,
                               bg='white', highlightthickness=0)
    eye_cal_canvas.pack()

    # Create gaze dot (red) - hidden initially
    eye_cal_gaze_dot = eye_cal_canvas.create_oval(0, 0, 30, 30, fill='red', state='hidden')

    # Create target dot (yellow)
    eye_cal_target = eye_cal_canvas.create_oval(0, 0, 45, 45, fill='gold', outline='black', width=3)

    # Bind click event to canvas for calibration
    eye_cal_canvas.bind("<Button-1>", on_calibration_click)

    # Position first target
    move_eye_calibration_target()

    status_label.configure(text="Eye Calibration: Look at yellow dot and CLICK or press SPACE", text_color="orange")


def move_eye_calibration_target():
    """Move the yellow calibration target to the next position"""
    global eye_cal_target, eye_cal_canvas

    if eye_current_point_idx < len(eye_calibration_points):
        x, y = eye_calibration_points[eye_current_point_idx]
        eye_cal_canvas.coords(eye_cal_target, x - 22, y - 22, x + 22, y + 22)


def on_calibration_click(event=None):
    """Handle click or space press during calibration"""
    global last_landmarks
    if eye_calibrating and last_landmarks:
        capture_eye_calibration_point(last_landmarks)


def capture_eye_calibration_point(landmarks):
    """Capture a single calibration point"""
    global eye_current_point_idx, eye_calibrating, eye_is_trained

    if eye_current_point_idx < len(eye_calibration_points):
        target_x, target_y = eye_calibration_points[eye_current_point_idx]

        # Capture burst of data for this point
        for _ in range(20):
            vec = get_eye_data(landmarks)
            if vec:
                eye_features.append(vec)
                eye_targets.append([target_x, target_y])

        eye_current_point_idx += 1

        if eye_current_point_idx >= len(eye_calibration_points):
            # Calibration complete, train the model
            finalize_eye_calibration()
        else:
            # Move to next point
            move_eye_calibration_target()


def finalize_eye_calibration():
    """Train the eye tracking model"""
    global eye_is_trained, eye_calibrating, calibrated, eye_cal_window
    global eye_overlay_window, eye_overlay_canvas, eye_overlay_dot

    if len(eye_features) > 50:
        X = np.array(eye_features)
        y = np.array(eye_targets)
        eye_model.fit(X, y)
        eye_is_trained = True
        eye_calibrating = False
        calibrated = True

        # Close calibration window
        if eye_cal_window:
            try:
                eye_cal_window.destroy()
            except:
                pass
            eye_cal_window = None

        # Create overlay window for red gaze dot
        create_eye_overlay()

        status_label.configure(text="Status: Eye Calibrated - Press Q to track, W to toggle dot", text_color="cyan")
    else:
        status_label.configure(text="Error: Not enough calibration data", text_color="red")
        eye_calibrating = False
        if eye_cal_window:
            try:
                eye_cal_window.destroy()
            except:
                pass
            eye_cal_window = None


def create_eye_overlay():
    """Create or recreate the eye overlay window with red dot"""
    global eye_overlay_window, eye_overlay_canvas, eye_overlay_dot

    # Clean up existing window if any
    if eye_overlay_window:
        try:
            eye_overlay_window.destroy()
        except:
            pass

    # Create overlay window for red gaze dot
    eye_overlay_window = tk.Toplevel()
    eye_overlay_window.attributes("-fullscreen", True, "-topmost", True)
    eye_overlay_window.config(bg='white')
    eye_overlay_window.wm_attributes("-transparentcolor", "white")

    eye_overlay_canvas = tk.Canvas(eye_overlay_window, width=screen_w, height=screen_h,
                                   bg='white', highlightthickness=0)
    eye_overlay_canvas.pack()

    # Create red gaze dot
    eye_overlay_dot = eye_overlay_canvas.create_oval(0, 0, 30, 30, fill='red')


def toggle_eye_dot():
    """Toggle visibility of the red eye tracking dot"""
    global eye_overlay_window

    if mode == "Eye" and eye_is_trained and eye_overlay_window:
        try:
            # Check if window is visible
            if eye_overlay_window.state() == 'normal':
                eye_overlay_window.withdraw()  # Hide
                status_label.configure(text="Status: Dot Hidden - Press W to show", text_color="yellow")
            else:
                eye_overlay_window.deiconify()  # Show
                status_label.configure(text="Status: Dot Visible - Press W to hide", text_color="cyan")
        except:
            pass


def move_mouse_eye(landmarks):
    """Move mouse using eye tracking"""
    global eye_dot_x, eye_dot_y, eye_overlay_canvas, eye_overlay_dot

    if not eye_is_trained:
        return

    vec = get_eye_data(landmarks)
    if vec:
        pred = eye_model.predict([vec])[0]
        eye_smoothing.append(pred)
        avg = np.mean(eye_smoothing, axis=0)

        # Clip coordinates to screen bounds
        x, y = int(avg[0]), int(avg[1])
        x = np.clip(x, 0, screen_w)
        y = np.clip(y, 0, screen_h)

        eye_dot_x, eye_dot_y = x, y

        # Update overlay dot position with safety check
        try:
            if eye_overlay_window and eye_overlay_canvas and eye_overlay_dot:
                eye_overlay_canvas.coords(eye_overlay_dot, x - 15, y - 15, x + 15, y + 15)
        except:
            pass

        # Move mouse if tracking is active
        if is_running:
            pyautogui.moveTo(x, y, duration=0)


def check_clicks(landmarks):
    left_eye = abs(landmarks[145].y - landmarks[159].y)
    right_eye = abs(landmarks[386].y - landmarks[374].y)
    if left_eye < 0.012 and right_eye > 0.012:
        pyautogui.click(button='left')
        pyautogui.sleep(0.2)
    elif right_eye < 0.012 and left_eye > 0.012:
        pyautogui.click(button='right')
        pyautogui.sleep(0.2)


def draw_visuals(frame, results):
    h, w, _ = frame.shape
    if not show_cam:
        frame = np.zeros(frame.shape, dtype=np.uint8)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        def draw_dot(idx, color=(0, 255, 255), radius=1):
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), radius, color, -1)

        # Draw landmarks based on mode
        if mode == "Eye":
            # Eye tracking landmarks
            # Iris centers (Red)
            draw_dot(468, (0, 0, 255), 2)
            draw_dot(473, (0, 0, 255), 2)
            # Eye corners (Green)
            draw_dot(133, (0, 255, 0), 2)
            draw_dot(33, (0, 255, 0), 2)
            draw_dot(362, (0, 255, 0), 2)
            draw_dot(263, (0, 255, 0), 2)
        else:
            # Original landmarks for Direct/Joystick modes
            # Face Outline / General points (Cyan)
            for i in [152, 361, 1]: draw_dot(i)

            # Bridge of Nose (Yellow-ish/Teal)
            draw_dot(205, (255, 255, 0))
            draw_dot(425, (255, 255, 0))

            # Mouth - Toggle Points (Magenta)
            draw_dot(13, (255, 0, 255), 2)
            draw_dot(14, (255, 0, 255), 2)

            # Face Frame / Corners (Yellow)
           # for i in [398, 263, 296, 330]: draw_dot(i, (255, 255, 0))

            # Eyelids - Click Detection (Red)
            for i in [145, 159, 386, 374]: draw_dot(i, (0, 0, 255))

            # Calibration & Bridge Centers (Visual Helpers)
            corrected_x = (landmarks[1].x + nose_middle[0] / 100)
            corrected_y = (landmarks[1].y + nose_middle[1] / 100)
            cv2.circle(frame, (int(corrected_x * w), int(corrected_y * h)), 3, (0, 0, 255), -1)

            bridge_center_x = (landmarks[205].x + landmarks[425].x) / 2
            bridge_center_y = (landmarks[205].y + landmarks[425].y) / 2
            cv2.circle(frame, (int(bridge_center_x * w), int(bridge_center_y * h)), 3, (255, 0, 0), -1)

    return frame

def draw_full_visuals(frame, results):
    h, w, _ = frame.shape
    if not show_cam:
        frame = np.zeros(frame.shape, dtype=np.uint8)

    if results.multi_face_landmarks:
        all_landmarks = results.multi_face_landmarks[0].landmark

        def draw_dot(idx):
            if idx < len(all_landmarks):
                cx, cy = int(all_landmarks[idx].x * w), int(all_landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 1, (255, 255, 0), -1)

        if mode == "Eye":
            # Precise Eye + Socket (Eyelids and Iris) only
            # Left Eye: 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246
            # Right Eye: 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466
            # Irises: 468, 469, 470, 471, 472 (Left) | 473, 474, 475, 476, 477 (Right)
            eye_indices = [
                # Left Eye Socket/Lids
                33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                # Right Eye Socket/Lids
                263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
                # All Iris Points
                468, 469, 470, 471, 472, 473, 474, 475, 476, 477
            ]
            for i in eye_indices:
                draw_dot(i)
        else:
            # EVERY single landmark for all other modes
            for i in range(len(all_landmarks)):
                draw_dot(i)

    return frame

def run_auto_calibration(landmarks=None):
    global nose_middle, calibrated, last_landmarks
    global eye_cal_window, eye_calibrating

    # Check if Eye mode is active
    if mode == "Eye":
        # If already calibrating, destroy the existing window to restart
        if eye_cal_window is not None:
            try:
                eye_cal_window.destroy()
            except:
                pass
            eye_cal_window = None
            eye_calibrating = False

        # Start (or restart) eye calibration
        start_eye_calibration()
        return

    # Original calibration for Direct/Joystick modes
    target = landmarks if landmarks else last_landmarks
    if target:
        c_x = (target[205].x + target[425].x) / 2
        c_y = (target[205].y + target[425].y) / 2
        dev_x = (c_x - target[1].x) * 100
        dev_y = (c_y - target[1].y) * 100
        nose_middle = (dev_x, dev_y)
        calibrated = True
        status_label.configure(text="Status: Calibrated", text_color="cyan")
    else:
        status_label.configure(text="Error: No Face Detected", text_color="red")

def change_camera(choice):
    global cam, current_cam_idx
    new_idx = int(choice.split(" ")[-1])
    if new_idx != current_cam_idx:
        current_cam_idx = new_idx
        cam.release()
        cam = cv2.VideoCapture(current_cam_idx)


def change_mode(choice):
    global mode, is_running, calibrated, eye_overlay_window, eye_cal_window, eye_calibrating
    global eye_overlay_canvas, eye_overlay_dot, eye_is_trained

    # Stop tracking when changing modes
    if is_running:
        is_running = False
        start_btn.configure(text="Start Tracking (Q)", fg_color="green")

    # Clean up eye calibration window if calibrating
    if eye_calibrating and eye_cal_window:
        try:
            eye_cal_window.destroy()
        except:
            pass
        eye_cal_window = None
        eye_calibrating = False

    # Hide eye overlay if leaving eye mode (but don't destroy - keep calibration)
    if mode == "Eye" and choice != "Eye":
        if eye_overlay_window:
            try:
                eye_overlay_window.withdraw()  # Hide instead of destroy
            except:
                pass

    # Show eye overlay if returning to eye mode and already calibrated
    if choice == "Eye" and eye_is_trained:
        if eye_overlay_window:
            try:
                eye_overlay_window.deiconify()  # Show the window
            except:
                # If window was destroyed, recreate it
                create_eye_overlay()
        else:
            create_eye_overlay()

    # Center the cursor
    pyautogui.moveTo(screen_w // 2, screen_h // 2)

    # Update mode
    mode = choice

    # Only reset calibrated flag for non-Eye modes or if Eye not trained
    if mode != "Eye":
        calibrated = False
    elif not eye_is_trained:
        calibrated = False
    else:
        calibrated = True

    # Update button text - Q for all modes
    start_btn.configure(text="Start Tracking (Q)")

    if mode == "Eye":
        if eye_is_trained:
            status_label.configure(text="Status: Eye Mode Ready - Press Q to track, W to toggle dot", text_color="cyan")
        else:
            status_label.configure(text="Status: Eye Mode - Calibrate First", text_color="yellow")
    else:
        status_label.configure(text="Status: Ready", text_color="yellow")


def toggle_tracking():
    global is_running

    # For eye mode, check if calibrated first
    if mode == "Eye" and not eye_is_trained:
        run_auto_calibration()
        return

    is_running = not is_running
    start_btn.configure(text=f"Stop Tracking (Q)" if is_running else f"Start Tracking (Q)",
                        fg_color="red" if is_running else "green")
    status_label.configure(text="Status: Active" if is_running else "Status: Paused",
                           text_color="green" if is_running else "yellow")


# --- KEYBOARD & UI HELPERS ---

def handle_hotkeys(e):
    if e.name == 'left':
        v = max(0.01, smooth_slider.get() - 0.05)
        smooth_slider.set(v)
        update_smoothing_ui(v)
    elif e.name == 'right':
        v = min(1.0, smooth_slider.get() + 0.05)
        smooth_slider.set(v)
        update_smoothing_ui(v)
    elif e.name == 'up':
        v = min(2000, speed_slider.get() + 50)
        speed_slider.set(v)
        speed_label.configure(text=f"Speed: {int(v)}")
    elif e.name == 'down':
        v = max(100, speed_slider.get() - 50)
        speed_slider.set(v)
        speed_label.configure(text=f"Speed: {int(v)}")


keyboard.on_press(handle_hotkeys)


def update_smoothing_ui(v):
    global alpha
    alpha = 1.01 - float(v)
    smooth_label.configure(text=f"Ai Smoothing: {round(v, 2)}")


def main_loop():
    global mouth_was_open, calibrated, last_landmarks
    while True:
        frame, results = get_frame()
        if frame is None: continue

        if results and results.multi_face_landmarks:
            last_landmarks = results.multi_face_landmarks[0].landmark

            # 1. ALWAYS check the mouth (for Q toggle), but NOT in Eye mode
            if mode != "Eye":
                is_open = check_mouth_open(last_landmarks)

                if is_open and not mouth_was_open:
                    toggle_tracking()
                    mouth_was_open = True
                    pyautogui.sleep(0.5)
                elif not is_open:
                    mouth_was_open = False
            else:
                # Reset mouth state when in Eye mode to avoid issues when switching back
                mouth_was_open = False

            # 2. Hotkeys
            if keyboard.is_pressed('space'):
                if mode == "Eye" and eye_calibrating:
                    # Capture calibration point in eye mode
                    on_calibration_click()
                    pyautogui.sleep(0.3)
                else:
                    run_auto_calibration(last_landmarks)

            # Q toggles tracking in ALL modes
            if keyboard.is_pressed('q'):
                toggle_tracking()
                pyautogui.sleep(0.3)

            # W toggles red dot visibility in Eye mode only
            if keyboard.is_pressed('w') and mode == "Eye":
                toggle_eye_dot()
                pyautogui.sleep(0.3)

            # 3. Only move mouse if running
            if is_running:
                if mode == "Eye":
                    move_mouse_eye(last_landmarks)
                    check_clicks(last_landmarks)
                else:
                    if not calibrated:
                        run_auto_calibration(last_landmarks)

                    if mode == "Joystick":
                        move_mouse(last_landmarks, speed_slider.get(), tol_slider.get())
                    else:
                        move_mouse_direct(last_landmarks, sensitivity=(speed_slider.get() / 30))

                    check_clicks(last_landmarks)

            # Update red dot position for Eye mode even when not running
            if mode == "Eye" and eye_is_trained and not is_running:
                vec = get_eye_data(last_landmarks)
                if vec:
                    pred = eye_model.predict([vec])[0]
                    eye_smoothing.append(pred)
                    avg = np.mean(eye_smoothing, axis=0)
                    x, y = int(avg[0]), int(avg[1])
                    eye_dot_x, eye_dot_y = np.clip(x, 0, screen_w), np.clip(y, 0, screen_h)

                    # Update overlay dot position with safety check
                    try:
                        if eye_overlay_window and eye_overlay_canvas and eye_overlay_dot:
                            eye_overlay_canvas.coords(eye_overlay_dot, eye_dot_x - 15, eye_dot_y - 15,
                                                      eye_dot_x + 15, eye_dot_y + 15)
                    except:
                        pass

            frame = draw_visuals(frame, results)

        cv2.imshow('Tracking Feed', frame)
        cv2.waitKey(1)


# --- APP STARTUP ---
app = ctk.CTk()
app.title("AI Mouse Procedural")
app.geometry("400x600")

cam_frame = ctk.CTkFrame(app, fg_color="transparent")
cam_frame.pack(fill="x", padx=10, pady=5)
cam_select = ctk.CTkOptionMenu(cam_frame, values=["Camera 0", "Camera 1", "Camera 2"], command=change_camera, width=120)
cam_select.set("Camera 0")
cam_select.pack(side="left")

ctk.CTkLabel(app, text="AI Mouse Controller", font=("Roboto", 20)).pack(pady=10)

# Changed to dropdown menu for mode selection
mode_select = ctk.CTkOptionMenu(app, values=["Direct", "Joystick", "Eye"], command=change_mode,
                                fg_color="purple", width=200)
mode_select.set("Direct")
mode_select.pack(pady=5)

view_btn = ctk.CTkButton(app, text="Camera View: ON", command=lambda: (
    globals().update(show_cam=not show_cam),
    view_btn.configure(text=f"Camera View: {'ON' if show_cam else 'OFF'}")))
view_btn.pack(pady=5)

# Sliders
smooth_label = ctk.CTkLabel(app, text="Ai Smoothing: 0.8")
smooth_label.pack()
smooth_slider = ctk.CTkSlider(app, from_=0.01, to=1.0, command=update_smoothing_ui)
smooth_slider.set(0.8)
alpha = 1.01 - 0.8
smooth_slider.pack()

speed_label = ctk.CTkLabel(app, text="Speed: 800")
speed_label.pack()
speed_slider = ctk.CTkSlider(app, from_=100, to=2000, command=lambda v: speed_label.configure(text=f"Speed: {int(v)}"))
speed_slider.set(800)
speed_slider.pack()

tol_label = ctk.CTkLabel(app, text="Tolerance: 0.5")
tol_label.pack()
tol_slider = ctk.CTkSlider(app, from_=0.0, to=3.0,
                           command=lambda v: tol_label.configure(text=f"Tolerance: {round(v, 2)}"))
tol_slider.set(0.5)
tol_slider.pack()

# Calibration and control buttons
ctk.CTkButton(app, text="Calibrate (Space)", command=run_auto_calibration).pack(pady=10)
start_btn = ctk.CTkButton(app, text="Start Tracking (Q)", fg_color="green", command=toggle_tracking)
start_btn.pack(pady=5)
status_label = ctk.CTkLabel(app, text="Status: Ready", text_color="yellow")
status_label.pack()

threading.Thread(target=main_loop, daemon=True).start()
app.mainloop()