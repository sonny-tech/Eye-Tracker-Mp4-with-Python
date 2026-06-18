"""
Eye Tracker — detects Left / Center / Right gaze and tracks dwell time.
Uses MediaPipe Tasks API (compatible with mediapipe 0.10.x on Python 3.11).
 
Dependencies:
    pip install opencv-python mediapipe==0.10.30

by Sonny Raminafshar, 06-17-2026
On first run, downloads the face landmarker model (~30 MB) to ~/Desktop/face_landmarker.task




Eye Tracker — Debugging Guide

1. Verify your environment is active
Before running anything, confirm you see (eye_tracker_env) at the start of your terminal line. If not:
bashsource ~/Desktop/eye_tracker_env/bin/activate
Then confirm versions:
bashpython --version          # should say 3.11.x
pip show mediapipe        # should say 0.10.30
pip show opencv-python    # should be installed

2. Camera not opening
If you get RuntimeError: Cannot open webcam, try changing the device index in the script:
pythoncap = cv2.VideoCapture(0)  # try 1 or 2 if 0 doesn't work
Also check that no other app (Zoom, FaceTime, etc.) is using the camera at the same time.

3. Model file issues
If you get an error about the model file, check it downloaded correctly:
bashls -lh ~/Desktop/face_landmarker.task
It should be around 29–31 MB. If it's tiny (a few KB), it downloaded incorrectly — delete it and rerun:
bashrm ~/Desktop/face_landmarker.task
python eye_tracker_Sonny.py

4. Face not being detected
If the HUD always shows Gaze: NONE:

Make sure your face is well lit — poor lighting is the most common cause
Sit 0.5–1.5 metres from the camera, not too close or too far
Confirm the camera window is actually showing your face

You can add this temporary debug line right after if result.face_landmarks: to print detection status each frame:
pythonprint(f"Faces detected: {len(result.face_landmarks)}")
Remove it once confirmed working.

5. Gaze direction feels wrong or too sensitive
The THRESHOLD variable at the top of the script controls sensitivity:
pythonTHRESHOLD = 0.35   # default

Increase toward 0.45 — makes LEFT/RIGHT harder to trigger (less sensitive)
Decrease toward 0.25 — makes LEFT/RIGHT easier to trigger (more sensitive)

If LEFT and RIGHT feel swapped, it's likely a camera mirroring issue. The script already flips the frame with cv2.flip(frame, 1) — if it's still wrong, remove that line.

6. Laggy or slow performance
If the video feed feels choppy:

Close other heavy applications
Add this after cap = cv2.VideoCapture(0) to lower resolution:

pythoncap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
This reduces processing load significantly.

7. Dwell times look wrong
If times seem to accumulate incorrectly, add a quick print inside the direction-change block to trace it:
pythonif gaze != gaze_now:
    if gaze_now != "NONE":
        dwell[gaze_now] += now - gaze_start
        print(f"Logged {now - gaze_start:.2f}s for {gaze_now}")  # debug line
    gaze_now   = gaze
    gaze_start = now
This prints every time a direction change is logged so you can verify the timing is being recorded correctly.

8. Quick sanity check sequence
Run through this checklist every time before a real session:

(eye_tracker_env) is active in terminal
Camera window opens and shows your face
Iris dots (coloured circles) appear on your eyes
Moving eyes left/right changes the HUD label and colour
Dwell times increment while holding a direction
Pressing Q closes the window and prints the summary

If all 6 pass, everything is working correctly.

"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode
)
import time
import os
import urllib.request
from collections import defaultdict
 
# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/Desktop/face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
 
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~30 MB) — one-time only...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model saved to", MODEL_PATH)
 
# ── Landmark indices (478-point mesh with iris) ────────────────────────────────
LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [362, 263]   # inner, outer
RIGHT_EYE_CORNERS = [133,  33]   # inner, outer
 
# ── Gaze classification ────────────────────────────────────────────────────────
THRESHOLD = 0.35   # tune between 0.25–0.45 to adjust sensitivity
 
def iris_ratio(iris_pts, corner_pts):
    iris_cx = sum(p[0] for p in iris_pts) / len(iris_pts)
    x_left  = min(corner_pts[0][0], corner_pts[1][0])
    x_right = max(corner_pts[0][0], corner_pts[1][0])
    eye_w   = x_right - x_left
    return 0.5 if eye_w == 0 else (iris_cx - x_left) / eye_w
 
def classify_gaze(ratio):
    if ratio < THRESHOLD:
        return "LEFT"
    elif ratio > (1 - THRESHOLD):
        return "RIGHT"
    return "CENTER"
 
def get_pts(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
 
# ── Colours ────────────────────────────────────────────────────────────────────
COLOURS = {
    "LEFT":   (255, 140,   0),   # orange
    "CENTER": ( 50, 205,  50),   # lime green
    "RIGHT":  ( 30, 144, 255),   # dodger blue
    "NONE":   (180, 180, 180),
}
 
# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam — check device index.")
 
    dwell      = defaultdict(float)
    gaze_now   = "NONE"
    gaze_start = time.time()
 
    print("Eye Tracker running — press  Q  to quit.\n")
 
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
 
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
 
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect(mp_image)
 
            gaze = "NONE"
 
            if result.face_landmarks:
                lms = result.face_landmarks[0]
 
                l_iris   = get_pts(lms, LEFT_IRIS,          w, h)
                r_iris   = get_pts(lms, RIGHT_IRIS,         w, h)
                l_corner = get_pts(lms, LEFT_EYE_CORNERS,   w, h)
                r_corner = get_pts(lms, RIGHT_EYE_CORNERS,  w, h)
 
                avg  = (iris_ratio(l_iris, l_corner) + iris_ratio(r_iris, r_corner)) / 2
                gaze = classify_gaze(avg)
 
                for pt in l_iris + r_iris:
                    cv2.circle(frame, pt, 3, COLOURS[gaze], -1)
 
            # ── Dwell accounting ───────────────────────────────────────────
            now = time.time()
            if gaze != gaze_now:
                if gaze_now != "NONE":
                    dwell[gaze_now] += now - gaze_start
                gaze_now   = gaze
                gaze_start = now
 
            running = (now - gaze_start) if gaze_now != "NONE" else 0
 
            # ── HUD ────────────────────────────────────────────────────────
            col = COLOURS[gaze]
 
            # Top bar — current direction
            cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.putText(frame, f"Gaze: {gaze}", (15, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2, cv2.LINE_AA)
 
            # Bottom panel — dwell times
            panel_y = h - 110
            cv2.rectangle(frame, (0, panel_y), (w, h), (20, 20, 20), -1)
            x_off = 15
            for d in ("LEFT", "CENTER", "RIGHT"):
                total = dwell[d] + (running if gaze_now == d else 0)
                cv2.putText(frame, f"{d}  {total:.1f}s", (x_off, panel_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS[d], 2, cv2.LINE_AA)
                x_off += 220
 
            # Position indicator dot
            bar_w, bar_h = 300, 18
            bx = (w - bar_w) // 2
            by = panel_y + 60
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
            if gaze != "NONE":
                filled = int(bar_w * {"LEFT": 0.1, "CENTER": 0.5, "RIGHT": 0.9}[gaze])
                cv2.circle(frame, (bx + filled, by + bar_h // 2), 10, col, -1)
            for lbl, xf in (("L", bx - 20), ("C", bx + bar_w // 2 - 5), ("R", bx + bar_w + 5)):
                cv2.putText(frame, lbl, (xf, by + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
 
            cv2.imshow("Eye Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    # Flush final segment
    now = time.time()
    if gaze_now != "NONE":
        dwell[gaze_now] += now - gaze_start
 
    cap.release()
    cv2.destroyAllWindows()
 
    # ── Summary ────────────────────────────────────────────────────────────────
    total = sum(dwell.values()) or 1
    print("\n── Gaze Summary ─────────────────────────────────────")
    for d in ("LEFT", "CENTER", "RIGHT"):
        t   = dwell[d]
        pct = 100 * t / total
        print(f"  {d:<8}  {t:6.2f}s  ({pct:5.1f}%)  {'█' * int(pct / 2)}")
    print(f"  {'TOTAL':<8}  {total:6.2f}s")
    print("─────────────────────────────────────────────────────\n")
 
if __name__ == "__main__":
    main()