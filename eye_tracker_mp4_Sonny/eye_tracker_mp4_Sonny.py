"""
Eye Tracker (Video Mode) — detects Left / Center / Right gaze and tracks dwell time.
Uses MediaPipe Tasks API (compatible with mediapipe 0.10.x on Python 3.11).

Dependencies:
    pip install opencv-python mediapipe==0.10.30

Place your video file as sample_video.mp4 in the same folder as this script.
On first run, downloads the face landmarker model (~30 MB) to ~/Desktop/face_landmarker.task

by Sonny Raminafshar, 06-18-2026
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

# ── Video input ────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_video.mp4")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video file not found: {VIDEO_PATH}\n"
        "Place your video file named 'sample_video.mp4' in the same folder as this script."
    )

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

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps

    print(f"Analyzing: {os.path.basename(VIDEO_PATH)}")
    print(f"Duration:  {duration:.1f}s  |  FPS: {fps:.1f}  |  Frames: {total_frames}\n")
    print("Press  Q  to quit early.\n")

    dwell        = defaultdict(float)
    gaze_now     = "NONE"
    # Use video timestamp (seconds) instead of wall clock
    gaze_start_t = 0.0
    frame_idx    = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # video ended

            frame_idx += 1
            video_time = frame_idx / fps   # current position in seconds

            h, w = frame.shape[:2]

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

            # ── Dwell accounting (using video timestamp) ───────────────────
            if gaze != gaze_now:
                if gaze_now != "NONE":
                    dwell[gaze_now] += video_time - gaze_start_t
                gaze_now     = gaze
                gaze_start_t = video_time

            running = (video_time - gaze_start_t) if gaze_now != "NONE" else 0

            # ── HUD ────────────────────────────────────────────────────────
            col = COLOURS[gaze]

            # Top bar — current direction + video timestamp
            cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.putText(frame, f"Gaze: {gaze}", (15, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2, cv2.LINE_AA)
            timestamp_str = f"{int(video_time // 60):02d}:{video_time % 60:05.2f} / {int(duration // 60):02d}:{duration % 60:05.2f}"
            cv2.putText(frame, timestamp_str, (w - 280, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

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

            cv2.imshow("Eye Tracker — Video Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit early by user.\n")
                break

    # Flush final segment
    final_time = frame_idx / fps
    if gaze_now != "NONE":
        dwell[gaze_now] += final_time - gaze_start_t

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────────────────────────
    total = sum(dwell.values()) or 1
    print("\n── Gaze Summary ─────────────────────────────────────")
    print(f"  Video: {os.path.basename(VIDEO_PATH)}")
    print(f"  Analyzed: {final_time:.2f}s of {duration:.2f}s")
    print()
    for d in ("LEFT", "CENTER", "RIGHT"):
        t   = dwell[d]
        pct = 100 * t / total
        print(f"  {d:<8}  {t:6.2f}s  ({pct:5.1f}%)  {'█' * int(pct / 2)}")
    print(f"  {'TOTAL':<8}  {total:6.2f}s")
    print("─────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()