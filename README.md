# Eye-Tracker-Mp4-with-Python
A real-time eye tracking tool built with Python, OpenCV, and MediaPipe that classifies gaze direction (Left, Center, Right) and tracks how long the user looks in each direction.
Supports both live webcam mode and MP4 video analysis mode. Uses MediaPipe's Face Landmarker Tasks API to detect iris position relative to eye corners, with a live HUD overlay showing current gaze direction and running dwell times. Prints a full summary on exit.
Features

Real-time iris tracking via MediaPipe's 478-point face mesh
Gaze classification: Left / Center / Right with adjustable sensitivity
Dwell time tracking per direction with percentage breakdown
Live HUD with colour-coded direction indicator and position bar
Video mode uses frame-accurate timestamps for precise analysis
Auto-downloads the face landmarker model on first run

Built With

Python 3.11
OpenCV
MediaPipe 0.10.30
