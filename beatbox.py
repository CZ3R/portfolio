from flask import Flask, Response, request, send_from_directory, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pygame
import os

app = Flask(__name__)

# Initialize pygame for sound playback
pygame.mixer.init()

# Load your drum sound samples dynamically
SOUND_DIR = "static/sounds"

# Function to load sounds
def load_sounds():
    sounds = {}
    for filename in os.listdir(SOUND_DIR):
        if filename.endswith(".wav"):
            note = filename.split(".")[0]  # Extract note number
            sounds[int(note)] = pygame.mixer.Sound(os.path.join(SOUND_DIR, filename))
    return sounds

# Load sounds into memory
sound_files = load_sounds()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam capture
camera = cv2.VideoCapture(0)

# Stream webcam feed
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play_sound', methods=['POST'])
def play_sound():
    """
    Play a sound when requested via API.
    Expected request: {"note": 1}
    """
    data = request.json
    note = data.get("note")

    if note is not None and note in sound_files:
        sound_files[note].play()
        return jsonify({"status": "success", "message": f"Playing note {note}"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid note"}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Beatbox API"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
