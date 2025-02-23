import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load your drum sound samples for left and right hands.
# Only non-thumb fingertips are used for triggering sound.
left_sounds = {
    8: pygame.mixer.Sound("do/fa.wav"),    # e.g., Kick
    12: pygame.mixer.Sound("do/me.wav"),    # e.g., Snare
    16: pygame.mixer.Sound("do/re.wav"),      # e.g., Hi-Hat
    20: pygame.mixer.Sound("do/do.wav"),     # e.g., Clap
}

right_sounds = {
    8: pygame.mixer.Sound("do/so.wav"),     # e.g., Kick
    12: pygame.mixer.Sound("do/la.wav"),     # e.g., Snare
    16: pygame.mixer.Sound("do/ti.wav"),       # e.g., Hi-Hat
    20: pygame.mixer.Sound("do/do_2.wav"),      # e.g., Clap
}

# Mapping for instrument labels (including the thumb as reference)
left_instruments = {
    4: "Thumb",
    8: "Kick",
    12: "Snare",
    16: "Hi-Hat",
    20: "Clap"
}

right_instruments = {
    4: "Thumb",
    8: "Kick",
    12: "Snare",
    16: "Hi-Hat",
    20: "Clap"
}

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

# For debouncing sound triggers
triggered = {}  # key: (hand_index, finger_tip_index)

# Define the threshold distance (in normalized coordinates) for a "touch"
THUMB_TOUCH_THRESHOLD = 0.05  # Adjust this value based on testing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Prepare the frame for output (convert back to BGR)
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    h, w, _ = output_frame.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness label ("Left" or "Right")
            handedness = results.multi_handedness[hand_idx].classification[0].label

            # Get the thumb tip (landmark 4) as the reference point
            thumb = hand_landmarks.landmark[4]

            # --- Sound Triggering for Non-Thumb Fingers ---
            for finger_tip_idx in [8, 12, 16, 20]:
                finger_tip = hand_landmarks.landmark[finger_tip_idx]
                # Compute Euclidean distance (only x and y are needed)
                distance = np.linalg.norm(
                    np.array([thumb.x, thumb.y]) - np.array([finger_tip.x, finger_tip.y])
                )
                key = (hand_idx, finger_tip_idx)
                if distance < THUMB_TOUCH_THRESHOLD:
                    if not triggered.get(key, False):
                        triggered[key] = True
                        # Select appropriate sound based on handedness
                        if handedness == "Left":
                            sound = left_sounds.get(finger_tip_idx)
                        else:
                            sound = right_sounds.get(finger_tip_idx)
                        if sound:
                            sound.play()
                    # Optional: draw a green circle to indicate a trigger (overridden later)
                    cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)
                    cv2.circle(output_frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                else:
                    triggered[key] = False

            # --- Drawing Only the 5 Fingertip Points per Hand ---
            # We will display the thumb (4) and the 4 finger tips (8, 12, 16, 20)
            for landmark_idx in [4, 8, 12, 16, 20]:
                landmark = hand_landmarks.landmark[landmark_idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Choose color: if this finger is actively triggering (only for non-thumb), draw green.
                if landmark_idx in [8, 12, 16, 20]:
                    key = (hand_idx, landmark_idx)
                    if triggered.get(key, False):
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                else:
                    # Thumb is always blue (or any color you prefer) since it's a reference.
                    color = (255, 0, 0)

                # Draw a circle at the landmark
                cv2.circle(output_frame, (cx, cy), 10, color, cv2.FILLED)

                # Get the instrument label based on handedness
                if handedness == "Left":
                    label = left_instruments.get(landmark_idx, "")
                else:
                    label = right_instruments.get(landmark_idx, "")

                # Print the instrument label next to the point
                cv2.putText(output_frame, label, (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow("Beatbox Fingertips", output_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
