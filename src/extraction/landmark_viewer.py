import cv2
import mediapipe as mp

# ----------------------------
# Initialize MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Open Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            print(f"\n======== Hand {hand_no + 1} ========")

            for idx, landmark in enumerate(hand_landmarks.landmark):

                print(
                    f"Landmark {idx:02d} | "
                    f"x={landmark.x:.4f} "
                    f"y={landmark.y:.4f} "
                    f"z={landmark.z:.4f}"
                )

    cv2.imshow("Landmark Viewer", frame)

    key = cv2.waitKey(1)

    if key in [ord('q'), ord('Q'), 27]:  # 27 = Esc
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()