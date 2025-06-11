import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the coordinates of specific landmarks (e.g., wrist and index finger tip)
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the x, y coordinates for wrist and index finger tip
            wrist_coords = (wrist.x * frame.shape[1], wrist.y * frame.shape[0])
            index_finger_tip_coords = (index_finger_tip.x * frame.shape[1], index_finger_tip.y * frame.shape[0])

            # Calculate Euclidean distance between wrist and index finger tip
            distance = euclidean_distance(wrist_coords, index_finger_tip_coords)
            
            # Display the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the resulting frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
