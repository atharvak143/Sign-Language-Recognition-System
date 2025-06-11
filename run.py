import cv2
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import joblib
import numpy as np
import sys
print(sys.path)

WINDOW = "Hand Tracking"
TEXT_WINDOW = "Predicted Text"  # Separate window for displaying predicted text
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = src.extra.connections
int_to_char = src.extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

gesture_clf = joblib.load(r'models\\gesture_clf.pkl')
if not gesture_clf:
    print("Failed to load gesture classifier. Exiting...")
    exit()

cv2.namedWindow(WINDOW)
cv2.namedWindow(TEXT_WINDOW)  # Create a separate window for predicted text
capture = cv2.VideoCapture(0)

word = []
letter = ""
staticGesture = 0
MAX_LINE_LENGTH = 15  # Maximum number of characters per line for display

def wrap_text(word_buffer, max_line_length):
    """
    Wraps the text in the word buffer into multiple lines.
    """
    text = ''.join(word_buffer)
    wrapped_lines = [text[i:i+max_line_length] for i in range(0, len(text), max_line_length)]
    return wrapped_lines

while True:
    # Read frame from webcam
    hasFrame, frame = capture.read()
    if not hasFrame:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB for detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image)

    if points is not None:
        # Draw detected points and make predictions
        src.extra.draw_points(points, frame)
        pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
        if letter == pred_sign:
            staticGesture += 1
        else:
            letter = pred_sign
            staticGesture = 0
        if staticGesture > 6:
            word.append(letter)
            staticGesture = 0
    else:
        # Handle spaces when no hand points are detected
        if word and word[-1] != " ":
            staticGesture += 1
            if staticGesture > 6:
                word.append(" ")
                staticGesture = 0

    # Display the hand-tracking video feed
    cv2.imshow(WINDOW, frame)

    # Create a black canvas for the predicted text
    text_frame = np.zeros((500, 800, 3), dtype=np.uint8)

    # Wrap and display the predicted text in the text window
    wrapped_lines = wrap_text(word, MAX_LINE_LENGTH)
    for i, line in enumerate(wrapped_lines):
        position = (50, 50 + (i * 30))  # Adjust vertical spacing for each line
        cv2.putText(text_frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the reset button on the text window
    cv2.putText(text_frame, "Press 'r' to reset", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the text frame
    cv2.imshow(TEXT_WINDOW, text_frame)

    # Handle key presses
    key = cv2.waitKey(1)

    if 32 <= key <= 126:  # Handle printable characters
        word.append(chr(key))
        print(f"Added '{chr(key)}' to the word buffer.")

    if key == 27:  # ESC key
        print("Exiting program...")
        break

    if key == 8:  # Backspace key
        if word:
            print(f"Deleted last letter: {word[-1]}")
            del word[-1]
        else:
            print("No letters to delete.")

    if key == ord('r'):  # 'r' key for reset
        word = []
        print("Resetting the word buffer...")

# Release the webcam and destroy all windows
capture.release()
cv2.destroyAllWindows()
