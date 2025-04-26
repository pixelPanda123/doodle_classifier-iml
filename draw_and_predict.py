import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- CONFIG ---
IMG_SIZE = 28
CATEGORIES = ["basketball", "book"]
DRAW_RADIUS = 7  # Bigger brush for easier drawing

# Load the trained model
model = load_model("model/model.h5")

# Initialize canvas
canvas = np.zeros((400, 400), dtype=np.uint8)

drawing = False
prediction_text = "Draw and press SPACE!"

def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), DRAW_RADIUS, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Setup window
cv2.namedWindow('Fruit Doodle Classifier')
cv2.setMouseCallback('Fruit Doodle Classifier', draw)

while True:
    # Create a color version of canvas to put text
    display_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Add heading
    cv2.putText(display_canvas, "Draw: Basketball or Book", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add prediction text
    cv2.putText(display_canvas, f"Prediction: {prediction_text}", (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Fruit Doodle Classifier', display_canvas)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key to exit
        break
    elif key == 32:  # SPACE key to predict
        # Invert the canvas
        canvas_inverted = cv2.bitwise_not(canvas)

        # Preprocess the image
        img = cv2.resize(canvas_inverted, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Predict
        prediction = model.predict(img)
        predicted_class = CATEGORIES[np.argmax(prediction)]
        confidence = np.max(prediction)

        prediction_text = f"{predicted_class} ({confidence*100:.1f}%)"

    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.zeros((400, 400), dtype=np.uint8)
        prediction_text = "Draw and press SPACE!"

cv2.destroyAllWindows()
