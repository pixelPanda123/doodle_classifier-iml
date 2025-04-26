import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib  # for saving model structure
from tensorflow.keras.models import load_model

# --- CONFIG ---
IMG_SIZE = 28
CATEGORIES = ["basketball", "book"]

DATA_DIR = "data"

def load_data():
    X = []
    y = []

    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = to_categorical(y, num_classes=len(CATEGORIES))
    return X, y

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    print("Saving model...")
    model.save("model/model.h5")
    print("Model saved to model/model.h5")

if __name__ == "__main__":
    main()
