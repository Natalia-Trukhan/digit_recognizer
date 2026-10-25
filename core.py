import os
from pathlib import Path

import numpy as np
from PIL import ImageOps, Image
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

# === GLOBAL VARIABLES ===
MODEL_PATH = "model.h5"
mode = "train"  # train, predict

# === CREATE OR LOAD MODEL ===
if os.path.exists(MODEL_PATH):
    mode = "predict"
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    mode = "train"
    print("Creating new model...")
    model = Sequential([
        Dense(128, input_shape=(784,), activation="relu"),
        Dense(256, activation="relu"),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

train_images, train_labels = [], []


def train(img, number_value, train_count):
    """
    Adds drawn images to the training set or starts training the model.
    """
    global train_images
    global train_labels
    global model

    # Invert colors to match MNIST format (black background, white digit)
    img = ImageOps.invert(img)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(784)

    train_images.append(img_array)
    train_labels.append(number_value)

    if len(train_images) >= train_count:
        print("Training on custom data...")
        model.fit(np.array(train_images), np.array(train_labels), epochs=5, verbose=0)
        model.save(MODEL_PATH)

        train_images = []
        train_labels = []

        return True
    return False


def predict(img):
    """
    Predicts the digit from the user's canvas drawing.
    """
    global train_images
    global train_labels

    img = ImageOps.invert(img)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 784)

    print(f"Predicting shape:\n{img_array.shape}")
    pred = model.predict(img_array)

    result_pred = np.argmax(pred)
    return result_pred


def train_mnist():
    """
    Trains the model using the official MNIST dataset from 'mnist/train'.
    """
    global model
    train_dir = Path("mnist/train")

    if not train_dir.exists():
        print("The mnist folder doesn't exist!")
        return

    image = []
    labels = []
    print("Loading MNIST training pictures...")

    for img_path in train_dir.glob('*.*'):
        try:
            file_name = img_path.name
            label = int(file_name.split("_")[0])
            with Image.open(img_path).convert('L') as img:
                img = img.resize((28, 28))
                img_array = np.array(img).astype("float32") / 255.0
                img_array = img_array.reshape(784)
                image.append(img_array)
                labels.append(label)
        except Exception:
            pass

    array_img = np.array(image, dtype='float32')
    array_lab = np.array(labels, dtype="int32")

    print(f"Starting training with {len(array_img)} MNIST pictures...")

    model.fit(array_img, array_lab, epochs=10, batch_size=32, verbose=1)
    model.save(MODEL_PATH)

    print("✅ Training finished, model saved to model.h5!")


def predict_mnist():
    """
    Evaluates model accuracy using the MNIST test dataset from 'mnist/test'.
    """
    test_dir = Path("mnist/test")
    if not test_dir.exists():
        print("The mnist test folder doesn't exist!")
        return

    images = []
    labels = []

    for img_path in test_dir.glob('*.*'):
        try:
            file_name = img_path.name
            label = int(file_name.split("_")[0])
            with Image.open(img_path).convert('L') as img:
                img = img.resize((28, 28))
                array_img = np.array(img).astype("float32") / 255.0
                array_img = array_img.reshape(784)
                images.append(array_img)
                labels.append(label)
        except Exception:
            pass

    array_img = np.array(images)
    array_lab = np.array(labels)

    print("Evaluating model accuracy on test dataset...")
    loss, accuracy = model.evaluate(array_img, array_lab, verbose=1)
    print(f"📊 Test Results (Accuracy): {accuracy * 100:.2f}% (Loss: {loss:.4f})")


def mnist():
    train_mnist()
    predict_mnist()