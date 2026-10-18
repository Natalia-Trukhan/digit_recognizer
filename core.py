import os
import numpy as np
from PIL import ImageOps
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

# ГЛОБАЛЬНІ ЗМІННІ
MODEL_PATH = "model.h5"
mode = "train" #train, predict

#СТВОРЕННЯ АБО ЗАВАНТАЖЕННЯ МОДЕЛІ
if os.path.exists(MODEL_PATH):
    mode = "predict"
    print("Loading exists model...")
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