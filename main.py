import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw

# === CORE MODULE ===
import core

# === GLOBAL VARIABLES ===
tmp_count = 0
TRAIN_COUNT = 1

# === INTERFACE SETUP ===
root = tk.Tk()
root.title("🧠 Digit Recognizer Trainer")
root.resizable(False, False)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.protocol("WM_DELETE_WINDOW", lambda: 0)

# === CONTROLS ===
canvas = tk.Canvas(root)
status_label = tk.Label(root, text="Ready ✅")
train_count = tk.Entry(root)
train_count.insert(0, str(TRAIN_COUNT))
entry = tk.Entry(root)

# === DRAWING CANVAS IMAGE ===
image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)

# === BUTTONS ===
train_mnist_btn = tk.Button(root, text="Train MNIST")
test_mnist_btn = tk.Button(root, text="Test MNIST")
mode_btn = tk.Button(root, text=f"{core.mode.upper()}")
delete_model_btn = tk.Button(root, text="Delete Model")
clear_btn = tk.Button(root, text="Clear")
ok_btn = tk.Button(root, text="OK")
quit_btn = tk.Button(root, text="Quit")


# === LISTENERS / EVENT HANDLERS ===
def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse((x - r, y - r, x + r, y + r), fill='black')


# === CLEAR CANVAS ===
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")
    entry.delete(0, tk.END)


# === TOGGLE MODE ===
def toggle_mode():
    core.mode = "predict" if core.mode == "train" else "train"
    mode_btn.config(text=f"{core.mode.upper()}")
    clear_canvas()


# === CENTER IMAGE UTILITY ===
def center_image(img):
    arr = np.array(img)
    coords = np.argwhere(arr < 128)
    if coords.size == 0:
        return img.resize((28, 28))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))

    new_img = Image.new("L", (28, 28), 255)
    w, h = cropped.size
    max_size = 20
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    cropped_resized = cropped.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)

    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    new_img.paste(cropped_resized, (paste_x, paste_y))
    return new_img


# === IMAGE PROCESSING & PREDICTION / TRAINING ===
def process_image_listener():
    global TRAIN_COUNT, tmp_count

    if core.mode != "train" and not os.path.exists(core.MODEL_PATH):
        status_label.config(text="⚠️ Model not found! Click Train")
        root.update()
        return

    img = center_image(image)

    if core.mode == "train":
        TRAIN_COUNT = int(train_count.get())
        if entry.get().isdigit():
            label_value = int(entry.get())
            tmp_count += 1
            status_label.config(text=f"🔄 Adding... {tmp_count}/{TRAIN_COUNT}")
            root.update()

            trained = core.train(img, label_value, TRAIN_COUNT)
            if trained:
                status_label.config(text="✅ Model updated")
                core.mode = "predict"
                mode_btn.config(text=f"{core.mode.upper()}")
                clear_canvas()
                root.update()
    else:  # predict mode
        tmp_count = 0
        result = core.predict(img)
        entry.delete(0, tk.END)
        entry.insert(0, f"{result}")

    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")


def train_mnist_listener():
    if os.path.exists(core.MODEL_PATH):
        os.remove(core.MODEL_PATH)
        print("Old model removed for fresh MNIST training.")

    status_label.config(text="🔄 Training on MNIST...")
    root.update()
    core.train_mnist()
    status_label.config(text="✅ Training finished!")
    mode_btn.config(text=f"{core.mode.upper()}")
    root.update()


def test_mnist_listener():
    if not os.path.exists(core.MODEL_PATH):
        status_label.config(text="⚠️ Model missing! Click Train")
        root.update()
        return

    status_label.config(text="🔄 Testing model...")
    root.update()
    core.predict_mnist()
    status_label.config(text="✅ Test done (check console)")
    root.update()


def delete_model_listener():
    if os.path.exists(core.MODEL_PATH):
        os.remove(core.MODEL_PATH)
        status_label.config(text="🗑️ Model deleted!")
        print("Model deleted!")
    else:
        status_label.config(text="⚠️ Model file not found")
    root.update()


# === BIND COMMANDS ===
train_mnist_btn.config(command=train_mnist_listener)
test_mnist_btn.config(command=test_mnist_listener)
mode_btn.config(command=toggle_mode)
delete_model_btn.config(command=delete_model_listener)
canvas.bind("<B1-Motion>", paint)
clear_btn.config(command=clear_canvas)
ok_btn.config(command=process_image_listener)
quit_btn.config(command=root.destroy)

# === STYLES ===
WIDTH = 13
FONT = ("serif", 10, "normal")
FONT_LARGE = ("serif", 20)

train_mnist_btn.config(bg="#DCDCDC", width=WIDTH, font=FONT)
test_mnist_btn.config(bg="#DCDCDC", width=WIDTH, font=FONT)
mode_btn.config(bg="lightblue", width=WIDTH, font=FONT)
delete_model_btn.config(bg="#ffcccc", width=WIDTH, font=FONT)
canvas.config(width=280, height=280, bg="white")
train_count.config(width=10, font=FONT_LARGE, justify=tk.CENTER)
status_label.config(fg="green", font=("serif", 10, "bold"))
entry.config(width=10, font=FONT_LARGE, justify=tk.CENTER)
clear_btn.config(bg="#DCDCDC", width=WIDTH, font=FONT)
ok_btn.config(bg="lightgreen", width=WIDTH, font=FONT)
quit_btn.config(bg="salmon", width=WIDTH, font=FONT)

# === GRID LAYOUT POSITIONING ===
train_mnist_btn.grid(row=0, column=0, padx=5, pady=4, sticky="ew")
test_mnist_btn.grid(row=0, column=1, padx=5, pady=4, sticky="ew")

mode_btn.grid(row=1, column=0, padx=5, pady=4, sticky="ew")
delete_model_btn.grid(row=1, column=1, padx=5, pady=4, sticky="ew")

train_count.grid(row=2, column=0, columnspan=2, pady=5)
status_label.grid(row=3, column=0, columnspan=2, pady=3)
canvas.grid(row=4, column=0, columnspan=2, pady=3)
entry.grid(row=5, column=0, columnspan=2, pady=5)

clear_btn.grid(row=6, column=0, padx=5, pady=4, sticky="ew")
ok_btn.grid(row=6, column=1, padx=5, pady=4, sticky="ew")

quit_btn.grid(row=7, column=0, columnspan=2, padx=5, pady=4, sticky="ew")

# === CENTER WINDOW ON SCREEN ===
root.update_idletasks()
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y - 100}")

# === MAIN LOOP ===
root.mainloop()