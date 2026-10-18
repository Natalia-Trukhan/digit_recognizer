import os
import tkinter as tk

from PIL import Image, ImageDraw

# === CORE !!! ===
import core

# === Глобальні змінні ===
tmp_count = 0
TRAIN_COUNT = 1

# === INTERFACE ===
# === ROOT ===
root = tk.Tk()
root.title("🧠 Digit Recognizer Trainer")
root.resizable(False, False)
root.columnconfigure(0, weight=1)
root.protocol("WM_DELETE_WINDOW", lambda: 0)

# === CONTROLS ===
canvas = tk.Canvas(root)
status_label = tk.Label(root, text="Готово ✅")
train_count = tk.Entry(root)
train_count.insert(0, str(TRAIN_COUNT))
entry = tk.Entry(root)
# === Drawing canvas ===
image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)

# === BUTTONS ===
train_mnist = tk.Button(root, text="Train Mnist")
mode_btn = tk.Button(root, text=f"{core.mode.upper()}")
ok_btn = tk.Button(root, text="OK")
clear_btn = tk.Button(root, text="Очистити")
quit_btn = tk.Button(root, text="Вихід")


# === LISTENERS ===
def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse((x - r, y - r, x + r, y + r), fill='black')


# === Очищення ===
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")
    entry.delete(0, tk.END)


# === Перемикання режиму ===
def toggle_mode():
    # global mode
    if os.path.exists(core.MODEL_PATH):
        os.remove(core.MODEL_PATH)
        print("Model deleted!")
    core.mode = "predict" if core.mode == "train" else "train"
    mode_btn.config(text=f"{core.mode.upper()}")
    clear_canvas()


# === Обробка зображення ===
def process_image_listener():
    global TRAIN_COUNT, tmp_count
    img = image.resize((28, 28))
    if core.mode == "train":
        TRAIN_COUNT = int(train_count.get())
        if entry.get().isdigit():
            label_value = int(entry.get())
            tmp_count += 1
            status_label.config(text=f"🔄 Додавання даних... {tmp_count}/{TRAIN_COUNT}")
            root.update()
            # TRAIN
            trained = core.train(img, label_value, TRAIN_COUNT)
            if trained:
                status_label.config(text="✅ Модель оновлена і збережена")
                core.mode = "predict"
                mode_btn.config(text=f"{core.mode.upper()}")
                clear_canvas()
                root.update()
    else:  # predict
        tmp_count = 0
        result = core.predict(img)
        entry.delete(0, tk.END)
        entry.insert(0, f"{result}")
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")

def train_mnist_listener():
    core.mnist()
    mode_btn.config(text=f"{core.mode.upper()}")
    root.update()



# === LISTENERS ===
train_mnist.config(command=train_mnist_listener)
canvas.bind("<B1-Motion>", paint)
mode_btn.config(command=toggle_mode)
ok_btn.config(command=process_image_listener)
clear_btn.config(command=clear_canvas)
quit_btn.config(command=root.destroy)

# === STYLE ===
WIDTH = 12
FONT = ("serif", 12, "normal")
FONT_LARGE = ("serif", 25)
#
train_mnist.config(bg="#DCDCDC", width=WIDTH, font=FONT)
canvas.config(width=280, height=280, bg="white")
train_count.config(width=WIDTH, font=FONT_LARGE, justify=tk.CENTER)
status_label.config(fg="green")
entry.config(width=WIDTH, font=FONT_LARGE, justify=tk.CENTER)
mode_btn.config(bg="lightblue", font=FONT, width=WIDTH)
ok_btn.config(bg="lightgreen", font=FONT, width=WIDTH)
clear_btn.config(bg="#DCDCDC", font=FONT, width=WIDTH)
quit_btn.config(bg="salmon", font=FONT, width=WIDTH)

# === POSITION ===
train_mnist.grid(row=0, column=0, columnspan=2)
quit_btn.grid(row=1, column=0)
mode_btn.grid(row=1, column=1)
train_count.grid(row=2, column=0, columnspan=2, pady=10)
status_label.grid(row=3, column=0, columnspan=2, pady=10)
canvas.grid(row=4, column=0, columnspan=2)
entry.grid(row=5, column=0, pady=15, columnspan=2)
clear_btn.grid(row=6, column=0)
ok_btn.grid(row=6, column=1)

# === WINDOW POSITION ===
root.update_idletasks()
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y - 100}")

# === START ===
root.mainloop()
