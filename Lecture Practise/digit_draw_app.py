"""
Interactive digit recognizer.
Draw a digit with your mouse, click Predict, and see what the model thinks.

Setup:
  1. In your training notebook, after model.fit(...), run:
         model.save("mnist_model.keras")
  2. Put mnist_model.keras in the same folder as this script
     (or update MODEL_PATH below).
  3. Run this script from a WSL terminal:
         python3 digit_draw_app.py
"""

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

MODEL_PATH = "mnist_model.keras"
CANVAS_SIZE = 280          # on-screen drawing area (280x280 px)
MODEL_INPUT_SIZE = 28      # MNIST expects 28x28
BRUSH_RADIUS = 10


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a digit")

        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Tkinter canvas the user draws on
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # In-memory image that mirrors what's drawn on the canvas
        # (so we can feed it to the model without screenshotting)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        tk.Button(root, text="Predict", command=self.predict, width=10).grid(
            row=1, column=0, pady=(0, 10)
        )
        tk.Button(root, text="Clear", command=self.clear, width=10).grid(
            row=1, column=1, pady=(0, 10)
        )

        # Result label
        self.result_var = tk.StringVar(value="Draw a digit, then click Predict")
        tk.Label(root, textvariable=self.result_var, font=("Arial", 16)).grid(
            row=2, column=0, columnspan=3, pady=(0, 10)
        )

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        # Draw on the visible canvas
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # Mirror the same stroke onto the in-memory PIL image
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_var.set("Draw a digit, then click Predict")

    def predict(self):
        # Resize the 280x280 drawing down to 28x28, like MNIST images
        img_small = self.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img_array = np.array(img_small).astype("float32") / 255.0
        img_array = img_array.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        self.result_var.set(f"Prediction: {predicted_digit}  ({confidence:.1f}% confident)")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
