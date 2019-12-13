import os
import pyaudio
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

import retrieval


# Setting audio parameters
DURATION = 2
FREQ_AUDIO = 44100
AUDIO_RANGE = 2**15
AUDIO = pyaudio.PyAudio()


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.x0 = 30
        self.y0 = 80
        self.rows = 2
        self.columns = 5
        self.dim = 350
        self.border = 10
        self.audio_batch = int(FREQ_AUDIO * DURATION / self.dim)
        self.root.title("DSIM Project - Context Based IR")
        self.root.geometry("{}x{}".format(
            self.x0 * 2 + (self.columns * self.dim) +
                ((self.columns - 1) * self.border),
            self.y0 +30 + (self.rows * self.dim) +
                ((self.rows - 1) * self.border)))
        self.root.resizable(False, False)
        self.query_interface = tk.Entry(self.root)
        self.query_interface.place(height=30, width=500,
                                   x=60, y=20)
        self.query_interface.bind("<Return>", lambda x: self.search())
        self.query_button = tk.Button(self.root,
                                      text="SEARCH",
                                      command=self.search)
        self.query_button.place(height=40, width=60,
                                x=570, y=15)
        self.images = [tk.Canvas(self.root,
                                 height=200, width=200,
                                 bg="white")
                       for _ in range(self.rows * self.columns)]
        self.df = pd.DataFrame()
        for y in range(2):
            for x in range(5):
                i = y*5 + x
                self.images[i].bind("<Button-1>",
                                    lambda x, i=i: self.press_button(i))
                self.images[i].place(height=self.dim, width=self.dim,
                                     x=self.x0+(self.dim + self.border)*x,
                                     y=self.y0+(self.dim + self.border)*y)
        self.photo_names = []
        self.root.mainloop()

    def clean_images(self):
        for img in self.images:
            img.delete("all")

    def search(self):
        self.df = retrieval.get_images_from_text(self.query_interface.get())
        self.clean_images()
        self.get_best_images()

    def update_images(self, n, score):
        selected_photo = self.photo_names[n]
        self.df = retrieval.update_scores(self.df,
                                          selected_photo,
                                          score)  # TODO: add method
        self.clean_images()
        self.get_best_images()

    def get_best_images(self):
        self.photo_names = retrieval.get_higher(self.df)
        photos = (open_image(photo, self.dim)
                  for photo in self.photo_names)
        for canvas, photo in zip(self.images, photos):
            x = (self.dim - photo.width()) / 2
            y = (self.dim - photo.height()) / 2
            canvas.create_image(x, y,
                                anchor=tk.NW,
                                image=photo)
            canvas.image = photo

    def record_audio(self, n):
        out = []
        audio_stream = AUDIO.open(format=pyaudio.paInt16,
                                  input=True,
                                  rate=FREQ_AUDIO,
                                  channels=1,
                                  frames_per_buffer=self.audio_batch)
        for i in range(int(DURATION * FREQ_AUDIO / self.audio_batch)):
            data = np.frombuffer(audio_stream.read(self.audio_batch),
                                 dtype=np.int16) / AUDIO_RANGE
            out.append(data)
            volume = np.mean(np.abs(data)) * self.dim
            self.images[n].create_line(i, self.dim, i, self.dim - volume, fill="blue")
        out = np.hstack(out)
        score = get_audio_score(out)
        self.update_images(n, score)

    def press_button(self, n):
        th = threading.Thread(target=self.record_audio, args=[n])
        th.start()


def open_image(img_file, size):
    img = Image.open(img_file)
    img.thumbnail((size, size), Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)

def get_audio_score(audio):
    return +0.75


if __name__ == "__main__":
    gui = GUI()
