import pyaudio
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

import librosa
from tensorflow.keras.models import load_model

import retrieval
import text



# Setting audio parameters
DURATION = 2
FREQ_AUDIO = 44100
AUDIO_RANGE = 2**15
AUDIO = pyaudio.PyAudio()

FEATURES = ("VGG", "YOLO", "SIFT")
DISTANCES = {
    "VGG": ["cosine", "euclidean", "manhattan"],
    "YOLO": ["cosine", "euclidean", "manhattan"],
    "SIFT": ["sift"]
}

SCORES = {
    "sì": +2/3,
    "no": -2/3,
    "forse": +1/3,
    "non capisco": 0
}


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
            self.y0 + 30 + (self.rows * self.dim) +
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
        self.features_var = tk.StringVar(self.root)
        self.features_box = tk.OptionMenu(self.root,
                                          self.features_var,
                                          *FEATURES)
        self.features_var.set(FEATURES[0])
        self.features_box.place(height=30, width=120,
                                x=700, y=20)
        self.distance_var = tk.StringVar(self.root)
        self.distance_box = tk.OptionMenu(self.root,
                                          self.distance_var,
                                          "")
        self.distance_box.place(height=30, width=120,
                                x=830, y=20)
        self.text_label = tk.Label(self.root,
                                   text=text.INITIAL_TEXT,
                                   justify="left",
                                   anchor="w")
        self.text_label.place(height=20, width=1000,
                              x=970, y=25)
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
        self.text_generator = text.text_generator()
        self.photo_names = []
        self.root.mainloop()

    def clean_images(self):
        for img in self.images:
            img.delete("all")

    def search(self):
        method = self.get_features_extractor()
        self.update_distances(method)
        self.df = retrieval.get_images_from_text(self.query_interface.get(),
                                                 method)
        self.clean_images()
        self.get_best_images()
        self.text_generator = text.text_generator()
        self.text_label["text"] = text.SEARCH_TEXT

    def update_distances(self, method):
        self.distance_box["menu"].delete(0, "end")
        for new_choice in DISTANCES[method]:
            self.distance_box["menu"].add_command(
                label=new_choice,
                command=tk._setit(self.distance_var, new_choice))
        self.distance_var.set(DISTANCES[method][0])

    def update_images(self, n, score):
        selected_photo = self.photo_names[n]
        self.df = retrieval.update_scores(self.df,
                                          selected_photo,
                                          score,
                                          self.get_method())
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

    def get_method(self):
        return self.distance_var.get()

    def get_features_extractor(self):
        return self.features_var.get()

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
            self.images[n].create_line(i, self.dim, i, self.dim - volume,
                                       fill="blue")
        out = np.hstack(out)
        answare, person = get_audio_score(out)
        self.text_label["text"] = self.text_generator(answare, person)
        self.update_images(n, SCORES[answare])

    def press_button(self, n):
        th = threading.Thread(target=self.record_audio, args=[n])
        th.start()


def open_image(img_file, size):
    img = Image.open(img_file)
    img.thumbnail((size, size), Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)


def extract_feature(X, sample_rate=44100):

    def energy(input):
    	return np.sum((input*1.0)**2, keepdims=True)

    def sdev(input):
    	return np.std(input, keepdims=True)

    def aavg(input):
    	return np.mean(np.abs(input), keepdims=True)

    X = X.reshape(sample_rate*2,)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X*1.0, sr=sample_rate,
                                         n_mfcc=40).T,
                    axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                 sr=sample_rate).T,
                     axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
                                                         sr=sample_rate).T,
                       axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    ene = energy(X)
    sd = sdev(X)
    avg = aavg(X)
    return np.concatenate((mfccs, chroma, mel, contrast, tonnetz, ene, sd, avg))


def get_audio_score(audio):
    y1 = model1.predict(audio.reshape(-1, 88200, 1).astype('float32'))

    audio_features = extract_feature(audio)
    y2 = model2.predict(audio_features.reshape(1, -1))

    what = list(map(lambda x: np.argmax(x) if x[np.argmax(x)] > 0.5 else -1,
                    (2*y1[0] + y2[0])/3))
    what_what = {0: 'forse', 1: 'no', 2: 'sì', -1: 'non capisco'}
    answare = list(map(lambda x: what_what[x], what))[0]

    who = list(map(lambda x: np.argmax(x) if x[np.argmax(x)] > 0.5 else -1,
                   np.hstack(((2*y1[1]+y2[1])/3,
                              (2*y1[2]+y2[2])/3,
                              (2*y1[3]+y2[3])/3))))
    #who = list(map(lambda x: np.argmax(x) if x[np.argmax(x)] > 0.8 else -1,
    #               np.hstack((y1[1],
    #                          y1[2],
    #                          y1[3]))))
    who_person = {0: 'Riccardo', 1: 'Federico',
                  2: 'Pranav', -1: 'Unknown'}
    person = list(map(lambda x: who_person[x], who))[0]
    print((2*y1[0] + y2[0])/3)
    print(np.hstack(((2*y1[1]+y2[1])/3,
               (2*y1[2]+y2[2])/3,
               (2*y1[3]+y2[3])/3)))
    print(answare, person)
    return answare, person


if __name__ == "__main__":
    model1 = load_model("audio part/cnn_model.h5")
    model2 = load_model("audio part/no_cnn_model.h5")
    gui = GUI()
