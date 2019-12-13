import os
import pyaudio
import threading
import numpy as np
import tkinter as tk

import main as constants


PERSON = "Federico"  # <- insert here your name :)
OUT_FOLDER = "./data/audio/"
SEP = ","
DURATION = constants.DURATION
FREQ_AUDIO = constants.FREQ_AUDIO
AUDIO_RANGE = constants.AUDIO_RANGE
AUDIO = pyaudio.PyAudio()


### README:
#
# This utility is just an audio recorder that allow you to store track
# directly in a .csv file so that you do not have to use a real aduio
# recorder :)
# Using it is very simple: you have just to edit this file (setting
# PERSON variable at line 10) and run it.
# An interface with a text entry and a button will open: write what do
# you want to say in the text entry and press the button to record
# your voice (audio augmentation will be done in an analogical way
# adding black metal in background).
# Output file is stored in ./data/audio/<YOUR NAME>.csv
# output audio format is very simple: first column is what you said
# (as text), remaining columns are the audio (44100 samples for 2
# seconds, with values between 0 and 1 - edit <main.py> file to change
# it).
#
### END SERIOUS README
#
# So, in LISP you can do:
# (dolist (line file)
#   (let ((text (car line))
#         (audio (cdr line)))
#     (train-model *model* :x audio :y text)))
# *model* and (train-model ...) constructions are left as exercises to
# the reader.
#
### END README, ENJOY


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DSIM Project - Audio recorder")
        self.root.geometry("320x320")
        self.root.resizable(False, False)
        self.dim_x, self.dim_y = 300, 250
        self.audio_batch = int(FREQ_AUDIO * DURATION / self.dim_x)
        self.out_entry = tk.Entry(self.root)
        self.record_button = tk.Button(self.root,
                                       text="REC.",
                                       command=self.start_rec)
        self.audio_image = tk.Canvas(self.root,
                                     height=self.dim_y, width=self.dim_x,
                                     bg="white")
        self.out_entry.place(height=30, width=230,
                              x=10, y=10)
        self.record_button.place(height=40, width=60,
                                 x=250, y=5)
        self.audio_image.place(x=10, y=55)
        self.root.mainloop()

    def start_rec(self):
        threading.Thread(target=self.record_audio).start()

    def update_image(self, x, batch):
        y0 = self.dim_y / 2
        update = batch[np.argmax(np.abs(batch))] * y0
        self.audio_image.create_line(x, y0, x, y0 + update, fill="blue")
        
    def record_audio(self):
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
            threading.Thread(target=self.update_image,
                             args=[i, data]).start()
        self.audio_image.delete("all")
        write_on_csv(np.hstack(out), self.out_entry.get())


def create_directory_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def write_on_csv(audio, label):
    create_directory_if_not_exists(OUT_FOLDER)
    filename = os.path.join(OUT_FOLDER, PERSON + ".csv")
    audio_str = SEP.join([str(x) for x in audio])
    with open(filename, "a") as out_stream:
        out_stream.write(SEP.join([label, audio_str]) + "\n")

if __name__ == "__main__":
    GUI()
