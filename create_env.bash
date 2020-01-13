conda create --name dsim \
      python=3.7 \
      numpy=1.17.2 \
      pandas=0.25.1 \
      tensorflow=2.0.0 \
      pillow=6.2.1 \
      opencv=3.4.2 \
      pyaudio=0.2.11 \
      librosa=0.6.3

conda activate dsim
pip install opencv-contrib-python==3.4.2.17
