import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import librosa

def mp3_to_spec(path_or_file):

    #get spectrogram
    y, sr = librosa.load(path_or_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    #preprocess it
    fig, ax = plt.subplots(figsize=(4.32, 2.88)) #432x288 as GTZAN dataset
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    buffer, size = fig.canvas.print_to_buffer() #it's RGBA
    plt.close() #TODO test if needed during data collection (eg with 30 images)
    with Image.frombuffer('RGBA', size, buffer) as im:
        return im.convert('RGB')