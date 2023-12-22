import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import librosa

def audio_to_spec(path_or_file):

    #get spectrogram
    y, sr = librosa.load(path_or_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    #TODO resize and check if <30

    #preprocess it
    fig, ax = plt.subplots(figsize=(4.32, 2.88)) #432x288 as GTZAN dataset
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    buffer, size = fig.canvas.print_to_buffer() #it's RGBA
    plt.close(fig) #to save memory
    with Image.frombuffer('RGBA', size, buffer) as spec:
        return spec.convert('RGB')