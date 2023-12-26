from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import librosa


def audio_to_spec(path_or_file):

    #trim audio
    y, sr = librosa.load(path_or_file)
    target_dur = 29
    offset = (len(y) - target_dur*sr) // 2
    assert offset>=0, target_dur #AssertionError if duration less than target_dur
    y = y[offset : offset+target_dur*sr]

    #get spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    #preprocess it
    fig, ax = plt.subplots(figsize=(4.32, 2.88)) #432x288 as GTZAN dataset
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    buffer, size = fig.canvas.print_to_buffer() #it's RGBA
    plt.close(fig) #to save memory
    with Image.frombuffer('RGBA', size, buffer) as spec:
        return spec.convert('RGB')


transform = transforms.Compose([
    transforms.Resize([224, 224]), #HxW
    transforms.ToTensor()])