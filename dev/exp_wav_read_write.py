from scipy.io.wavfile import write
from scipy.io import wavfile
import numpy as np


samplerate = 44100
fs = 100
t = np.linspace(0., 1000, samplerate)
amplitude = np.iinfo(np.int16).max
data = amplitude * np.sin(2. * np.pi * fs * t)
print(data)
write("example.wav", samplerate, data.astype(np.int32))


samplerate, data = wavfile.read("example.wav")
print(data)