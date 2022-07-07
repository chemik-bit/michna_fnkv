"""
Comparison of matplotlib specgram and scipy spectrogram functions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np


file_path = Path("../data/sound_files/DR00F-CZ-T-36721_2022-05-26_090010.wav")
sound = AudioSegment.from_wav(file_path)
sound = sound.set_channels(1)
sound.export(Path("../data/mono/DR00F-CZ-T-36721_2022-05-26_090010.wav"), format="wav")

file_path = Path("../data/mono/DR00F-CZ-T-36721_2022-05-26_090010.wav")

sample_rate, samples = wavfile.read(file_path)
print(f"samples shape.. {samples.shape}")
print(f"sample rate {sample_rate}")


frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate,
                                                     scaling="spectrum", nfft=None, mode="psd",
                                                     noverlap=128, window=np.hamming(256))

#plt.imshow(spectrogram)
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap="viridis")

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')


plt.figure()
values, ybins, xbins, im = plt.specgram(samples,
                                        mode="psd", Fs=sample_rate, window=np.hamming(256),
                                        scale_by_freq=False, noverlap=128, scale="dB")
plt.figure()
plt.pcolormesh(xbins, ybins, 10 * np.log10(values))
plt.show()
