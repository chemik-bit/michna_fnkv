from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy

sos = signal.butter(10, [1000, 2000], 'bandpass', analog=True)


w, h = signal.freqs(sos[0], sos[1])
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

sig = scipy.io.wavfile.read("voice001_nonhealthy_00005.wav")

sos = signal.butter(10, [1000, 2000], 'bandpass', output="sos", fs=8000)
filtered = signal.sosfilt(sos, sig[1])
plt.plot(filtered)

plt.show()


frequencies, times, spectrogram = signal.spectrogram(filtered, 8000, window=np.hamming(256),
                                                         noverlap=256-1)
plt.pcolormesh(times, frequencies, spectrogram, cmap="viridis")
plt.axis("off")
plt.savefig("filtered.png", dpi=300, format="png",bbox_inches='tight', pad_inches=0)
plt.close("all")

frequencies, times, spectrogram = signal.spectrogram(sig[1], 8000, window=np.hamming(256),
                                                         noverlap=256-1)
plt.pcolormesh(times, frequencies, spectrogram, cmap="viridis")
plt.axis("off")
plt.savefig("nonfiltered.png", dpi=300, format="png",bbox_inches='tight', pad_inches=0)
plt.close("all")