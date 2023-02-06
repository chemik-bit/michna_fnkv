from scipy import signal
import numpy as np

def octave_filtering(octaves: list, x: list):
    """
    31.25, 93.75, 187.5, 375, 750, 1500, 3000, 6000
    :param octaves:
    :param x: input signal (list)
    :return: filtered signal (list)
    """
    octave_limits = {
        1: [31.25, 62.5],
        2: [62.5, 125],
        3: [125, 250],
        4: [250, 500],
        5: [500, 1000],
        6: [1000, 2000],
        7: [2000, 4000],
        8: [4000, 8000]
    }
    result = [0 for x in range(len(octaves))]

    for idx, octave in enumerate(octaves):
        sos = signal.butter(12, octave_limits[octave], "bandpass", output="sos", fs=8000)
        result[idx] = signal.sosfilt(sos, x)

    return np.asarray([sum(x) for x in zip(*result)])
