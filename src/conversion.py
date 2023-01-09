"""
Helper module..
"""
import os
import sys
from timeit import default_timer as timer
from utilities.converters import stereo2mono, wav2spectrogram, txt2wav, rename_voiced
from utilities.helpers import clear_folder


def convert_voiced(wav_chunks):
    """
    Functions that:
    1) It renames VOICED DB so each txt filename contains label (healthy/nonhealthy)
    2) It converts VOICED txt data files to WAV. The number of wav files corresponds to wav_chunks parameter.
    3) It converts soundfiles from stereo to mono (disabled now)
    4) It converts mono sound files to spectrograms
    :param wav_chunks: specify the number of chunk for every wav file
    :return: None
    """
    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    os.chdir(sys.path[1])

    ########################################################################
    #                           RENAME VOICED                              #
    ########################################################################

    SOURCE_PATH = PATHS["PATH_VOICED"]
    DESTINATION_PATH = PATHS["PATH_VOICED_RENAMED"]
    print(SOURCE_PATH.resolve())
    rename_voiced(SOURCE_PATH, DESTINATION_PATH)

    ########################################################################
    #                           TXT to WAV                                 #
    ########################################################################
    SOURCE_PATH = PATHS["PATH_VOICED_RENAMED"]
    DESTINATION_PATH = PATHS["PATH_VOICED_WAV"]

    clear_folder(DESTINATION_PATH)

    for sound_file in SOURCE_PATH.glob("*.txt"):
        start = timer()
        txt2wav(sound_file, DESTINATION_PATH, chunks=wav_chunks)
        end = timer()
        print(f"{sound_file.name} conversion: {end-start:.2f} s")

    ########################################################################
    #                        WAV to MONO WAV                               #
    ########################################################################

    # SOURCE_PATH = PATHS["PATH_VOICED"]
    # DESTINATION_PATH = PATHS["PATH_MONO"]
    # # convert stereo soundfiles to mono
    # for sound_file in SOURCE_PATH.glob("*.wav"):
    #     print(sound_file.resolve())
    #     stereo2mono(sound_file, DESTINATION_PATH)

    ########################################################################
    #                        WAV to SPECTROGRAM                             #
    ########################################################################
    SOURCE_PATH = PATHS["PATH_VOICED_WAV"]
    DESTINATION_PATH = PATHS["PATH_SPECTROGRAMS"]

    clear_folder(DESTINATION_PATH)

    for sound_file in SOURCE_PATH.glob("*.wav"):
        start = timer()
        print(f"converting {sound_file.name}")
        wav2spectrogram(sound_file, DESTINATION_PATH, 512)
        end = timer()
        print(f"{sound_file.name} conversion: {end-start:.2f} s")
