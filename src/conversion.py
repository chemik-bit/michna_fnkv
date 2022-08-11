"""
Helper module..
1) It converts soundfiles from stereo to mono
2) It converts mono sound files to spectrograms
3) It renames VOICED DB so each txt filename contains label (healthy/nonhealthy)
4) It converts VOICED txt data files to WAV
"""
from pathlib import Path
from timeit import default_timer as timer
from utilities.converters import stereo2mono, wav2spectrogram, txt2wav, rename_voiced


########################################################################
#                           RENAME VOICED                              #
########################################################################

SOURCE_PATH = Path("../data/voiced")
DESTINATION_PATH = Path("../data/voiced_renamed")

rename_voiced(SOURCE_PATH, DESTINATION_PATH)

########################################################################
#                           TXT to WAV                                 #
########################################################################
SOURCE_PATH = Path("../data/voiced_renamed")
DESTINATION_PATH = Path("../data/voiced_renamed/spectrograms")
for sound_file in SOURCE_PATH.glob("*.txt"):
    start = timer()

    txt2wav(sound_file, SOURCE_PATH, chunks=10)

    end = timer()
    print(f"{sound_file.name} conversion: {end-start:.2f} s")

########################################################################
#                        WAV to MONO WAV                               #
########################################################################

# SOURCE_PATH = Path("../data/voiced")
# DESTINATION_PATH = Path("../data/voiced/mono")
# # convert stereo soundfiles to mono
# for sound_file in SOURCE_PATH.glob("*.wav"):
#     print(sound_file.resolve())
#     stereo2mono(sound_file, DESTINATION_PATH)

########################################################################
#                        WAV to SPECTROGRAM                             #
########################################################################
SOURCE_PATH = Path("../data/voiced_renamed")
DESTINATION_PATH = Path("../data/voiced_renamed/spectrograms")
for sound_file in SOURCE_PATH.glob("*.wav"):
    start = timer()
    print(f"converting {sound_file.name}")
    wav2spectrogram(sound_file, DESTINATION_PATH, 256)
    end = timer()
    print(f"{sound_file.name} conversion: {end-start:.2f} s")


