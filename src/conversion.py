"""
Helper module..
1) It converts soundfiles from stereo to mono
2) It converts mono sound files to spectrograms
"""
from pathlib import Path
from timeit import default_timer as timer
from utilities.converters import stereo2mono, wav2spectrogram, txt2wav




# SOURCE_PATH = Path("../data/voiced")
# DESTINATION_PATH = Path("../data/voiced/spectrograms")
# for sound_file in SOURCE_PATH.glob("voice???.txt"):
#
#     txt2wav(sound_file, SOURCE_PATH)
#
#     start = timer()
#     wav2spectrogram(sound_file, DESTINATION_PATH, 256)
#     end = timer()
#     print(f"{sound_file.name} conversion: {end-start:.2f} s")

#
SOURCE_PATH = Path("../data/voiced")
DESTINATION_PATH = Path("../data/voiced/mono")
# convert stereo soundfiles to mono
for sound_file in SOURCE_PATH.glob("*.wav"):
    print(sound_file.resolve())
    stereo2mono(sound_file, DESTINATION_PATH)
# # #
# SOURCE_PATH = Path("../data/voiced")
# DESTINATION_PATH = Path("../data/voiced/spectrograms")
# for sound_file in SOURCE_PATH.glob("*.wav"):
#     start = timer()
#     print(f"converting {sound_file.name}")
#     wav2spectrogram(sound_file, DESTINATION_PATH, 256)
#     end = timer()
#     print(f"{sound_file.name} conversion: {end-start:.2f} s")

#

