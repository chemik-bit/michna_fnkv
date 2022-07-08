"""
Helper module..
1) It converts soundfiles from stereo to mono
2) It converts mono sound files to spectrograms
"""
from pathlib import Path
from timeit import default_timer as timer
from utilities.converters import mono_wav_convert, to_spectrogram


SOURCE_PATH = Path("../data/sound_files")
DESTINATION_PATH = Path("../data/mono")
# convert stereo soundfiles to mono
for sound_file in SOURCE_PATH.iterdir():
    mono_wav_convert(sound_file, DESTINATION_PATH)

SOURCE_PATH = Path("../data/mono")
DESTINATION_PATH = Path("../data/spectrograms")
for sound_file in SOURCE_PATH.iterdir():
    start = timer()
    to_spectrogram(sound_file, DESTINATION_PATH)
    end = timer()
    print(f"{sound_file.name} conversion: {end-start:.2f} s")
