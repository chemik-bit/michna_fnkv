"""
Conversion from stereo wav to mono wav.
"""
from pathlib import Path

from pydub import AudioSegment


def mono_wav_convert(input_path, output_path):
    """
    Converts stereo wav sound file to mono (single channel) wav file.
    :param input_path: path to stereo wav file
    :param output_path: path to save converted mono wav file
    :return: None
    """
    sound = AudioSegment.from_wav(input_path)
    sound = sound.set_channels(1)
    sound.export(output_path.joinpath(input_path.name), format="wav")


if __name__ == "__main__":
    SOURCE_PATH = Path("../data/sound_files")
    DESTINATION_PATH = Path("../data/mono")
    for sound_file in SOURCE_PATH.iterdir():
        mono_wav_convert(sound_file, DESTINATION_PATH)
