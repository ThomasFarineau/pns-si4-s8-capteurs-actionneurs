import os
import librosa
import numpy as np
import soundfile as sf


def split_audio_file(audio_file_path, output_folder, output_file_name, split_length=1):
    """
    Split an audio file into smaller segments of a specified length.

    Args:
        audio_file_path (str): The path of the input audio file.
        output_folder (str): The folder where the splitted audio files will be saved.
        output_file_name (str): The base name of the output splitted audio files.
        split_length (int, optional): The length of each segment in seconds. Defaults to 1.
    """
    # Skip if the file has already been splitted
    if output_file_name.startswith("splitted_"):
        print("File already splitted, skipping")
        return

    # Check if splitted file already exists
    if any("splitted_" + output_file_name in f for f in os.listdir(output_folder)):
        print("File already exists, skipping")
        return

    # Load the audio file
    audio_signal, sample_rate = librosa.load(audio_file_path, sr=None)

    # Calculate the amount of samples in each split
    split_length_samples = split_length * sample_rate

    # Trim the audio signal to ensure an integer amount of splits.
    len_audio_signal = len(audio_signal)
    audio_signal = audio_signal[:len_audio_signal - len_audio_signal % split_length_samples]

    # Split the audio signal
    if len(audio_signal) > split_length:
        audio_signal = np.split(audio_signal, len(audio_signal) / split_length_samples)

    # Write the splitted audio files
    number_of_files = str(int(len_audio_signal / sample_rate))
    for index, y_split in enumerate(audio_signal):
        output_file = os.path.join(output_folder, f"splitted_{output_file_name}_{index + 1}_of_{number_of_files}.wav")
        sf.write(output_file, y_split, sample_rate)

    # Remove the original audio file
    #os.remove(audio_file_path)
