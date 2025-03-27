#!/usr/bin/env python3
import os
import soundfile as sf

def trim_wav_files(folder: str, trim_duration: float = 1.6) -> None:
    """
    Trims the first `trim_duration` seconds of every .wav file in the specified folder.
    The trimmed audio overwrites the original file.
    
    Args:
        folder (str): Path to the folder containing .wav files.
        trim_duration (float): Duration in seconds to trim from the start of each file.
    """
    if not os.path.isdir(folder):
        print(f"Error: The folder '{folder}' does not exist or is not a directory.")
        return

    for filename in os.listdir(folder):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(folder, filename)
            try:
                # Load the entire WAV file.
                data, samplerate = sf.read(file_path)
                offset_samples = int(trim_duration * samplerate)
                
                if offset_samples >= len(data):
                    print(f"Skipping {filename}: File duration is less than {trim_duration} seconds.")
                    continue
                
                # Trim the first `trim_duration` seconds.
                trimmed_data = data[offset_samples:]
                
                # Overwrite the original file with the trimmed data.
                sf.write(file_path, trimmed_data, samplerate)
                print(f"Trimmed first {trim_duration} seconds from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    folder_path = "./"
    trim_wav_files(folder_path)
