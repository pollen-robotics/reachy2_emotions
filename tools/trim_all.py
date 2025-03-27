#!/usr/bin/env python3

import os
import argparse
import soundfile as sf

def trim_wav_files(folder: str, trim_duration: float = 1.6) -> None:
    """
    Trims the first `trim_duration` seconds from each .wav file in the given folder.
    Overwrites the original files.

    Args:
        folder (str): Directory containing the .wav files.
        trim_duration (float): Duration (in seconds) to trim from start of each file.
    """
    if not os.path.isdir(folder):
        print(f"❌ Error: The folder '{folder}' does not exist or is not a directory.")
        return

    for filename in os.listdir(folder):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(folder, filename)
            try:
                data, samplerate = sf.read(file_path)
                offset_samples = int(trim_duration * samplerate)

                if offset_samples >= len(data):
                    print(f"⚠️  Skipping {filename}: shorter than {trim_duration} seconds.")
                    continue

                trimmed_data = data[offset_samples:]
                sf.write(file_path, trimmed_data, samplerate)
                print(f"✅ Trimmed {trim_duration:.2f}s from {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trim the beginning of each .wav file in a folder."
    )
    parser.add_argument(
        '--folder', '-f',
        default='data/recordings',
        help="Path to folder containing .wav files (default: data/recordings)"
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=1.6,
        help="Trim duration in seconds (default: 1.6)"
    )
    args = parser.parse_args()
    trim_wav_files(args.folder, args.duration)
