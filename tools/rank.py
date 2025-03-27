#!/usr/bin/env python3

import os
import argparse
import soundfile as sf

def get_wav_duration(filepath: str) -> float:
    try:
        with sf.SoundFile(filepath) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"❌ Error reading '{filepath}': {e}")
        return -1.0

def rank_wav_files(folder: str) -> None:
    if not os.path.isdir(folder):
        print(f"❌ Error: '{folder}' is not a valid directory.")
        return

    wav_files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    durations = []

    for fname in wav_files:
        path = os.path.join(folder, fname)
        duration = get_wav_duration(path)
        if duration > 0:
            durations.append((duration, fname))

    durations.sort(reverse=True)

    print(f"🎧 .wav durations in '{folder}':\n")
    for duration, fname in durations:
        print(f"{duration:.2f} sec\t{fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank .wav files by duration (descending)."
    )
    parser.add_argument(
        '--folder', '-f',
        default='data/recordings',
        help="Folder containing .wav files (default: data/recordings)"
    )
    args = parser.parse_args()
    rank_wav_files(args.folder)
