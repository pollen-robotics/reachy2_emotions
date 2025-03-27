#!/usr/bin/env python3

import os
import argparse

def verify_pairs(folder: str) -> None:
    if not os.path.isdir(folder):
        print(f"❌ Error: '{folder}' is not a valid directory.")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".json") or f.endswith(".wav")]
    json_files = {f[:-5] for f in files if f.endswith(".json")}
    wav_files = {f[:-4] for f in files if f.endswith(".wav")}

    missing_json = wav_files - json_files
    missing_wav = json_files - wav_files

    print("📂 Files present (both JSON and WAV), sorted alphabetically:")
    for file in sorted(json_files & wav_files):
        print(f"  {file}")

    if missing_json:
        print("\n⚠️  .wav files without corresponding .json:")
        for f in sorted(missing_json):
            print(f"  {f}.wav")

    if missing_wav:
        print("\n⚠️  .json files without corresponding .wav:")
        for f in sorted(missing_wav):
            print(f"  {f}.json")

    if not missing_json and not missing_wav:
        print("\n✅ All files are consistent. No missing pairs detected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify .json/.wav file pairing in a folder.")
    parser.add_argument(
        "--folder", "-f", default="data/recordings",
        help="Path to the folder containing emotion recordings (default: data/recordings)"
    )
    args = parser.parse_args()
    verify_pairs(args.folder)
