import os

# Get all .json and .wav files in the current directory
files = [f for f in os.listdir() if f.endswith(".json") or f.endswith(".wav")]

# Separate JSON and WAV filenames (without extensions)
json_files = {f[:-5] for f in files if f.endswith(".json")}
wav_files = {f[:-4] for f in files if f.endswith(".wav")}

# Check for mismatches
missing_json = wav_files - json_files
missing_wav = json_files - wav_files

# Display matched files
print("📂 Files present (both JSON and WAV), sorted alphabetically:")
for file in sorted(json_files & wav_files):
    print(file)

# Show anomalies
if missing_json:
    print("\n⚠️  .wav files without corresponding .json:", sorted(missing_json))
if missing_wav:
    print("\n⚠️  .json files without corresponding .wav:", sorted(missing_wav))

# Final check message
if not missing_json and not missing_wav:
    print("\n✅ All files are consistent. No missing pairs detected!")
