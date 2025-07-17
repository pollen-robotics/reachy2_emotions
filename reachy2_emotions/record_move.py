import argparse
import datetime
import json
import os
import pathlib
import threading  # Added for scheduling the beep
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
# from reachy2_sdk import ReachySDK  # type: ignore
# from stewart_little_control import Client
from reachy_mini import ReachyMini



from reachy2_emotions.utils import RECORD_FOLDER


def record(ip: str, filename: str, freq: int, audio_device: str, record_folder: str) -> None:
    mini = ReachyMini()
    
    # This data structure will be populated with the processed recording at the end.
    data: dict = {
        "time": [],
        "reachy_mini": [],
    }

    # --- Setup Audio Recording ---
    sample_rate = 44100  # samples per second (you can adjust as needed)
    channels = 1  # mono recording; use 2 for stereo
    audio_frames = []  # will store chunks of recorded audio

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio callback status:", status)
        # Store a copy of the current audio chunk
        audio_frames.append(indata.copy())

    try:
        audio_stream = sd.InputStream(device=audio_device, channels=channels, samplerate=sample_rate, callback=audio_callback)
        audio_stream.start()
        print("Audio recording started using device:", audio_stream.device)
    except Exception as e:
        print("Error starting audio recording:", e)
        print("Available audio devices:")
        print(sd.query_devices())
        return

    # --- Schedule a beep sound 1 second after start ---
    def beep():
        duration = 0.2  # Beep duration in seconds
        freq_beep = 440  # Frequency in Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        beep_sound = 0.5 * np.sin(2 * np.pi * freq_beep * t)
        sd.play(beep_sound, sample_rate)
        sd.wait()  # Wait until the beep finishes playing

    threading.Timer(1.5, beep).start()

    try:
        t0 = time.time()
        # Start the recording.
        mini.start_recording()
        print("\nRecording started. Press Ctrl+C here to stop recording.")

        while True:
            # Keep the script alive to listen for Ctrl+C
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        # Stop recording and retrieve the logged data
        recorded_motion = mini.stop_recording()
        print(f"\nRecording stopped. {len(recorded_motion)} motion frames captured.")
        print(f"Duration of recording: {time.time() - t0:.2f} seconds")
        
        
        # Populate the 'data' dictionary from the retrieved 'recorded_motion' list
        for frame in recorded_motion:
            data["time"].append(frame.get("time"))
            # Each "reachy_mini" entry will contain the pose data for that frame
            pose_info = {
                'head': frame.get('head'),
                'antennas': frame.get('antennas'),
                'body_yaw': frame.get('body_yaw'),
                'check_collision': frame.get('check_collision'),
            }
            data["reachy_mini"].append(pose_info)
        # ---
        
        # Stop the audio stream
        audio_stream.stop()
        audio_stream.close()
        
        # Create recordings folder if needed
        directory = pathlib.Path(record_folder)
        os.makedirs(directory, exist_ok=True)

        # Save motion data to JSON file
        full_filename = filename + ".json"
        file_path = os.path.join(directory, full_filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Robot motion data saved to {file_path}.")

        # Save recorded audio to a WAV file.
        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=0)
            # Use same base name but with .wav extension.
            audio_filename = os.path.splitext(filename)[0] + ".wav"
            audio_file_path = os.path.join(directory, audio_filename)
            sf.write(audio_file_path, audio_data, sample_rate)
            print(f"Audio data saved to {audio_filename}")
        else:
            print("No audio data was recorded.")


# ------------------------------------------------------------------------------
# Main entry point
def main():
    d = datetime.datetime.now()
    default_filename = f"recording_{d.strftime('%m%d_%H%M')}.json"
    parser = argparse.ArgumentParser(description="Record the movements of Reachy and voice from the microphone.")
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="IP address of the robot",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=default_filename,
        help="Name of the file to save the robot data (audio will use the same base name)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=100,
        help="Frequency of the recording (in Hz)",
    )
    # Audio device selection options:
    parser.add_argument(
        "--audio-device", type=str, default=None, help="Identifier of the audio input device (see --list-audio-devices)"
    )
    parser.add_argument("--list-audio-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument(
        "--record-folder", type=str, default=str(RECORD_FOLDER), help="Folder to store recordings (default: data/recordings)"
    )
    args = parser.parse_args()

    if args.list_audio_devices:
        print(sd.query_devices())
        exit(0)

    record(args.ip, args.name, args.freq, args.audio_device, args.record_folder)


if __name__ == "__main__":
    main()
