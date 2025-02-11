import argparse
import datetime
import json
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from reachy2_sdk import ReachySDK  # type: ignore

def main(ip: str, filename: str, freq: int, audio_device: str):
    # connect to Reachy
    reachy = ReachySDK(host=ip)

    # Data container for robot motion
    data: dict = {
        "time": [],
        "l_arm": [],
        "l_hand": [],
        "r_arm": [],
        "r_hand": [],
        "head": [],
        "l_antenna": [],
        "r_antenna": [],
    }

    # --- Setup Audio Recording ---
    sample_rate = 44100  # samples per second (you can adjust as needed)
    channels = 1         # mono recording; use 2 for stereo
    audio_frames = []    # will store chunks of recorded audio

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio callback status:", status)
        # Store a copy of the current audio chunk
        audio_frames.append(indata.copy())

    try:
        audio_stream = sd.InputStream(
            device=audio_device,
            channels=channels,
            samplerate=sample_rate,
            callback=audio_callback
        )
        audio_stream.start()
        print("Audio recording started using device:", audio_stream.device)
    except Exception as e:
        print("Error starting audio recording:", e)
        print("Available audio devices:")
        print(sd.query_devices())
        return

    #input("Press Enter to start the recording (robot motions and audio).")

    try:
        t0 = time.time()
        print("Recording in progress, press Ctrl+C to stop")

        while True:
            # Get current positions from Reachy
            l_arm = reachy.l_arm.get_current_positions()
            r_arm = reachy.r_arm.get_current_positions()
            head = reachy.head.get_current_positions()
            l_hand = reachy.l_arm.gripper.get_current_opening()
            r_hand = reachy.r_arm.gripper.get_current_opening()
            l_antenna = reachy.head.l_antenna.present_position
            r_antenna = reachy.head.r_antenna.present_position

            # Save timestamp and joint data
            data["time"].append(time.time() - t0)
            data["l_arm"].append(l_arm)
            data["l_hand"].append(l_hand)
            data["r_arm"].append(r_arm)
            data["r_hand"].append(r_hand)
            data["head"].append(head)
            data["l_antenna"].append(l_antenna)
            data["r_antenna"].append(r_antenna)

            time.sleep(1 / freq)
    except KeyboardInterrupt:
        # Stop the audio stream
        audio_stream.stop()
        audio_stream.close()

        # Create recordings folder if needed
        directory = "recordings"
        os.makedirs(directory, exist_ok=True)

        # Save motion data to JSON file
        full_filename = filename + ".json"
        file_path = os.path.join(directory, full_filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Robot motion data saved to {full_filename}.")
        print(f"Time of recording: {time.time() - t0:.2f} seconds")

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

if __name__ == "__main__":
    d = datetime.datetime.now()
    default_filename = f"recording_{d.strftime('%m%d_%H%M')}.json"
    parser = argparse.ArgumentParser(
        description="Record the movements of Reachy and voice from the microphone."
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="IP address of the robot",
    )
    parser.add_argument(
        "--filename",
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
        "--audio-device",
        type=str,
        default=None,
        help="Identifier of the audio input device (see --list-audio-devices)"
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    args = parser.parse_args()

    if args.list_audio_devices:
        print(sd.query_devices())
        exit(0)

    main(args.ip, args.filename, args.freq, args.audio_device)
