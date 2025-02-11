import argparse
import json
import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

from typing import Optional, Tuple
from reachy2_sdk import ReachySDK  # type: ignore


def get_last_recording(folder: str) -> str:
    """Retrieve the most recent JSON recording file from a folder."""
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.endswith('.json')
    ]
    files.sort(key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return files[-1]


def load_data(path: str) -> Tuple[dict, float]:
    """Load the JSON recording and compute the timeframe between frames."""
    with open(path, "r") as f:
        data = json.load(f)
        print(f"Data loaded from {path}")
    if len(data["time"]) < 2:
        raise ValueError("Insufficient time data in the recording.")
    timeframe = data["time"][1] - data["time"][0]
    return data, timeframe


def distance_with_new_pose(reachy: ReachySDK, data: dict) -> float:
    """Compute the maximum Euclidean distance between the current arm poses and the first recorded poses."""
    first_l_arm_pose = reachy.l_arm.forward_kinematics(data["l_arm"][0])
    first_r_arm_pose = reachy.r_arm.forward_kinematics(data["r_arm"][0])

    current_l_arm_pose = reachy.l_arm.forward_kinematics()
    current_r_arm_pose = reachy.r_arm.forward_kinematics()

    distance_l_arm = np.linalg.norm(
        first_l_arm_pose[:3, 3] - current_l_arm_pose[:3, 3]
    )
    distance_r_arm = np.linalg.norm(
        first_r_arm_pose[:3, 3] - current_r_arm_pose[:3, 3]
    )

    return np.max([distance_l_arm, distance_r_arm])


def play_audio(audio_file: str, audio_device: Optional[str],
               start_event: threading.Event, audio_offset: float,
               default_sample_rate: int = 44100):
    """
    Load the recorded audio file and wait for a common start trigger.
    
    If audio_offset is positive, delay playback by that many seconds;
    if negative, start playback immediately.
    """
    try:
        data, sample_rate = sf.read(audio_file, dtype="float32")
        if sample_rate != default_sample_rate:
            print(f"Warning: Recorded sample rate ({sample_rate}) differs from default ({default_sample_rate}).")
        print("Audio thread ready. Waiting for start trigger...")
        
        # Wait for the common start trigger.
        start_event.wait()

        # If the offset is positive, delay the audio playback.
        if audio_offset > 0:
            print(f"Delaying audio playback for {audio_offset} seconds.")
            time.sleep(audio_offset)
        # If negative, start immediately.
        print("Starting audio playback on device:", audio_device)
        sd.play(data, samplerate=sample_rate, device=audio_device, latency='low')
        sd.wait()
        print("Audio playback finished.")
    except Exception as e:
        print("Error during audio playback:", e)
        print("Available audio devices:")
        print(sd.query_devices())


def main(ip: str, filename: Optional[str], audio_device: Optional[str], audio_offset: float):
    # Connect to Reachy.
    reachy = ReachySDK(host=ip)

    # Determine which JSON file to use.
    if filename is None:
        folder = "recordings"
        filename = get_last_recording(folder)
        path = os.path.join(folder, filename)
    else:
        folder = "recordings"
        path = os.path.join(folder, filename+".json")

    data, timeframe = load_data(path)

    # Determine corresponding audio file (same base name with .wav extension)
    audio_file = os.path.splitext(path)[0] + ".wav"
    audio_available = os.path.exists(audio_file)
    if audio_available:
        print(f"Found corresponding audio file: {audio_file}")
    else:
        print("No audio file found. Only motion replay will be executed.")

    # Check current positions to adapt the duration of the initial move.
    max_dist = distance_with_new_pose(reachy, data)
    first_duration = max_dist * 10 if max_dist > 0.2 else 2

    # Create an event to synchronize the start of motion and audio replay.
    start_event = threading.Event()

    # Start audio playback in a separate thread if an audio file is available.
    audio_thread = None
    if audio_available:
        audio_thread = threading.Thread(target=play_audio,
                                        args=(audio_file, audio_device, start_event, audio_offset))
        audio_thread.start()

    input("Is Reachy ready to move? Press Enter to continue.")
    reachy.turn_on()
    reachy.head.r_antenna.turn_on()
    reachy.head.l_antenna.turn_on()

    # Move Reachy to the initial recorded position.
    reachy.l_arm.goto(data["l_arm"][0], duration=first_duration)
    reachy.r_arm.goto(data["r_arm"][0], duration=first_duration)
    reachy.l_arm.gripper.set_opening(data["l_hand"][0])
    reachy.r_arm.gripper.set_opening(data["r_hand"][0])
    reachy.head.goto(data["head"][0], duration=first_duration, wait=True)
    print("First position reached.")

    # Signal the start event.
    start_event.set()
    
    # If the audio offset is negative, wait before starting motion replay
    # so that audio starts playing earlier.
    if audio_offset < 0:
        wait_time = abs(audio_offset)
        print(f"Waiting {wait_time} seconds before starting motion replay to allow audio to lead.")
        time.sleep(wait_time)
    
    t0 = time.time()

    # Start replaying motion data.
    try:
        for ite in range(len(data["time"])):
            start_t = time.time() - t0

            # Update joint goals for left arm, right arm, and head.
            for joint, goal in zip(reachy.l_arm.joints.values(), data["l_arm"][ite]):
                joint.goal_position = goal
            for joint, goal in zip(reachy.r_arm.joints.values(), data["r_arm"][ite]):
                joint.goal_position = goal
            for joint, goal in zip(reachy.head.joints.values(), data["head"][ite]):
                joint.goal_position = goal

            reachy.l_arm.gripper.goal_position = data["l_hand"][ite]
            reachy.r_arm.gripper.goal_position = data["r_hand"][ite]
            reachy.head.l_antenna.goal_position = data["l_antenna"][ite]
            reachy.head.r_antenna.goal_position = data["r_antenna"][ite]

            reachy.send_goal_positions(check_positions=False)

            # Maintain the same timeframe between frames.
            left_time = timeframe - (time.time() - t0 - start_t)
            if left_time > 0:
                time.sleep(left_time)
        else:
            print("End of the recording. Replay duration: {:.2f} seconds".format(time.time() - t0))
    except KeyboardInterrupt:
        print("Replay stopped by the user.")

    # Wait for the audio thread to finish if it was started.
    if audio_thread is not None:
        audio_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay Reachy's movements along with recorded audio (if available). "
                    "Use --audio-offset to adjust the timing: positive delays audio, negative starts it earlier."
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="IP address of the robot"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional name of the JSON recording file to replay"
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Identifier of the audio output device for playback (if needed)"
    )
    parser.add_argument(
        "--audio-offset",
        type=float,
        default=0.0,
        help="Time offset (in seconds) for audio playback relative to motion replay. "
             "Negative means audio starts earlier, positive means audio is delayed."
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

    main(args.ip, args.filename, args.audio_device, args.audio_offset)
