#!/usr/bin/env python3
import json
import logging
import os
import pathlib
import threading
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# For the Flask server mode:
from reachy2_sdk import ReachySDK  # type: ignore

# Folder with recordings (JSON + corresponding WAV files)
RECORD_FOLDER = pathlib.Path(__file__).resolve().parent.parent / "data" / "recordings"


# Print all available emotions
def print_available_emotions() -> None:
    """
    Print all available emotions in the record folder.
    """

    emotions = list_available_emotions(RECORD_FOLDER)
    print("Available emotions:")
    print(emotions)


def lerp(v0, v1, alpha):
    """Linear interpolation between two values."""
    return v0 + alpha * (v1 - v0)


def interruptible_sleep(duration: float, stop_event: threading.Event):
    """Sleep in small increments while checking if stop_event is set."""
    end_time = time.time() + duration
    while time.time() < end_time:
        if stop_event.is_set():
            break
        time.sleep(0.01)


def list_available_emotions(folder: str) -> list:
    """
    List all available emotions based on the JSON files in the folder.
    The emotion name is the filename without the .json extension.
    """
    emotions = []
    if not os.path.exists(folder):
        logging.error("Record folder %s does not exist.", folder)
        return emotions
    for file in os.listdir(folder):
        if file.endswith(".json"):
            emotion = os.path.splitext(file)[0]
            emotions.append(emotion)
    return sorted(emotions)


def get_last_recording(folder: str) -> str:
    """Retrieve the most recent JSON recording file from a folder."""
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON recordings found in folder.")
    files.sort(key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return files[-1]


def load_data(path: str) -> Tuple[dict, float]:
    """Load the JSON recording and compute the timeframe between frames."""
    with open(path, "r") as f:
        data = json.load(f)
    logging.info("Data loaded from %s", path)
    if len(data["time"]) < 2:
        raise ValueError("Insufficient time data in the recording.")
    timeframe = (data["time"][-1] - data["time"][0]) / len(data["time"])
    return data, timeframe


def distance_with_new_pose(reachy: ReachySDK, data: dict) -> float:
    """
    Compute the maximum Euclidean distance between the current arm poses
    and the first recorded poses.
    """
    first_l_arm_pose = reachy.l_arm.forward_kinematics(data["l_arm"][0])
    first_r_arm_pose = reachy.r_arm.forward_kinematics(data["r_arm"][0])
    current_l_arm_pose = reachy.l_arm.forward_kinematics()
    current_r_arm_pose = reachy.r_arm.forward_kinematics()

    distance_l_arm = np.linalg.norm(first_l_arm_pose[:3, 3] - current_l_arm_pose[:3, 3])
    distance_r_arm = np.linalg.norm(first_r_arm_pose[:3, 3] - current_r_arm_pose[:3, 3])

    return np.max([distance_l_arm, distance_r_arm])


def joint_distance_with_new_pose(reachy: ReachySDK, data: dict) -> float:
    """Similar to distance_with_new_pose but returns the max angle distance that any joint must travel to reach the new pose."""
    max_dist = 0
    for group, joints in [("l_arm", reachy.l_arm.joints), ("r_arm", reachy.r_arm.joints), ("head", reachy.head.joints)]:
        idx = -1
        for name, joint in joints.items():
            idx += 1
            dist = np.abs(joint.present_position - data[group][0][idx])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def play_audio(
    audio_file: str, audio_device: Optional[str], start_event: threading.Event, audio_offset: float, stop_event: threading.Event
):
    """
    Load the recorded audio file and wait for a common start trigger.
    If audio_offset is positive, delay playback; if negative, start immediately.
    """
    try:
        data, sample_rate = sf.read(audio_file, dtype="float32")
        if sample_rate != 44100:
            logging.warning("Recorded sample rate (%s) differs from default (44100).", sample_rate)
        logging.info("Audio thread ready. Waiting for start trigger...")
        # Replace blocking wait with an interruptible loop.
        while not start_event.is_set():
            if stop_event.is_set():
                return
            time.sleep(0.01)
        logging.info("Start trigger received in audio thread.")
        if audio_offset > 0:
            logging.info("Delaying audio playback for %s seconds.", audio_offset)
            interruptible_sleep(audio_offset, stop_event)
        if stop_event.is_set():
            return
        logging.info("Starting audio playback on device: %s", audio_device)
        sd.play(data, samplerate=sample_rate, device=audio_device, latency="low")

        # Compute the duration of the audio in seconds.
        duration = len(data) / sample_rate
        logging.info("Audio playback duration: %.3f seconds", duration)
        start_time = time.time()
        # Instead of sd.wait(), use an interruptible loop.
        while (time.time() - start_time) < duration:
            if stop_event.is_set():
                logging.info("Audio playback interrupted during wait loop.")
                sd.stop()
                return
            time.sleep(0.01)
        sd.stop()
        logging.info("Audio playback finished.")
    except Exception as e:
        logging.error("Error during audio playback: %s", e)
        logging.info("Available audio devices: %s", sd.query_devices())
