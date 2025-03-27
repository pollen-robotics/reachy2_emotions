#!/usr/bin/env python3
import argparse
import json
import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
from typing import Optional, Tuple
from reachy2_sdk import ReachySDK  # type: ignore
import bisect

# For the Flask server mode:
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib
import traceback
import pathlib

# ------------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Folder with recordings (JSON + corresponding WAV files)
RECORD_FOLDER = pathlib.Path(__file__).resolve().parent.parent / "data" / "recordings"

"""
TODOs
- Maybe movements should start a bit after 1.6s. Because at that time most of them are static. So coming from a goto makes a weird stop.
- IDLE -> weird looking small pause with 0 movements -> movement     
"""

# ------------------------------------------------------------------------------
# Helper functions

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
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and f.endswith('.json')]
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
    timeframe = (data["time"][-1] - data["time"][0])/len(data["time"])
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

    distance_l_arm = np.linalg.norm(
        first_l_arm_pose[:3, 3] - current_l_arm_pose[:3, 3])
    distance_r_arm = np.linalg.norm(
        first_r_arm_pose[:3, 3] - current_r_arm_pose[:3, 3])

    return np.max([distance_l_arm, distance_r_arm])

def joint_distance_with_new_pose(reachy: ReachySDK, data: dict) -> float:
    """Similar to distance_with_new_pose but returns the max angle distance that any joint must travel to reach the new pose.
    """
    max_dist = 0
    for group, joints in [("l_arm", reachy.l_arm.joints),
                          ("r_arm", reachy.r_arm.joints),
                          ("head", reachy.head.joints)]:
        idx=-1
        for name, joint in joints.items():
            idx +=1
            dist = np.abs(joint.present_position - data[group][0][idx])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def play_audio(audio_file: str, audio_device: Optional[str],
               start_event: threading.Event, audio_offset: float, stop_event: threading.Event):
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
        sd.play(data, samplerate=sample_rate, device=audio_device, latency='low')
        
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

# ------------------------------------------------------------------------------
# EmotionPlayer class

class EmotionPlayer:
    """
    This class wraps the replay functionality (motion + audio) for an emotion.
    It supports interruption: a new play request calls stop() on any current playback.
    """
    def __init__(self, ip: str, audio_device: Optional[str],
                 audio_offset: float, record_folder: str,
                 auto_start: bool = True, verbose: bool = True):
        self.ip = ip
        self.audio_device = audio_device
        self.audio_offset = audio_offset
        self.record_folder = record_folder
        self.auto_start = auto_start  # In server mode, auto_start is True (no prompt)
        self.max_joint_speed = 40.0  # degrees per second. Tunned on robot
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        # Create the Reachy instance once here.
        try:
            self.reachy = ReachySDK(host=self.ip)
            
        except Exception as e:
            logging.error("Error connecting to Reachy in constructor: %s", e)
            self.reachy = None
            exit(1)
        try:
            logging.info("Turn on reachy")
            self.reachy.turn_on()
            logging.info("Turn on antennas")
            
            self.reachy.head.r_antenna.turn_on()
            self.reachy.head.l_antenna.turn_on()
            logging.info("Turn on done")   
        except Exception as e:
            logging.error("Error turning on Reachy: %s", e)
            return
        # NEW: add a send lock and idle thread controls.
        self.send_lock = threading.Lock()
        self.idle_thread = None
        self.idle_stop_event = threading.Event()
    
    def play_emotion(self, filename: str):
        """
        Interrupt any ongoing playback and start playing the specified emotion.
        Filename can be provided with or without the ".json" extension.
        """
        with self.lock:
            self.stop()  # Stop current playback if any.
            # Stop idle thread if it's running.
            if self.idle_thread and self.idle_thread.is_alive():
                self.idle_stop_event.set()
                self.idle_thread.join()
                self.idle_stop_event.clear()
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._replay_thread, args=(filename,))
            # Seems to work but it's shaky. The dt of the recordings is quite high (~30Hz), maybe setting a fix dt for speed calculations is bad?
            # self.thread = threading.Thread(target=self._replay_thread_smart_interpol, args=(filename,))
            
            self.thread.start()
    
    def _idle_loop(self, idle_final_positions, idle_params, gripper_params, antenna_params, dt):
        logging.info("Starting idle animation loop.")
        # Define idle animation parameters.
        idle_amplitude = 0.5        # maximum offset magnitude
        idle_amplitude_antenna = 10.0
        idle_amplitude_gripper = 10.0
        idle_start_time = time.time()
        while not self.idle_stop_event.is_set():
            t_idle = time.time() - idle_start_time
            # Update arm and head joints with smooth sinusoidal idle offsets.
            for group, joints in [("l_arm", self.reachy.l_arm.joints),
                                  ("r_arm", self.reachy.r_arm.joints),
                                  ("head", self.reachy.head.joints)]:
                for name, joint in joints.items():
                    freq, phase = idle_params[group][name]
                    offset = idle_amplitude * np.sin(2 * np.pi * freq * t_idle + phase)
                    joint.goal_position = idle_final_positions[group][name] + offset
            # Update grippers.
            for gripper, params in gripper_params.items():
                freq, phase = params
                offset = idle_amplitude_gripper * np.sin(2 * np.pi * freq * t_idle + phase)
                if gripper == "l_hand":
                    self.reachy.l_arm.gripper.goal_position = idle_final_positions["l_hand"] + offset
                else:
                    self.reachy.r_arm.gripper.goal_position = idle_final_positions["r_hand"] + offset
            # Update antennas.
            for antenna, params in antenna_params.items():
                freq, phase = params
                offset = idle_amplitude_antenna * np.sin(2 * np.pi * freq * t_idle + phase)
                if antenna == "l_antenna":
                    self.reachy.head.l_antenna.goal_position = idle_final_positions["l_antenna"] + offset
                else:
                    self.reachy.head.r_antenna.goal_position = idle_final_positions["r_antenna"] + offset
            with self.send_lock:
                self.reachy.send_goal_positions(check_positions=False)
            time.sleep(dt)
        logging.info("Idle animation loop stopped.")
    
    def _replay_thread(self, filename: str):
        logging.info("Starting emotion playback for %s", filename)
        if self.reachy is None:
            logging.error("No valid Reachy instance available.")
            return
        
        # Build full path to the recording.
        if not filename.endswith(".json"):
            filename += ".json"
        path = os.path.join(self.record_folder, filename)
        
        # If the file doesn't exist, try to find the closest match.
        if not os.path.exists(path):
            available_emotions = list_available_emotions(self.record_folder)
            # Remove .json extension from filename for matching.
            base_name = os.path.splitext(filename)[0]
            close_matches = difflib.get_close_matches(base_name, available_emotions, n=1, cutoff=0.6)
            if close_matches:
                new_filename = close_matches[0] + ".json"
                logging.warning("Recording file %s not found; using closest match %s", filename, new_filename)
                filename = new_filename
                path = os.path.join(self.record_folder, filename)
            else:
                logging.error("Recording file %s not found and no close match available.", path)
                return
        try:
            data, timeframe = load_data(path)
        except Exception as e:
            logging.error("Error loading data from %s: %s", path, e)
            return
        
        # Determine corresponding audio file.
        audio_file = os.path.splitext(path)[0] + ".wav"
        audio_available = os.path.exists(audio_file)
        if audio_available:
            logging.info("Found corresponding audio file: %s", audio_file)
        else:
            logging.info("No audio file found; only motion replay will be executed.")
        
        # Check current positions to adapt the duration of the initial move.
        try:
            # max_dist = distance_with_new_pose(self.reachy, data)
            max_dist = joint_distance_with_new_pose(self.reachy, data) # better way imo
            logging.info(f"max_dist = {max_dist}")
            
        except Exception as e:
            logging.error("Error computing distance: %s", e)
            max_dist = 0
        # first_duration = max_dist * 5 # TODO: do 
        first_duration = max_dist / self.max_joint_speed
        logging.info("Computed initial move duration: %.2f seconds", first_duration)

        # For now we set a fixed short duration.
        # first_duration = 0.5
        
        start_event = threading.Event()
        self.audio_thread = None
        if audio_available:
            self.audio_thread = threading.Thread(target=play_audio,
                                            args=(audio_file, self.audio_device,
                                                  start_event, self.audio_offset, self.stop_event))
            self.audio_thread.start()
        
        if not self.auto_start:
            input("Is Reachy ready to move? Press Enter to continue.")
        else:
            logging.info("Auto-start mode: proceeding without user confirmation.")
        # Recordings have a "BIP" at 1.5 seconds, so we start at 1.6 seconds. The sound file has also been trimmed.
        playback_offset = 1.6
        try:
            if first_duration > 0.0:
                current_time = playback_offset
                index = bisect.bisect_right(data["time"], current_time)
                logging.info(f"l_arm goto: {data['l_arm'][index]}")
                self.reachy.l_arm.goto(data["l_arm"][index], duration=first_duration, interpolation_mode="linear")
                logging.info("r_arm goto")
                self.reachy.r_arm.goto(data["r_arm"][index], duration=first_duration, interpolation_mode="linear")
                # self.reachy.l_arm.gripper.set_opening(data["l_hand"][index]) # we need a goto for gripper so it's continuous
                # self.reachy.r_arm.gripper.set_opening(data["r_hand"][index])
                logging.info("head goto")
                self.reachy.head.goto(data["head"][index], duration=first_duration, interpolation_mode="linear") # not using wait=true because it backfires if unreachable
                # Instead, we interpolate the antennas and grippers by hand during first_duration. This also provides the delay needed for the arms+head gotos.
                l_gripper_goal = data["l_hand"][index]
                r_gripper_goal = data["r_hand"][index]
                l_antenna_goal = data["l_antenna"][index]
                r_antenna_goal = data["r_antenna"][index]
                l_gripper_pos = self.reachy.l_arm.gripper.present_position
                r_gripper_pos = self.reachy.r_arm.gripper.present_position
                l_antenna_pos = self.reachy.head.l_antenna.present_position
                r_antenna_pos = self.reachy.head.r_antenna.present_position
                t0 = time.time()
                while time.time() - t0 < first_duration:
                    alpha = (time.time() - t0) / first_duration
                    self.reachy.l_arm.gripper.goal_position = lerp(l_gripper_pos, l_gripper_goal, alpha)
                    self.reachy.r_arm.gripper.goal_position = lerp(r_gripper_pos, r_gripper_goal, alpha)
                    self.reachy.head.l_antenna.goal_position = lerp(l_antenna_pos, l_antenna_goal, alpha)
                    self.reachy.head.r_antenna.goal_position = lerp(r_antenna_pos, r_antenna_goal, alpha)
                    self.reachy.send_goal_positions(check_positions=False)
                    time.sleep(0.01)
            logging.info("First position reached.")
        except Exception as e:
            logging.error("Error moving to initial position: %s", e)
            return
        
        start_event.set()
        
        if self.audio_offset < 0:
            wait_time = abs(self.audio_offset)
            logging.info("Waiting %s seconds before starting motion replay to allow audio to lead.", wait_time)
            interruptible_sleep(wait_time, self.stop_event)
            if self.stop_event.is_set():
                logging.info("Emotion playback interrupted during audio lead wait.")
                return

        dt = timeframe
        
        t0 = time.time() - playback_offset
        
        try:
            while not self.stop_event.is_set():
                current_time = time.time() - t0 # elapsed time since playback started

                # If we've reached or passed the last recorded time, use the final positions.
                if current_time >= data["time"][-1]:
                    logging.info("Reached end of recording; setting final positions.")
                    # Set final positions for each component:
                    for joint, goal in zip(self.reachy.l_arm.joints.values(), data["l_arm"][-1]):
                        joint.goal_position = goal
                    for joint, goal in zip(self.reachy.r_arm.joints.values(), data["r_arm"][-1]):
                        joint.goal_position = goal
                    for joint, goal in zip(self.reachy.head.joints.values(), data["head"][-1]):
                        joint.goal_position = goal

                    self.reachy.l_arm.gripper.goal_position = data["l_hand"][-1]
                    self.reachy.r_arm.gripper.goal_position = data["r_hand"][-1]
                    self.reachy.head.l_antenna.goal_position = data["l_antenna"][-1]
                    self.reachy.head.r_antenna.goal_position = data["r_antenna"][-1]

                    self.reachy.send_goal_positions(check_positions=False)
                    
                    logging.info("Reached end of recording normally, starting idle motion.")

                    # Capture the final positions as a reference.
                    idle_final_positions = {
                        "l_arm": {name: joint.goal_position for name, joint in self.reachy.l_arm.joints.items()},
                        "r_arm": {name: joint.goal_position for name, joint in self.reachy.r_arm.joints.items()},
                        "head": {name: joint.goal_position for name, joint in self.reachy.head.joints.items()},
                        "l_hand": self.reachy.l_arm.gripper.goal_position,
                        "r_hand": self.reachy.r_arm.gripper.goal_position,
                        "l_antenna": self.reachy.head.l_antenna.goal_position,
                        "r_antenna": self.reachy.head.r_antenna.goal_position,
                    }

                    # For each joint, assign a random frequency (Hz) and phase offset.
                    # Note : setting phase at 0 otherwise we have a discontinuity
                    
                    idle_params = {
                        "l_arm": {},
                        "r_arm": {},
                        "head": {}
                    }
                    for group, joints in [("l_arm", self.reachy.l_arm.joints),
                                            ("r_arm", self.reachy.r_arm.joints),
                                            ("head", self.reachy.head.joints)]:
                        for name in idle_final_positions[group]:
                            freq = np.random.uniform(0.1, 0.3)  # smooth oscillation (0.1-0.3 Hz)
                            phase = 0.0 #np.random.uniform(0, 2 * np.pi)
                            idle_params[group][name] = (freq, phase)

                    # Also assign parameters for grippers and antennas.
                    gripper_params = {
                        "l_hand": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_hand": (np.random.uniform(0.1, 0.3), 0.0)
                    }
                    antenna_params = {
                        "l_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_antenna": (np.random.uniform(0.1, 0.3), 0.0)
                    }

                    # Instead of running the idle loop inline, start it in a separate thread.
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(target=self._idle_loop, args=(idle_final_positions, idle_params, gripper_params, antenna_params, 0.01))
                    self.idle_thread.start()
                    break

                # Locate the right interval in the recorded time array.
                # 'index' is the insertion point which gives us the next timestamp.
                index = bisect.bisect_right(data["time"], current_time)
                logging.debug(f"index: {index}, expected index: {current_time/dt:.0f}")
                idx_prev = index - 1 if index > 0 else 0
                idx_next = index if index < len(data["time"]) else idx_prev

                t_prev = data["time"][idx_prev]
                t_next = data["time"][idx_next]

                # Avoid division by zero (if by any chance two timestamps are identical).
                if t_next == t_prev:
                    alpha = 0.0
                else:
                    alpha = (current_time - t_prev) / (t_next - t_prev)

                # Interpolate positions for each joint in left arm, right arm, and head.
                for joint, pos_prev, pos_next in zip(self.reachy.l_arm.joints.values(), data["l_arm"][idx_prev], data["l_arm"][idx_next]):
                    joint.goal_position = lerp(pos_prev, pos_next, alpha)
                for joint, pos_prev, pos_next in zip(self.reachy.r_arm.joints.values(), data["r_arm"][idx_prev], data["r_arm"][idx_next]):
                    joint.goal_position = lerp(pos_prev, pos_next, alpha)
                for joint, pos_prev, pos_next in zip(self.reachy.head.joints.values(), data["head"][idx_prev], data["head"][idx_next]):
                    joint.goal_position = lerp(pos_prev, pos_next, alpha)

                # Similarly interpolate for grippers and antennas.
                self.reachy.l_arm.gripper.goal_position = lerp(data["l_hand"][idx_prev], data["l_hand"][idx_next], alpha)
                self.reachy.r_arm.gripper.goal_position = lerp(data["r_hand"][idx_prev], data["r_hand"][idx_next], alpha)
                self.reachy.head.l_antenna.goal_position = lerp(data["l_antenna"][idx_prev], data["l_antenna"][idx_next], alpha)
                self.reachy.head.r_antenna.goal_position = lerp(data["r_antenna"][idx_prev], data["r_antenna"][idx_next], alpha)

                # Send the updated positions to the robot.
                self.reachy.send_goal_positions(check_positions=False)

                calculation_duration = time.time() - t0 - current_time
                margin = dt - calculation_duration
                if margin > 0:
                    time.sleep(margin)
                                    
            else :
                logging.info("End of the recording. Replay duration: %.2f seconds", time.time() - t0)
        except Exception as e:
            logging.error("Error during replay: %s", e)
        finally:
            logging.info(f"Finally of replay. if self.audio_thread and self.audio_thread.is_alive() = {self.audio_thread and self.audio_thread.is_alive()}")
            if self.audio_thread and self.audio_thread.is_alive():
                # sd.stop()
                self.audio_thread.join()
            logging.info(f"Endend Finally of replay")
            
    def _replay_thread_smart_interpol(self, filename: str):
        logging.info("Starting emotion playback for %s", filename)
        if self.reachy is None:
            logging.error("No valid Reachy instance available.")
            return
        
        # Build full path to the recording.
        if not filename.endswith(".json"):
            filename += ".json"
        path = os.path.join(self.record_folder, filename)
        
        # If the file doesn't exist, try to find the closest match.
        if not os.path.exists(path):
            available_emotions = list_available_emotions(self.record_folder)
            # Remove .json extension from filename for matching.
            base_name = os.path.splitext(filename)[0]
            close_matches = difflib.get_close_matches(base_name, available_emotions, n=1, cutoff=0.6)
            if close_matches:
                new_filename = close_matches[0] + ".json"
                logging.warning("Recording file %s not found; using closest match %s", filename, new_filename)
                filename = new_filename
                path = os.path.join(self.record_folder, filename)
            else:
                logging.error("Recording file %s not found and no close match available.", path)
                return
        try:
            data, timeframe = load_data(path)
        except Exception as e:
            logging.error("Error loading data from %s: %s", path, e)
            return
        
        # Determine corresponding audio file.
        audio_file = os.path.splitext(path)[0] + ".wav"
        audio_available = os.path.exists(audio_file)
        if audio_available:
            logging.info("Found corresponding audio file: %s", audio_file)
        else:
            logging.info("No audio file found; only motion replay will be executed.")
        
        # Check current positions to adapt the duration of the initial move.
        try:
            # max_dist = distance_with_new_pose(self.reachy, data)
            max_dist = joint_distance_with_new_pose(self.reachy, data) # better way imo
            logging.info(f"max_dist = {max_dist}")
            
        except Exception as e:
            logging.error("Error computing distance: %s", e)
            max_dist = 0
        first_duration = max_dist / self.max_joint_speed
        logging.info("Computed initial move duration: %.2f seconds", first_duration)
        
        start_event = threading.Event()
        self.audio_thread = None
        if audio_available:
            self.audio_thread = threading.Thread(target=play_audio,
                                            args=(audio_file, self.audio_device,
                                                  start_event, self.audio_offset, self.stop_event))
            self.audio_thread.start()
        
        if not self.auto_start:
            input("Is Reachy ready to move? Press Enter to continue.")
        else:
            logging.info("Auto-start mode: proceeding without user confirmation.")
        # Recordings have a "BIP" at 1.5 seconds, so we start at 1.6 seconds. The sound file has also been trimmed.
        playback_offset = 1.6
        
        start_event.set()
        
        if self.audio_offset < 0:
            wait_time = abs(self.audio_offset)
            logging.info("Waiting %s seconds before starting motion replay to allow audio to lead.", wait_time)
            interruptible_sleep(wait_time, self.stop_event)
            if self.stop_event.is_set():
                logging.info("Emotion playback interrupted during audio lead wait.")
                return

        try:
            # dt_loop = 0.01
            dt_loop = timeframe
            t0 = time.time() - playback_offset
            # Local helper: update a goal value for a given channel.
            # If joint_index is provided, channel_data is assumed to be a list of lists.
            def update_goal(channel_data, current_time, dt, current_goal, max_speed, joint_index=None):
                # Find recorded target at current_time.
                index = bisect.bisect_right(data["time"], current_time)
                idx_prev = index - 1 if index > 0 else 0
                idx_next = index if index < len(data["time"]) else idx_prev
                t_prev = data["time"][idx_prev]
                t_next = data["time"][idx_next]
                # logging.info(f"t_next - t_prev = {t_next - t_prev}")
                alpha = 0.0 if t_next == t_prev else (current_time - t_prev) / (t_next - t_prev)
                if joint_index is not None:
                    target = lerp(channel_data[idx_prev][joint_index], channel_data[idx_next][joint_index], alpha)
                else:
                    target = lerp(channel_data[idx_prev], channel_data[idx_next], alpha)
                
                # Find recorded target at current_time+dt.
                index_dt = bisect.bisect_right(data["time"], current_time+dt)
                idx_prev_dt = index_dt - 1 if index_dt > 0 else 0
                idx_next_dt = index_dt if index_dt < len(data["time"]) else idx_prev_dt
                t_prev_dt = data["time"][idx_prev_dt]
                t_next_dt = data["time"][idx_next_dt]
                alpha_dt = 0.0 if t_next_dt == t_prev_dt else (current_time + dt - t_prev_dt) / (t_next_dt - t_prev_dt)
                if joint_index is not None:
                    target_next = lerp(channel_data[idx_prev_dt][joint_index], channel_data[idx_next_dt][joint_index], alpha_dt)
                else:
                    target_next = lerp(channel_data[idx_prev_dt], channel_data[idx_next_dt], alpha_dt)
                
                # Recorded speed: the movement requested in the recording.
                rec_speed = 10*(target_next - target) / dt
                # Interpolation speed: the speed required to reach the target from the current goal.
                interp_speed = np.clip((target - current_goal) / dt, -max_speed, max_speed)
                
                # return current_goal + dt * (0+  interp_speed)
                # return current_goal + dt * (rec_speed + interp_speed)
                return current_goal + dt * (rec_speed + 0)
            
            while not self.stop_event.is_set():
                current_time = time.time() - t0  # elapsed time since playback started

                # If we've reached the end of the recording, use final positions.
                if current_time >= data["time"][-1]:
                    logging.info("Reached end of recording; setting final positions.")
                    for joint, goal in zip(self.reachy.l_arm.joints.values(), data["l_arm"][-1]):
                        joint.goal_position = goal
                    for joint, goal in zip(self.reachy.r_arm.joints.values(), data["r_arm"][-1]):
                        joint.goal_position = goal
                    for joint, goal in zip(self.reachy.head.joints.values(), data["head"][-1]):
                        joint.goal_position = goal
                    self.reachy.l_arm.gripper.goal_position = data["l_hand"][-1]
                    self.reachy.r_arm.gripper.goal_position = data["r_hand"][-1]
                    self.reachy.head.l_antenna.goal_position = data["l_antenna"][-1]
                    self.reachy.head.r_antenna.goal_position = data["r_antenna"][-1]
                    self.reachy.send_goal_positions(check_positions=False)
                    
                    logging.info("Reached end of recording normally, starting idle motion.")

                    # Capture the final positions as a reference.
                    idle_final_positions = {
                        "l_arm": {name: joint.goal_position for name, joint in self.reachy.l_arm.joints.items()},
                        "r_arm": {name: joint.goal_position for name, joint in self.reachy.r_arm.joints.items()},
                        "head": {name: joint.goal_position for name, joint in self.reachy.head.joints.items()},
                        "l_hand": self.reachy.l_arm.gripper.goal_position,
                        "r_hand": self.reachy.r_arm.gripper.goal_position,
                        "l_antenna": self.reachy.head.l_antenna.goal_position,
                        "r_antenna": self.reachy.head.r_antenna.goal_position,
                    }

                    # For each joint, assign a random frequency (Hz) and phase offset.
                    # Note : setting phase at 0 otherwise we have a discontinuity
                    
                    idle_params = {
                        "l_arm": {},
                        "r_arm": {},
                        "head": {}
                    }
                    for group, joints in [("l_arm", self.reachy.l_arm.joints),
                                            ("r_arm", self.reachy.r_arm.joints),
                                            ("head", self.reachy.head.joints)]:
                        for name in idle_final_positions[group]:
                            freq = np.random.uniform(0.1, 0.3)  # smooth oscillation (0.1-0.3 Hz)
                            phase = 0.0 #np.random.uniform(0, 2 * np.pi)
                            idle_params[group][name] = (freq, phase)

                    # Also assign parameters for grippers and antennas.
                    gripper_params = {
                        "l_hand": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_hand": (np.random.uniform(0.1, 0.3), 0.0)
                    }
                    antenna_params = {
                        "l_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_antenna": (np.random.uniform(0.1, 0.3), 0.0)
                    }

                    # Instead of running the idle loop inline, start it in a separate thread.
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(target=self._idle_loop, args=(idle_final_positions, idle_params, gripper_params, antenna_params, 0.01))
                    self.idle_thread.start()
                    break

                # Update left arm joints.
                joints = list(self.reachy.l_arm.joints.values())
                for i in range(len(joints)):
                    current_goal = joints[i].goal_position
                    new_goal = update_goal(data["l_arm"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i)
                    joints[i].goal_position = new_goal

                # Update right arm joints.
                joints = list(self.reachy.r_arm.joints.values())
                for i in range(len(joints)):
                    current_goal = joints[i].goal_position
                    new_goal = update_goal(data["r_arm"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i)
                    joints[i].goal_position = new_goal

                # Update head joints.
                joints = list(self.reachy.head.joints.values())
                for i in range(len(joints)):
                    current_goal = joints[i].goal_position
                    new_goal = update_goal(data["head"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i)
                    joints[i].goal_position = new_goal

                # Update left gripper.
                current_goal = self.reachy.l_arm.gripper.goal_position
                new_goal = update_goal(data["l_hand"], current_time, dt_loop, current_goal, self.max_joint_speed)
                self.reachy.l_arm.gripper.goal_position = new_goal

                # Update right gripper.
                current_goal = self.reachy.r_arm.gripper.goal_position
                new_goal = update_goal(data["r_hand"], current_time, dt_loop, current_goal, self.max_joint_speed)
                self.reachy.r_arm.gripper.goal_position = new_goal

                # Update left antenna.
                current_goal = self.reachy.head.l_antenna.goal_position
                new_goal = update_goal(data["l_antenna"], current_time, dt_loop, current_goal, self.max_joint_speed)
                self.reachy.head.l_antenna.goal_position = new_goal

                # Update right antenna.
                current_goal = self.reachy.head.r_antenna.goal_position
                new_goal = update_goal(data["r_antenna"], current_time, dt_loop, current_goal, self.max_joint_speed)
                self.reachy.head.r_antenna.goal_position = new_goal

                with self.send_lock:
                    self.reachy.send_goal_positions(check_positions=False)

                calculation_duration = time.time() - t0 - current_time
                margin = dt_loop - calculation_duration
                # logging.info(f"(all in ms) margin = {margin*1000:.0f}, dt_loop = {dt_loop*1000:.0f}, calculation_duration = {calculation_duration*1000:.0f}")
                
                if margin > 0:
                    time.sleep(margin)

                                    
            else :
                logging.info("End of the recording. Replay duration: %.2f seconds", time.time() - t0)
        except Exception as e:
            logging.error("Error during replay: %s", e)
            # print traceback
            traceback.print_exc()
        finally:
            logging.info(f"Finally of replay. if self.audio_thread and self.audio_thread.is_alive() = {self.audio_thread and self.audio_thread.is_alive()}")
            if self.audio_thread and self.audio_thread.is_alive():
                # sd.stop()
                self.audio_thread.join()
            logging.info(f"Endend Finally of replay")
    
    def stop(self):
        if self.thread and self.thread.is_alive():
            logging.info("Stopping current emotion playback.")
            self.stop_event.set()
            time.sleep(0.1)
            logging.info("Calling stop()")
            
            
            logging.info("calling join()")
            self.thread.join()
            self.thread = None
            logging.info("stop() finished.")
        else:
            logging.info("No active playback to stop.")

# ------------------------------------------------------------------------------
# Play All Available Emotions

def run_all_emotions_mode(ip: str, audio_device: Optional[str], audio_offset: float):
    """
    Mode that plays all available emotions sequentially.
    It prints a big header for each emotion so it stands out among the logs.
    Each emotion is fully played (i.e. its playback thread is joined)
    before moving on to the next one.
    """
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True, verbose=False)
    emotions = list_available_emotions(RECORD_FOLDER)
    if not emotions:
        logging.error("No available emotions found in %s", RECORD_FOLDER)
        return

    for emotion in emotions:
        print("\n" + "="*40)
        print("==== PLAYING EMOTION: {} ====".format(emotion.upper()))
        print("="*40 + "\n")
        player.play_emotion(emotion)
        if player.thread:
            player.thread.join()  # Ensure this emotion is finished before the next
        # Optional short pause between emotions
        time.sleep(0.5)

# ------------------------------------------------------------------------------
# Modes

def run_cli_mode(ip: str, filename: Optional[str],
                 audio_device: Optional[str], audio_offset: float):
    """
    CLI mode (one-shot replay).
    If filename is omitted, the most recent recording is used.
    """
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True)
    if filename is None:
        try:
            last = get_last_recording(RECORD_FOLDER)
            filename = os.path.splitext(last)[0]
        except Exception as e:
            logging.error("Error retrieving last recording: %s", e)
            return
    player.play_emotion(filename)
    if player.thread:
        player.thread.join()

def run_server_mode(ip: str, audio_device: Optional[str],
                    audio_offset: float, flask_port: int):
    """
    Server mode: start a Flask server to listen for emotion requests.
    The available emotions are scanned from the RECORD_FOLDER.
    New requests interrupt current playback.
    """
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True)
    # allowed_emotions = list_available_emotions(RECORD_FOLDER) # enables all possible emotions
    # Allowing only a subset of emotions
    allowed_emotions = ["dodo1", "ecoute2", "fatigue1", "ecoute1", "macarena1", "curieux1", "solitaire1", "ennui2", "fatigue2", "furieux2", "ennui1", "apaisant1", "timide1", "anxiete1", "perdu1", "triste1", "abattu1", "furieux1", "attentif1", "enthousiaste2", "enthousiaste3", "attentif2", "confus1", "penseur1", "oui_triste1", "fier1", "frustre1", "incertain1", "enthousiaste1", "serenite1", "aimant1", "serenite2", "impatient1", "serviable2", "degoute1", "accueillant1", "enjoue1", "mecontent1", "peur2", "mecontent2", "interrogatif2", "non_triste1", "incomprehensif1", "reconnaissant1", "rieur1", "soulagement1", "comprehension1", "enerve2", "impatient2", "non", "serviable1", "patient1", "oui1", "enerve1", "frustre2", "mepris1", "amical1", "non_excite1", "etonne1", "fier2", "emerveille1", "oui_excite1", "resigne1", "interrogatif1", "oups1", "peur1", "surpris1", "rieur2", "comprehension2", "celebrant1"]

    logging.info("Available emotions: %s", allowed_emotions)
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/play_emotion", methods=["POST"])
    def handle_play_emotion():
        data = request.get_json()
        if not data:
            logging.error("No JSON data received.")
            return jsonify({"status": "error", "result": "No JSON data received"}), 400

        input_text = data.get("input_text", "")
        thought_process = data.get("thought_process", "")
        emotion_name = data.get("emotion_name", "")
        logging.info("Executing play_emotion:")
        logging.info("Input text: %s", input_text)
        logging.info("Thought process: %s", thought_process)
        logging.info("Emotion: %s", emotion_name)
        
        # Interrupt current playback and start new one.
        player.play_emotion(emotion_name)

        return jsonify({"status": "success", "result": f"Playing emotion {emotion_name}."}), 200
    
    # Start by playing a short emotion so that the idle state exists ASAP
    player.play_emotion("proud2")
    
    app.run(port=flask_port, host="0.0.0.0")
    
# Print all available emotions
def print_available_emotions() -> None:
    """
    Print all available emotions in the record folder.
    """

    emotions = list_available_emotions(RECORD_FOLDER)
    print("Available emotions:")
    print(emotions)

# ------------------------------------------------------------------------------
# Main entry point
def main():
    parser = argparse.ArgumentParser(
        description="Replay Reachy's movements with recorded audio and/or run a Flask server for emotion requests. Example: \npython3 replay_move.py --ip localhost --filename abattu1"
    )
    parser.add_argument("--ip", type=str, default="localhost",
                        help="IP address of the robot")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the JSON recording file to replay (without .json extension)")
    parser.add_argument("--audio-device", type=str, default=None,
                        help="Identifier of the audio output device")
    parser.add_argument("--audio-offset", type=float, default=0.0,
                        help="Time offset for audio playback relative to motion replay.")
    parser.add_argument("--list-audio-devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--server", action="store_true",
                        help="Run in Flask server mode to accept emotion requests")
    parser.add_argument("--flask-port", type=int, default=5001,
                        help="Port for the Flask server (default: 5001)")
    parser.add_argument("--all-emotions", action="store_true",
                        help="Play all available emotions sequentially.")
    parser.add_argument("--list", action="store_true",
                        help="Print all available emotions")
    
    args = parser.parse_args()
    
    if args.list_audio_devices:
        print(sd.query_devices())
        exit(0)
    
    if args.all_emotions:
        run_all_emotions_mode(args.ip, args.audio_device, args.audio_offset)
    elif args.list:
        print_available_emotions()
    elif args.server:
        run_server_mode(args.ip, args.audio_device, args.audio_offset, args.flask_port)
    elif args.name is not None:
        run_cli_mode(args.ip, args.name, args.audio_device, args.audio_offset)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()