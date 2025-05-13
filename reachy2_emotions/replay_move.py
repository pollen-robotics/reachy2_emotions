#!/usr/bin/env python3
import argparse
import bisect
import difflib
import logging
import os
import threading
import time
import traceback
from typing import Optional

import numpy as np
import sounddevice as sd

# For the Flask server mode:
from flask import Flask, jsonify, request
from flask_cors import CORS
from reachy2_sdk import ReachySDK  # type: ignore

from reachy2_emotions.utils import (
    RECORD_FOLDER,
    PART_JOINT_COUNTS,
    get_last_recording,
    interruptible_sleep,
    joint_distance_with_new_pose,
    lerp,
    list_available_emotions,
    load_data,
    play_audio,
    print_available_emotions,
    load_processed_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EmotionPlayer:
    """
    This class wraps the replay functionality (motion + audio) for an emotion.
    It supports interruption: a new play request calls stop() on any current playback.
    """
    def __init__(
        self,
        ip: str,
        audio_device: Optional[str],
        audio_offset: float,
        record_folder: str,
        auto_start: bool = True,
    ):
        self.ip = ip
        self.audio_device = audio_device
        self.audio_offset = audio_offset
        self.record_folder = record_folder
        self.auto_start = auto_start  # In server mode, auto_start is True (no prompt)
        self.max_joint_speed = 40.0  # degrees per second. Tunned on robot
        self.p_gain = 0.5
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        try:
            self.reachy = ReachySDK(host=self.ip)

        except Exception as e:
            logging.error(f"Error connecting to Reachy in constructor: {e}")
            self.reachy = None
            exit(1)
        try:
            self.reachy.turn_on()

            self.reachy.head.r_antenna.turn_on()
            self.reachy.head.l_antenna.turn_on()
            logging.info("Turn ON done")
        except Exception as e:
            logging.error(f"Error turning on Reachy: {e}")
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
            # self.thread = threading.Thread(target=self._replay_thread, args=(filename,))
            self.thread = threading.Thread(target=self._replay_thread_dynamic, args=(filename,))
            

            self.thread.start()

    def _idle_loop(self, idle_final_positions, idle_params, gripper_params, antenna_params, dt):
        logging.info("Starting idle animation loop.")
        # Define idle animation parameters.
        idle_amplitude = 0.5  # maximum offset magnitude
        idle_amplitude_antenna = 10.0
        idle_amplitude_gripper = 10.0
        idle_start_time = time.time()
        while not self.idle_stop_event.is_set():
            t_idle = time.time() - idle_start_time
            # Update arm and head joints with smooth sinusoidal idle offsets.
            for group, joints in [
                ("l_arm", self.reachy.l_arm.joints),
                ("r_arm", self.reachy.r_arm.joints),
                ("head", self.reachy.head.joints),
            ]:
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
        
    def _replay_thread_dynamic(self, filename_base: str): # You can rename this to _replay_thread
        logging.info(f"Starting DYNAMIC emotion playback for {filename_base}")
        if self.reachy is None:
            logging.error("No valid Reachy instance available for dynamic replay.")
            return

        # Path to the PROCESSED file
        processed_filename = filename_base
        if not processed_filename.endswith(".json"):
            processed_filename += ".json"
        
        # Assuming RECORD_FOLDER is where your *processed* files are for this dynamic replay
        # If you have a separate self.processed_record_folder, use that instead.
        # path = os.path.join(self.processed_record_folder, processed_filename) 
        path = os.path.join(RECORD_FOLDER, processed_filename) # Using RECORD_FOLDER as per your last provided snippet

        if not os.path.exists(path):
            # Use the same folder for difflib matching as used for path construction
            available_files_in_folder = [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".json")]
            available_basenames_in_folder = [os.path.splitext(f)[0] for f in available_files_in_folder]
            base_to_match = os.path.splitext(processed_filename)[0]
            close_matches = difflib.get_close_matches(base_to_match, available_basenames_in_folder, n=1, cutoff=0.6)
            
            if close_matches:
                new_filename_base_matched = close_matches[0]
                path = os.path.join(os.path.dirname(path), new_filename_base_matched + ".json") # Use folder from original path attempt
                logging.warning(f"Processed file for '{filename_base}' (as '{processed_filename}') not found directly; using closest match '{new_filename_base_matched}.json' at {path}")
            else:
                logging.error(f"Processed recording file for '{filename_base}' (expected at {path}) not found and no close match available.")
                return

        try:
            processed_data, dt_control_loop, target_hz = load_processed_data(path) 
            if processed_data is None or dt_control_loop is None or target_hz is None:
                raise ValueError("Failed to load or parse processed data correctly (None returned).")
            logging.info(f"Using processed data at {target_hz}Hz (dt={dt_control_loop:.4f}s).")
        except Exception as e:
            logging.error(f"Error loading processed data from {path}: {e}")
            return

        # Audio setup (uses original filename base to find .wav relative to self.record_folder)
        # Ensure self.record_folder is the correct path to original recordings if different from processed.
        original_audio_path_base = os.path.join(self.record_folder, filename_base) 
        audio_file = original_audio_path_base + ".wav"
        audio_available = os.path.exists(audio_file)

        if audio_available:
            logging.debug(f"Found corresponding audio file: {audio_file}")
        else:
            # Fallback: check if audio is next to processed file
            processed_file_dir = os.path.dirname(path)
            audio_file_alt = os.path.join(processed_file_dir, filename_base + ".wav")
            if os.path.exists(audio_file_alt):
                audio_file = audio_file_alt
                audio_available = True
                logging.debug(f"Found corresponding audio file next to processed data: {audio_file}")
            else:
                logging.warning(f"No audio file found for {filename_base} at expected locations. Motion only.")
                audio_file = None

        playback_offset_in_recording = 1.6 
        current_integrated_positions = {} # Keyed by data key from JSON (e.g., "l_arm", "l_hand")
        
        # --- SDK Object Mapping ---
        # This maps JSON data keys to a list of relevant SDK joint objects for multi-joint parts
        sdk_parts_map = {}
        if "l_arm" in processed_data:
            sdk_parts_map["l_arm"] = list(self.reachy.l_arm.joints.values())[:PART_JOINT_COUNTS.get("l_arm", 7)]
        if "r_arm" in processed_data:
            sdk_parts_map["r_arm"] = list(self.reachy.r_arm.joints.values())[:PART_JOINT_COUNTS.get("r_arm", 7)]
        if "head" in processed_data:
            sdk_parts_map["head"]  = list(self.reachy.head.joints.values())[:PART_JOINT_COUNTS.get("head", 3)]
        
        # This maps JSON data keys to a single SDK joint object for single-joint parts
        sdk_single_joint_map = {}
        if "l_hand" in processed_data:
            sdk_single_joint_map["l_hand"] = self.reachy.l_arm.gripper # Assumes gripper is a direct attribute
        if "r_hand" in processed_data:
            sdk_single_joint_map["r_hand"] = self.reachy.r_arm.gripper
        if "l_antenna" in processed_data:
            # Assuming antenna is a named joint in head.joints, not a direct attribute like gripper
            # Adjust if your SDK has direct antenna attributes (e.g., self.reachy.head.l_antenna_obj)
            if "l_antenna" in self.reachy.head.joints:
                 sdk_single_joint_map["l_antenna"] = self.reachy.head.joints["l_antenna"]
            elif hasattr(self.reachy.head, 'l_antenna'): # Fallback if it's a direct attribute
                 sdk_single_joint_map["l_antenna"] = self.reachy.head.l_antenna
        if "r_antenna" in processed_data:
            if "r_antenna" in self.reachy.head.joints:
                sdk_single_joint_map["r_antenna"] = self.reachy.head.joints["r_antenna"]
            elif hasattr(self.reachy.head, 'r_antenna'):
                 sdk_single_joint_map["r_antenna"] = self.reachy.head.r_antenna

        # Initialize current_integrated_positions from robot's current state
        for data_key, sdk_joint_list in sdk_parts_map.items():
            # sdk_parts_map is already filtered by keys in processed_data
            current_integrated_positions[data_key] = [j.present_position for j in sdk_joint_list]
        
        for data_key, sdk_joint_obj in sdk_single_joint_map.items():
            # sdk_single_joint_map is already filtered by keys in processed_data
            current_integrated_positions[data_key] = sdk_joint_obj.present_position
        
        if not current_integrated_positions:
            logging.error(f"No parts from processed data '{filename_base}' match known SDK parts or data is missing/empty. Cannot replay.")
            return

        # --- Audio Thread & Start Synchronization ---
        start_event = threading.Event()
        self.audio_thread = None
        if audio_available and audio_file:
            self.audio_thread = threading.Thread(
                target=play_audio, args=(audio_file, self.audio_device, start_event, self.audio_offset, self.stop_event)
            )
            self.audio_thread.start()
        else:
            if not (audio_available and self.audio_offset < 0): # If audio is not leading, set event
                start_event.set()

        if not self.auto_start:
            input("Dynamic Replay: Is Reachy ready to move? Press Enter to continue.")
        
        if audio_available and self.audio_offset < 0: # Audio leads
            wait_time = abs(self.audio_offset)
            logging.info(f"Motion waiting {wait_time:.2f}s for audio to lead.")
            interruptible_sleep(wait_time, self.stop_event)
            if self.stop_event.is_set():
                if self.audio_thread and self.audio_thread.is_alive(): self.audio_thread.join()
                return
            start_event.set() # Signal that motion "start" (conceptually) can align with audio
        elif not (audio_available and self.audio_offset < 0) : # If audio does not lead
            start_event.set() 


        logging.info("Dynamic replay loop starting...")
        t_start_of_replay_loop = time.time()
        
        time_data_from_file = np.array(processed_data["time"])
        if len(time_data_from_file) == 0:
            logging.error("Time data array in processed file is empty. Cannot replay.")
            if self.audio_thread and self.audio_thread.is_alive(): self.stop_event.set(); self.audio_thread.join()
            return
        max_recording_time_in_file = time_data_from_file[-1]

        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                
                elapsed_since_loop_start = loop_start_time - t_start_of_replay_loop
                current_playback_time_in_recording = elapsed_since_loop_start + playback_offset_in_recording

                if current_playback_time_in_recording >= max_recording_time_in_file:
                    logging.info("Reached end of recorded data for dynamic replay.")
                    self.reachy.send_goal_positions(check_positions=False) # Send final calculated goals
                    break # Exit loop, then transition to idle

                # Interpolation Indexing for processed_data["time"]
                idx_next = bisect.bisect_right(time_data_from_file, current_playback_time_in_recording)
                idx_prev = max(0, idx_next - 1)
                idx_next = min(idx_next, len(time_data_from_file) - 1) 
                if idx_prev == idx_next and idx_prev > 0 : 
                    idx_prev = idx_next -1

                t_prev_in_file = time_data_from_file[idx_prev]
                t_next_in_file = time_data_from_file[idx_next]

                alpha = 0.0
                if (t_next_in_file - t_prev_in_file) > 1e-9:
                    alpha = (current_playback_time_in_recording - t_prev_in_file) / (t_next_in_file - t_prev_in_file)
                alpha = np.clip(alpha, 0.0, 1.0)

                # --- Iterate over parts that were initialized ---
                for part_data_key in current_integrated_positions.keys():
                    try:
                        pos_frames_from_file = processed_data[part_data_key]
                        speed_frames_from_file = processed_data.get(f"{part_data_key}_speed")
                    except KeyError:
                        # This shouldn't happen if current_integrated_positions was built correctly
                        logging.warning(f"Data key '{part_data_key}' (expected from init) not in processed data. Skipping part.")
                        continue
                    
                    # Robustness Check 1: Ensure part's data array is long enough for time indices
                    if not (0 <= idx_prev < len(pos_frames_from_file) and \
                            0 <= idx_next < len(pos_frames_from_file)):
                        logging.error(f"Time index out of bounds for '{part_data_key}' POSITION data. "
                                      f"tp={current_playback_time_in_recording:.3f}, prev_idx={idx_prev}, next_idx={idx_next}, "
                                      f"len(pos_data)={len(pos_frames_from_file)}, len(time)={len(time_data_from_file)}. "
                                      f"Skipping this part for this iteration.")
                        continue 
                    
                    is_multi_joint_part = part_data_key in sdk_parts_map

                    if is_multi_joint_part:
                        sdk_joint_objs = sdk_parts_map[part_data_key]
                        # num_joints_in_this_part is now implicitly len(sdk_joint_objs)
                        # and also PART_JOINT_COUNTS[part_data_key]
                        num_joints_in_this_part = PART_JOINT_COUNTS[part_data_key]


                        frame_prev_pos = pos_frames_from_file[idx_prev]
                        frame_next_pos = pos_frames_from_file[idx_next]

                        # Robustness Check 2: Ensure frames have correct number of joints
                        if not (isinstance(frame_prev_pos, list) and len(frame_prev_pos) == num_joints_in_this_part and \
                                isinstance(frame_next_pos, list) and len(frame_next_pos) == num_joints_in_this_part):
                            logging.error(f"Inconsistent joint count in POSITION frame data for '{part_data_key}'. "
                                          f"Prev frame (idx {idx_prev}, t {t_prev_in_file:.3f}) type: {type(frame_prev_pos)}, len: {len(frame_prev_pos) if isinstance(frame_prev_pos, list) else 'N/A'}. "
                                          f"Next frame (idx {idx_next}, t {t_next_in_file:.3f}) type: {type(frame_next_pos)}, len: {len(frame_next_pos) if isinstance(frame_next_pos, list) else 'N/A'}. "
                                          f"Expected {num_joints_in_this_part}. Skipping this part for this iteration.")
                            continue
                        
                        target_rec_pos_part = [lerp(frame_prev_pos[j], frame_next_pos[j], alpha) for j in range(num_joints_in_this_part)]
                        
                        current_speeds_for_part = [0.0] * num_joints_in_this_part # Default
                        if speed_frames_from_file:
                            if not (0 <= idx_prev < len(speed_frames_from_file) and \
                                    0 <= idx_next < len(speed_frames_from_file)):
                                logging.warning(f"Time index out of bounds for '{part_data_key}' SPEED data. Using zero speed for this iteration.")
                            else:
                                frame_prev_spd = speed_frames_from_file[idx_prev]
                                frame_next_spd = speed_frames_from_file[idx_next]
                                if not (isinstance(frame_prev_spd, list) and len(frame_prev_spd) == num_joints_in_this_part and \
                                        isinstance(frame_next_spd, list) and len(frame_next_spd) == num_joints_in_this_part):
                                    logging.warning(f"Inconsistent joint count in SPEED frame data for '{part_data_key}'. Using zero speed for this iteration.")
                                else:
                                    current_speeds_for_part = [lerp(frame_prev_spd[j], frame_next_spd[j], alpha) for j in range(num_joints_in_this_part)]
                        
                        # Component A: Update integrated positions
                        for j in range(num_joints_in_this_part):
                            current_integrated_positions[part_data_key][j] += current_speeds_for_part[j] * dt_control_loop
                        
                        # Component B: Calculate and saturate correction & Set Final Goal
                        for j in range(num_joints_in_this_part):
                            error = target_rec_pos_part[j] - sdk_joint_objs[j].present_position
                            max_allowed_correction = self.max_joint_speed * dt_control_loop
                            saturated_correction = np.clip(self.p_gain * error, -max_allowed_correction, max_allowed_correction)
                            sdk_joint_objs[j].goal_position = current_integrated_positions[part_data_key][j] + saturated_correction
                    
                    else: # Single joint part
                        sdk_joint_obj = sdk_single_joint_map[part_data_key]
                        target_rec_pos_single = lerp(pos_frames_from_file[idx_prev], pos_frames_from_file[idx_next], alpha)
                        
                        current_speed_single = 0.0 # Default
                        if speed_frames_from_file:
                             if not (0 <= idx_prev < len(speed_frames_from_file) and \
                                     0 <= idx_next < len(speed_frames_from_file)):
                                logging.warning(f"Time index out of bounds for '{part_data_key}' SPEED data (single joint). Using zero speed for this iteration.")
                             else:
                                current_speed_single = lerp(speed_frames_from_file[idx_prev], speed_frames_from_file[idx_next], alpha)
                        
                        # Component A
                        current_integrated_positions[part_data_key] += current_speed_single * dt_control_loop
                        
                        # Component B
                        error = target_rec_pos_single - sdk_joint_obj.present_position
                        max_allowed_correction = self.max_joint_speed * dt_control_loop 
                        saturated_correction = np.clip(self.p_gain * error, -max_allowed_correction, max_allowed_correction)
                        
                        # Final Goal
                        sdk_joint_obj.goal_position = current_integrated_positions[part_data_key] + saturated_correction

                self.reachy.send_goal_positions(check_positions=False)

                loop_end_time = time.time()
                processing_duration = loop_end_time - loop_start_time
                sleep_duration = dt_control_loop - processing_duration
                if sleep_duration > 0:
                    interruptible_sleep(sleep_duration, self.stop_event) # Ensure interruptible_sleep is defined
                elif sleep_duration < -0.001: 
                    logging.warning(f"Dynamic replay loop overran by {-sleep_duration*1000:.2f} ms.")
            
            # --- End of while loop ---
            if self.stop_event.is_set():
                logging.info("Dynamic replay loop interrupted by stop event.")
            else: 
                logging.info(f"Dynamic replay loop finished. Total effective duration from start: {elapsed_since_loop_start:.2f}s")
                logging.info("Transitioning to idle motion...")
                idle_final_positions = {
                    "l_arm": {name: joint.present_position for name, joint in self.reachy.l_arm.joints.items()},
                    "r_arm": {name: joint.present_position for name, joint in self.reachy.r_arm.joints.items()},
                    "head": {name: joint.present_position for name, joint in self.reachy.head.joints.items()},
                    "l_hand": self.reachy.l_arm.gripper.present_position,
                    "r_hand": self.reachy.r_arm.gripper.present_position,
                    "l_antenna": self.reachy.head.l_antenna.present_position,
                    "r_antenna": self.reachy.head.r_antenna.present_position,
                }
                idle_params = {"l_arm": {}, "r_arm": {}, "head": {}}
                for group_key, joints_dict_sdk in [
                    ("l_arm", self.reachy.l_arm.joints), ("r_arm", self.reachy.r_arm.joints), ("head", self.reachy.head.joints),
                ]:
                    for name_sdk in joints_dict_sdk.keys(): # Use name from SDK iteration
                        idle_params[group_key][name_sdk] = (np.random.uniform(0.1, 0.3), 0.0)

                gripper_params = {
                    "l_hand": (np.random.uniform(0.1, 0.3), 0.0), "r_hand": (np.random.uniform(0.1, 0.3), 0.0),
                }
                antenna_params = {
                    "l_antenna": (np.random.uniform(0.1, 0.3), 0.0), "r_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                }
                if hasattr(self, '_idle_loop') and callable(self._idle_loop):
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(
                        target=self._idle_loop, args=(idle_final_positions, idle_params, gripper_params, antenna_params, 0.01)
                    )
                    self.idle_thread.start()
                else:
                    logging.warning("_idle_loop method not found, cannot start idle motion.")


        except Exception as e:
            logging.error(f"Error during dynamic replay for {filename_base}: {e}", exc_info=True)
        finally:
            logging.debug(f"Finally block of dynamic replay thread for {filename_base}.")
            if self.audio_thread and self.audio_thread.is_alive():
                if not self.stop_event.is_set():
                    self.stop_event.set()
                self.audio_thread.join(timeout=1.0) 
            logging.debug(f"Dynamic replay thread method for {filename_base} finished.")

    def _replay_thread(self, filename: str):
        logging.info(f"Starting emotion playback for {filename}")
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
                logging.warning(f"Recording file {filename} not found; using closest match {new_filename}")
                filename = new_filename
                path = os.path.join(self.record_folder, filename)
            else:
                logging.error(f"Recording file {path} not found and no close match available.")
                return
        try:
            data, timeframe = load_data(path)
        except Exception as e:
            logging.error(f"Error loading data from {path}: {e}")
            return

        # Determine corresponding audio file.
        audio_file = os.path.splitext(path)[0] + ".wav"
        audio_available = os.path.exists(audio_file)
        if audio_available:
            logging.debug(f"Found corresponding audio file: {audio_file}")
        else:
            logging.error("No audio file found; only motion replay will be executed.")

        # Check current positions to adapt the duration of the initial move.
        try:
            # max_dist = distance_with_new_pose(self.reachy, data)
            max_joint_diff = joint_distance_with_new_pose(self.reachy, data)
            first_duration = max_joint_diff / (self.max_joint_speed)

        except Exception as e:
            logging.error(f"Error computing distance: {e}. Using default duration.")
            max_dist = 0
            first_duration = 0.3

        logging.info(f"Max angle diff: {max_joint_diff:.1f}°, interpolation duration: {first_duration:.1f}s")

        start_event = threading.Event()
        self.audio_thread = None
        if audio_available:
            self.audio_thread = threading.Thread(
                target=play_audio, args=(audio_file, self.audio_device, start_event, self.audio_offset, self.stop_event)
            )
            self.audio_thread.start()

        if not self.auto_start:
            input("Is Reachy ready to move? Press Enter to continue.")
        else:
            logging.debug("Auto-start mode: proceeding without user confirmation.")
        # Recordings have a "BIP" at 1.5 seconds, so we start at 1.6 seconds. The sound file has also been trimmed.
        playback_offset = 1.6
        try:
            if first_duration > 0.0:
                current_time = playback_offset
                index = bisect.bisect_right(data["time"], current_time)
                self.reachy.l_arm.goto(data["l_arm"][index], duration=first_duration, interpolation_mode="linear")
                self.reachy.r_arm.goto(data["r_arm"][index], duration=first_duration, interpolation_mode="linear")
                # self.reachy.l_arm.gripper.set_opening(data["l_hand"][index]) # we need a goto for gripper so it's continuous
                # self.reachy.r_arm.gripper.set_opening(data["r_hand"][index])
                self.reachy.head.goto(
                    data["head"][index], duration=first_duration, interpolation_mode="linear"
                )  # not using wait=true because it backfires if unreachable
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
            logging.debug("First position reached.")
        except Exception as e:
            logging.error(f"Error moving to initial position: {e}")
            return

        start_event.set()

        if self.audio_offset < 0:
            wait_time = abs(self.audio_offset)
            logging.info(f"Waiting {wait_time} seconds before starting motion replay to allow audio to lead.")
            interruptible_sleep(wait_time, self.stop_event)
            if self.stop_event.is_set():
                logging.info("Emotion playback interrupted during audio lead wait.")
                return

        dt = timeframe

        t0 = time.time() - playback_offset

        try:
            while not self.stop_event.is_set():
                current_time = time.time() - t0  # elapsed time since playback started

                # If we've reached or passed the last recorded time, use the final positions.
                if current_time >= data["time"][-1]:
                    logging.debug("Reached end of recording; setting final positions.")
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

                    idle_params = {"l_arm": {}, "r_arm": {}, "head": {}}
                    for group, joints in [
                        ("l_arm", self.reachy.l_arm.joints),
                        ("r_arm", self.reachy.r_arm.joints),
                        ("head", self.reachy.head.joints),
                    ]:
                        for name in idle_final_positions[group]:
                            freq = np.random.uniform(0.1, 0.3)  # smooth oscillation (0.1-0.3 Hz)
                            phase = 0.0  # np.random.uniform(0, 2 * np.pi)
                            idle_params[group][name] = (freq, phase)

                    # Also assign parameters for grippers and antennas.
                    gripper_params = {
                        "l_hand": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_hand": (np.random.uniform(0.1, 0.3), 0.0),
                    }
                    antenna_params = {
                        "l_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                        "r_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                    }

                    # Instead of running the idle loop inline, start it in a separate thread.
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(
                        target=self._idle_loop, args=(idle_final_positions, idle_params, gripper_params, antenna_params, 0.01)
                    )
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
                for joint, pos_prev, pos_next in zip(
                    self.reachy.l_arm.joints.values(), data["l_arm"][idx_prev], data["l_arm"][idx_next]
                ):
                    joint.goal_position = lerp(pos_prev, pos_next, alpha)
                for joint, pos_prev, pos_next in zip(
                    self.reachy.r_arm.joints.values(), data["r_arm"][idx_prev], data["r_arm"][idx_next]
                ):
                    joint.goal_position = lerp(pos_prev, pos_next, alpha)
                for joint, pos_prev, pos_next in zip(
                    self.reachy.head.joints.values(), data["head"][idx_prev], data["head"][idx_next]
                ):
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

            else:
                logging.info(f"End of the recording. Replay duration: {time.time() - t0:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during replay: {e}")
        finally:
            logging.debug(
                f"Finally of replay. if self.audio_thread and self.audio_thread.is_alive() = {self.audio_thread and self.audio_thread.is_alive()}"
            )
            if self.audio_thread and self.audio_thread.is_alive():
                # sd.stop()
                self.audio_thread.join()
            logging.debug("End Finally of replay")

    def stop(self):
        if self.thread and self.thread.is_alive():
            logging.info("Stopping current emotion playback.")
            self.stop_event.set()
            time.sleep(0.1)
            logging.debug("Calling stop()")

            logging.debug("calling join()")
            self.thread.join()
            self.thread = None
            logging.debug("stop() finished.")
        else:
            logging.debug("No active playback to stop.")


# ------------------------------------------------------------------------------
# Modes


def run_cli_mode(ip: str, filename: Optional[str], audio_device: Optional[str], audio_offset: float):
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
            logging.error(f"Error retrieving last recording: {e}")
            return
    player.play_emotion(filename)
    if player.thread:
        player.thread.join()


def run_server_mode(ip: str, audio_device: Optional[str], audio_offset: float, flask_port: int):
    """
    Server mode: start a Flask server to listen for emotion requests.
    The available emotions are scanned from the RECORD_FOLDER.
    New requests interrupt current playback.
    """
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True)
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
        logging.info(f"Input text: {input_text}")
        logging.info(f"Thought process: {thought_process}")
        logging.info(f"Emotion: {emotion_name}")

        # Interrupt current playback and start new one.
        player.play_emotion(emotion_name)

        return jsonify({"status": "success", "result": f"Playing emotion {emotion_name}."}), 200

    # Start by playing a short emotion so that the idle state exists ASAP
    player.play_emotion("proud2")

    app.run(port=flask_port, host="0.0.0.0")


def run_all_emotions_mode(ip: str, audio_device: Optional[str], audio_offset: float):
    """
    Mode that plays all available emotions sequentially.
    It prints a big header for each emotion so it stands out among the logs.
    Each emotion is fully played (i.e. its playback thread is joined)
    before moving on to the next one.
    """
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True)
    emotions = list_available_emotions(RECORD_FOLDER)
    if not emotions:
        logging.error(f"No available emotions found in {RECORD_FOLDER}")
        return

    for emotion in emotions:
        print("\n" + "=" * 40)
        print(f"==== PLAYING EMOTION: {emotion.upper()} ====")
        print("=" * 40 + "\n")
        player.play_emotion(emotion)
        if player.thread:
            player.thread.join()  # Ensure this emotion is finished before the next
        # Optional short pause between emotions
        time.sleep(0.5)


# ------------------------------------------------------------------------------
# Main entry point
def main():
    parser = argparse.ArgumentParser(
        description="Replay Reachy's movements with recorded audio and/or run a Flask server for emotion requests. Example: \npython3 replay_move.py --ip localhost --filename abattu1"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of the robot")
    parser.add_argument(
        "--name", type=str, default=None, help="Name of the JSON recording file to replay (without .json extension)"
    )
    parser.add_argument("--audio-device", type=str, default=None, help="Identifier of the audio output device")
    parser.add_argument(
        "--audio-offset", type=float, default=0.0, help="Time offset for audio playback relative to motion replay."
    )
    parser.add_argument("--list-audio-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--server", action="store_true", help="Run in Flask server mode to accept emotion requests")
    parser.add_argument("--flask-port", type=int, default=5001, help="Port for the Flask server (default: 5001)")
    parser.add_argument("--all-emotions", action="store_true", help="Play all available emotions sequentially.")
    parser.add_argument("--list", action="store_true", help="Print all available emotions")

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
