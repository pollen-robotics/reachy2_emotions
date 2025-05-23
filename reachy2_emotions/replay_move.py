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
# from reachy2_sdk import ReachySDK  # type: ignore
from stewart_little_control import Client


from reachy2_emotions.utils import (
    RECORD_FOLDER,
    get_last_recording,
    interruptible_sleep,
    joint_distance_with_new_pose,
    lerp,
    list_available_emotions,
    load_data,
    play_audio,
    print_available_emotions,
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
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        try:
            # self.reachy = ReachySDK(host=self.ip)
            self.reachy_mini = Client()

        except Exception as e:
            logging.error(f"Error connecting to Reachy in constructor: {e}")
            self.reachy_mini = None
            exit(1)
        # try:
        #     self.reachy.turn_on()

        #     self.reachy.head.r_antenna.turn_on()
        #     self.reachy.head.l_antenna.turn_on()
        #     logging.info("Turn ON done")
        # except Exception as e:
        #     logging.error(f"Error turning on Reachy: {e}")
        #     return
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

            self.thread.start()

    def _idle_loop(self):
        logging.info("Starting idle animation loop.")
        # Define idle animation parameters.
        idle_amplitude = 0.01  # maximum offset magnitude
        idle_amplitude_antenna = 10.0
        idle_amplitude_gripper = 10.0
        idle_start_time = time.time()
        while not self.idle_stop_event.is_set():
            t_idle = time.time() - idle_start_time
            print("Idle loop")
            # self.reachy_mini.send_joints([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0])

            pose = np.eye(4)
            position = np.array([0.0, 0.0, 0.0 + idle_amplitude * np.sin(2 * np.pi * 0.1 * t_idle)])
            pose[:3, 3] = position
            self.reachy_mini.send_pose(pose, antennas=[-0.5, 0.5], offset_zero=True)


            # # Update arm and head joints with smooth sinusoidal idle offsets.
            # for group, joints in [
            #     ("l_arm", self.reachy.l_arm.joints),
            #     ("r_arm", self.reachy.r_arm.joints),
            #     ("head", self.reachy.head.joints),
            # ]:
            #     for name, joint in joints.items():
            #         freq, phase = idle_params[group][name]
            #         offset = idle_amplitude * np.sin(2 * np.pi * freq * t_idle + phase)
            #         joint.goal_position = idle_final_positions[group][name] + offset
            # # Update grippers.
            # for gripper, params in gripper_params.items():
            #     freq, phase = params
            #     offset = idle_amplitude_gripper * np.sin(2 * np.pi * freq * t_idle + phase)
            #     if gripper == "l_hand":
            #         self.reachy.l_arm.gripper.goal_position = idle_final_positions["l_hand"] + offset
            #     else:
            #         self.reachy.r_arm.gripper.goal_position = idle_final_positions["r_hand"] + offset
            # # Update antennas.
            # for antenna, params in antenna_params.items():
            #     freq, phase = paramse
            #     offset = idle_amplitude_antenna * np.sin(2 * np.pi * freq * t_idle + phase)
            #     if antenna == "l_antenna":
            #         self.reachy.head.l_antenna.goal_position = idle_final_positions["l_antenna"] + offset
            #     else:
            #         self.reachy.head.r_antenna.goal_position = idle_final_positions["r_antenna"] + offset
            # with self.send_lock:
            #     self.reachy.send_goal_positions(check_positions=False)
            time.sleep(0.01)
        logging.info("Idle animation loop stopped.")

    def _replay_thread(self, filename: str):
        logging.info(f"Starting emotion playback for {filename}")
        if self.reachy_mini is None:
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
            logging.debug("No audio file found; only motion replay will be executed.")

        # Check current positions to adapt the duration of the initial move.
        # try:
        #     # max_dist = distance_with_new_pose(self.reachy, data)
        #     max_joint_diff = joint_distance_with_new_pose(self.reachy, data)
        #     first_duration = max_joint_diff / (self.max_joint_speed)

        # except Exception as e:
        #     logging.error(f"Error computing distance: {e}. Using default duration.")
        max_dist = 0
        first_duration = 0.3

        # logging.info(f"Max angle diff: {max_joint_diff:.1f}°, interpolation duration: {first_duration:.1f}s")

        start_event = threading.Event()
        self.audio_thread = None
        print(self.audio_offset)
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
                # self.reachy.l_arm.goto(data["l_arm"][index], duration=first_duration, interpolation_mode="linear")
                # self.reachy.r_arm.goto(data["r_arm"][index], duration=first_duration, interpolation_mode="linear")
                # # self.reachy.l_arm.gripper.set_opening(data["l_hand"][index]) # we need a goto for gripper so it's continuous
                # # self.reachy.r_arm.gripper.set_opening(data["r_hand"][index])
                # self.reachy.head.goto(
                #     data["head"][index], duration=first_duration, interpolation_mode="linear"
                # )  # not using wait=true because it backfires if unreachable
                # # Instead, we interpolate the antennas and grippers by hand during first_duration. This also provides the delay needed for the arms+head gotos.
                # l_gripper_goal = data["l_hand"][index]
                # r_gripper_goal = data["r_hand"][index]
                # l_antenna_goal = data["l_antenna"][index]
                # r_antenna_goal = data["r_antenna"][index]
                reachy_mini_goal_joints = data["reachy_mini"][index]
                reachy_mini_joints = self.reachy_mini.get_joint_positions()
                # l_gripper_pos = self.reachy.l_arm.gripper.present_position
                # r_gripper_pos = self.reachy.r_arm.gripper.present_position
                # l_antenna_pos = self.reachy.head.l_antenna.present_position
                # r_antenna_pos = self.reachy.head.r_antenna.present_position
                t0 = time.time()
                while time.time() - t0 < first_duration:
                    alpha = (time.time() - t0) / first_duration
                    self.reachy_mini.send_joints(
                        [
                            lerp(pos_prev, pos_next, alpha)
                            for pos_prev, pos_next in zip(reachy_mini_joints, reachy_mini_goal_joints)
                        ]
                    )
                    # self.reachy.l_arm.gripper.goal_position = lerp(l_gripper_pos, l_gripper_goal, alpha)
                    # self.reachy.r_arm.gripper.goal_position = lerp(r_gripper_pos, r_gripper_goal, alpha)
                    # self.reachy.head.l_antenna.goal_position = lerp(l_antenna_pos, l_antenna_goal, alpha)
                    # self.reachy.head.r_antenna.goal_position = lerp(r_antenna_pos, r_antenna_goal, alpha)
                    # self.reachy.send_goal_positions(check_positions=False)
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
                    # for joint, goal in zip(self.reachy.l_arm.joints.values(), data["l_arm"][-1]):
                    #     joint.goal_position = goal
                    # for joint, goal in zip(self.reachy.r_arm.joints.values(), data["r_arm"][-1]):
                    #     joint.goal_position = goal
                    # for joint, goal in zip(self.reachy.head.joints.values(), data["head"][-1]):
                    #     joint.goal_position = goal

                    # self.reachy.l_arm.gripper.goal_position = data["l_hand"][-1]
                    # self.reachy.r_arm.gripper.goal_position = data["r_hand"][-1]
                    # self.reachy.head.l_antenna.goal_position = data["l_antenna"][-1]
                    # self.reachy.head.r_antenna.goal_position = data["r_antenna"][-1]

                    # self.reachy.send_goal_positions(check_positions=False)
                    self.reachy_mini.send_joints(data["reachy_mini"][-1])

                    logging.info("Reached end of recording normally, starting idle motion.")

                    # Capture the final positions as a reference.
                    # idle_final_positions = {
                    #     "l_arm": {name: joint.goal_position for name, joint in self.reachy.l_arm.joints.items()},
                    #     "r_arm": {name: joint.goal_position for name, joint in self.reachy.r_arm.joints.items()},
                    #     "head": {name: joint.goal_position for name, joint in self.reachy.head.joints.items()},
                    #     "l_hand": self.reachy.l_arm.gripper.goal_position,
                    #     "r_hand": self.reachy.r_arm.gripper.goal_position,
                    #     "l_antenna": self.reachy.head.l_antenna.goal_position,
                    #     "r_antenna": self.reachy.head.r_antenna.goal_position,
                    # }

                    # # For each joint, assign a random frequency (Hz) and phase offset.
                    # # Note : setting phase at 0 otherwise we have a discontinuity

                    # idle_params = {"l_arm": {}, "r_arm": {}, "head": {}}
                    # for group, joints in [
                    #     ("l_arm", self.reachy.l_arm.joints),
                    #     ("r_arm", self.reachy.r_arm.joints),
                    #     ("head", self.reachy.head.joints),
                    # ]:
                    #     for name in idle_final_positions[group]:
                    #         freq = np.random.uniform(0.1, 0.3)  # smooth oscillation (0.1-0.3 Hz)
                    #         phase = 0.0  # np.random.uniform(0, 2 * np.pi)
                    #         idle_params[group][name] = (freq, phase)

                    # # Also assign parameters for grippers and antennas.
                    # gripper_params = {
                    #     "l_hand": (np.random.uniform(0.1, 0.3), 0.0),
                    #     "r_hand": (np.random.uniform(0.1, 0.3), 0.0),
                    # }
                    # antenna_params = {
                    #     "l_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                    #     "r_antenna": (np.random.uniform(0.1, 0.3), 0.0),
                    # }
                    reachy_mini_goal_joints = [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0]
                    reachy_mini_joints = self.reachy_mini.get_joint_positions()
                    duration = 1
                    t0 = time.time()
                    while time.time() - t0 < duration:
                        alpha = (time.time() - t0) / duration
                        self.reachy_mini.send_joints(
                            [
                                lerp(pos_prev, pos_next, alpha)
                                for pos_prev, pos_next in zip(reachy_mini_joints, reachy_mini_goal_joints)
                            ]
                        )
                        time.sleep(0.01)

                    # # Instead of running the idle loop inline, start it in a separate thread.
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(
                        target=self._idle_loop, args=()
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
                # for joint, pos_prev, pos_next in zip(
                #     self.reachy.l_arm.joints.values(), data["l_arm"][idx_prev], data["l_arm"][idx_next]
                # ):
                #     joint.goal_position = lerp(pos_prev, pos_next, alpha)
                # for joint, pos_prev, pos_next in zip(
                #     self.reachy.r_arm.joints.values(), data["r_arm"][idx_prev], data["r_arm"][idx_next]
                # ):
                #     joint.goal_position = lerp(pos_prev, pos_next, alpha)
                # for joint, pos_prev, pos_next in zip(
                #     self.reachy.head.joints.values(), data["head"][idx_prev], data["head"][idx_next]
                # ):
                #     joint.goal_position = lerp(pos_prev, pos_next, alpha)
                # Interpolate for the reachy_mini joints.
                reachy_mini_joints = data["reachy_mini"][idx_prev]
                reachy_mini_joints_next = data["reachy_mini"][idx_next]
                reachy_mini_joints = [
                    lerp(pos_prev, pos_next, alpha) for pos_prev, pos_next in zip(reachy_mini_joints, reachy_mini_joints_next)
                ]
                self.reachy_mini.send_joints(reachy_mini_joints)

                # Similarly interpolate for grippers and antennas.
                # self.reachy.l_arm.gripper.goal_position = lerp(data["l_hand"][idx_prev], data["l_hand"][idx_next], alpha)
                # self.reachy.r_arm.gripper.goal_position = lerp(data["r_hand"][idx_prev], data["r_hand"][idx_next], alpha)
                # self.reachy.head.l_antenna.goal_position = lerp(data["l_antenna"][idx_prev], data["l_antenna"][idx_next], alpha)
                # self.reachy.head.r_antenna.goal_position = lerp(data["r_antenna"][idx_prev], data["r_antenna"][idx_next], alpha)

                # Send the updated positions to the robot.
                # self.reachy.send_goal_positions(check_positions=False)

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
            logging.debug("End Finally of replay")  # Typo was in original, kept it as per "do only this change"

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
