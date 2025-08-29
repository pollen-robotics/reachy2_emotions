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
# from stewart_little_control import Client
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation, distance_between_poses



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


def delta_angle_between_mat_rot(P, Q):
    """Compute the angle between two rotation matrices P and Q.

    Think of this as an angular distance in the axis-angle representation.
    """
    # https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R = np.dot(P, Q.T)
    tr = (np.trace(R) - 1) / 2
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr)


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
        self.ms_per_degree = 10
        self.ms_per_magic_mm = 10 
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        try:
            self.reachy_mini = ReachyMini()


        except Exception as e:
            logging.error(f"Error connecting to Reachy in constructor: {e}")
            self.reachy_mini = None
            exit(1)

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
        cur_head_joints, cur_antenna_joints = self.reachy_mini.get_current_joint_positions()
        current_head_pose = self.reachy_mini.get_current_head_pose()
        _, _, distance_to_goal = distance_between_poses(
            np.eye(4), 
            current_head_pose,
        )
        
        antenna_dist = max(abs(cur_antenna_joints - np.array([0,0])))
        antenna_dist = np.rad2deg(antenna_dist)
        antenna_interpol_duration = antenna_dist * self.ms_per_degree / 1000
        print(f"Current antenna distance: {antenna_dist:.2f} degrees")

        head_interpol_duration = distance_to_goal * self.ms_per_magic_mm / 1000
        
        first_duration = max(head_interpol_duration, antenna_interpol_duration)
        logging.info(f"First target pose distance: {distance_to_goal:.0f} in magic mm (1°==1mm) => {head_interpol_duration*1000:.0f} ms")
        logging.info(f"First target antenna distance: {antenna_dist:.0f}° => {antenna_interpol_duration*1000:.0f} ms")
        logging.info(f"==> First target pose duration: {first_duration*1000:.0f} ms")
        

        # Interpolation phase to reach the first target pose.
        self.reachy_mini.goto_target(
            np.eye(4),
            antennas=(0,0),
            body_yaw=0.0,
            duration=first_duration,
            method="minjerk",
        )
        idle_start_time = time.time()
        while not self.idle_stop_event.is_set():
            t_idle = time.time() - idle_start_time
            # self.reachy_mini.send_joints([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0])

            pose = np.eye(4)
            antenna_target = np.deg2rad(15) * np.sin(2 * np.pi * 0.5 * t_idle)
            position = np.array([0.0, 0.0, 0.0 + idle_amplitude * np.sin(2 * np.pi * 0.1 * t_idle)])
            pose[:3, 3] = position
            self.reachy_mini.set_target(head=pose, antennas=np.array([antenna_target, -antenna_target]))
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

        cur_head_joints, cur_antenna_joints = self.reachy_mini.get_current_joint_positions()
        current_head_pose = self.reachy_mini.get_current_head_pose()
        
        _, _, distance_to_goal = distance_between_poses(
            np.array(data["set_target_data"][0]["head"]),
            current_head_pose,
        )
        antenna_dist = max(abs(cur_antenna_joints - np.array(data["set_target_data"][0]["antennas"])))
        antenna_dist = np.rad2deg(antenna_dist)
        antenna_interpol_duration = antenna_dist * self.ms_per_degree / 1000
        print(f"Current antenna distance: {antenna_dist:.2f} degrees")

        head_interpol_duration = distance_to_goal * self.ms_per_magic_mm / 1000
        
        first_duration = max(head_interpol_duration, antenna_interpol_duration)
        logging.info(f"First target pose distance: {distance_to_goal:.0f} in magic mm (1°==1mm) => {head_interpol_duration*1000:.0f} ms")
        logging.info(f"First target antenna distance: {antenna_dist:.0f}° => {antenna_interpol_duration*1000:.0f} ms")
        logging.info(f"==> First target pose duration: {first_duration*1000:.0f} ms")


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
        
        # Interpolation phase to reach the first target pose.
        self.reachy_mini.goto_target(
            np.array(data["set_target_data"][0]["head"]),
            antennas=data["set_target_data"][0]["antennas"],
            body_yaw=data["set_target_data"][0].get("body_yaw", 0.0),
            duration=first_duration,
            method="minjerk",
        )

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
        t0_recording = data["time"][0]

        try:
            while not self.stop_event.is_set():
                current_time = time.time() - t0  # elapsed time since playback started
                # If we've reached or passed the last recorded time, use the final positions.
                if current_time >= (data["time"][-1]-t0_recording):
                    logging.info("Reached end of recording normally, starting idle motion.")

                    # Instead of running the idle loop inline, start it in a separate thread.
                    self.idle_stop_event.clear()
                    self.idle_thread = threading.Thread(
                        target=self._idle_loop, args=()
                    )
                    self.idle_thread.start()
                    break

                # Locate the right interval in the recorded time array.
                # 'index' is the insertion point which gives us the next timestamp.
                index = bisect.bisect_right(data["time"], current_time+t0_recording)
                logging.debug(f"index: {index}, expected index: {current_time/dt:.0f}")
                idx_prev = index - 1 if index > 0 else 0
                idx_next = index if index < len(data["time"]) else idx_prev

                t_prev = data["time"][idx_prev]-t0_recording
                t_next = data["time"][idx_next]-t0_recording

                # Avoid division by zero (if by any chance two timestamps are identical).
                if t_next == t_prev:
                    alpha = 0.0
                else:
                    alpha = (current_time - t_prev) / (t_next - t_prev)
                    
                head_prev = np.array(data["set_target_data"][idx_prev]["head"])
                head_next = np.array(data["set_target_data"][idx_next]["head"])
                antennas_prev = data["set_target_data"][idx_prev]["antennas"]
                antennas_next = data["set_target_data"][idx_next]["antennas"]
                body_yaw_prev = data["set_target_data"][idx_prev].get("body_yaw", 0.0)
                body_yaw_next = data["set_target_data"][idx_next].get("body_yaw", 0.0)
                

                # Interpolate to infer a better position at the current time.
                # Joint interpolations are easy:
                antennas_joints = np.array([lerp(pos_prev, pos_next, alpha) for pos_prev, pos_next in zip(antennas_prev, antennas_next)])
                body_yaw = lerp(body_yaw_prev, body_yaw_next, alpha)
                
                # Head position interpolation is more complex:
                head_pose = linear_pose_interpolation(head_prev, head_next, alpha)
                self.reachy_mini.set_target(head_pose, antennas_joints, body_yaw=body_yaw)

                calculation_duration = time.time() - t0 - current_time
                margin = dt - calculation_duration
                if margin > 0:
                    time.sleep(margin)

            else:
                logging.info(f"End of the recording. Replay duration: {time.time() - t0:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during replay: {e}")
            #traceback:
            logging.error(traceback.format_exc())
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

def run_scripted_emotions_mode(ip: str, audio_device: Optional[str], audio_offset: float):
    player = EmotionPlayer(ip, audio_device, audio_offset, RECORD_FOLDER, auto_start=True)
    emotions_delay = [("enthusiastic1", 5.5), ("welcoming2", 5.0), ("laughing1", 0.5), ] #("resigned1", 2.0), ] 
    death_look = np.array([
        [ 0.92037855, -0.3900239,   0.02801237, -0.01335223],
        [ 0.39013759,  0.92075533,  0.00151064, -0.00303172],
        [-0.02638172,  0.00953832,  0.99960644,  0.0163515 ],
        [ 0.0,         0.0,         0.0,         1.0       ]
    ])
    
    input("press enter to start scripted emotions")
    
    for emotion, delay in emotions_delay:
        print("\n" + "=" * 40)
        print(f"==== PLAYING EMOTION: {emotion.upper()} ====")
        print("=" * 40 + "\n")
        player.play_emotion(emotion)
        if player.thread:
            player.thread.join()
        time.sleep(delay)  # Wait for the specified delay before the next emotion
        
    with player.lock:
        player.stop()  # Stop current playback if any.
        # Stop idle thread if it's running.
        if player.idle_thread and player.idle_thread.is_alive():
            player.idle_stop_event.set()
            player.idle_thread.join()
            player.idle_stop_event.clear()
        player.stop_event.clear()
    player.reachy_mini.goto_target(head=death_look, antennas=np.array([0.0, 0.0]), body_yaw=0.0, duration=2.0)

    time.sleep(3.0)
        

    player.play_emotion("resigned1")
    if player.thread:
        player.thread.join()
    time.sleep(20000.0)
    
    
    


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
    parser.add_argument("--script", action="store_true", help="Plays a scripted sequence of emotions.")
    parser.add_argument("--list", action="store_true", help="Print all available emotions")

    args = parser.parse_args()

    if args.list_audio_devices:
        print(sd.query_devices())
        exit(0)

    if args.all_emotions:
        run_all_emotions_mode(args.ip, args.audio_device, args.audio_offset)
    elif args.script:
        run_scripted_emotions_mode(args.ip, args.audio_device, args.audio_offset)
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
