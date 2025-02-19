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

# ------------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Folder with recordings (JSON + corresponding WAV files)
RECORD_FOLDER = "recordings"


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
    timeframe = data["time"][1] - data["time"][0]
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


# def play_audio(audio_file: str, audio_device: Optional[str],
#                start_event: threading.Event, audio_offset: float, stop_event: threading.Event):
#     """
#     Load the recorded audio file and wait for a common start trigger.
#     If audio_offset is positive, delay playback; if negative, start immediately.
#     """
#     try:
#         data, sample_rate = sf.read(audio_file, dtype="float32")
#         if sample_rate != 44100:
#             logging.warning("Recorded sample rate (%s) differs from default (44100).", sample_rate)
#         logging.info("Audio thread ready. Waiting for start trigger...")
#         # Replace blocking wait with an interruptible loop.
#         while not start_event.is_set():
#             if stop_event.is_set():
#                 return
#             time.sleep(0.01)
#         if audio_offset > 0:
#             logging.info("Delaying audio playback for %s seconds.", audio_offset)
#             interruptible_sleep(audio_offset, stop_event)
#         if stop_event.is_set():
#             return
#         logging.info("Starting audio playback on device: %s", audio_device)
#         sd.play(data, samplerate=sample_rate, device=audio_device, latency='low')
#         sd.wait() # TODO I think this is not interuptable :/
#         logging.info("Audio playback finished.")
#     except Exception as e:
#         logging.error("Error during audio playback: %s", e)
#         logging.info("Available audio devices: %s", sd.query_devices())

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
                 auto_start: bool = True):
        self.ip = ip
        self.audio_device = audio_device
        self.audio_offset = audio_offset
        self.record_folder = record_folder
        self.auto_start = auto_start  # In server mode, auto_start is True (no prompt)
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        # Create the Reachy instance once here.
        try:
            self.reachy = ReachySDK(host=self.ip)
        except Exception as e:
            logging.error("Error connecting to Reachy in constructor: %s", e)
            self.reachy = None
    
    def play_emotion(self, filename: str):
        """
        Interrupt any ongoing playback and start playing the specified emotion.
        Filename can be provided with or without the ".json" extension.
        """
        with self.lock:
            self.stop()  # Stop current playback if any.
            self.stop_event.clear()
            time.sleep(1.0)
            self.thread = threading.Thread(target=self._replay_thread, args=(filename,))
            self.thread.start()
    
    def _replay_thread(self, filename: str):
        logging.info("Starting emotion playback for %s", filename)
        if self.reachy is None:
            logging.error("No valid Reachy instance available.")
            return
        # Build full path to the recording.
        if not filename.endswith(".json"):
            filename += ".json"
        path = os.path.join(self.record_folder, filename)
        if not os.path.exists(path):
            logging.error("Recording file %s not found", path)
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
            max_dist = distance_with_new_pose(self.reachy, data)
        except Exception as e:
            logging.error("Error computing distance: %s", e)
            max_dist = 0
        first_duration = max_dist * 5 # TODO: do better
        if first_duration > 1.0:
            first_duration = 1.0
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
        
        try:
            self.reachy.turn_on()
            self.reachy.head.r_antenna.turn_on()
            self.reachy.head.l_antenna.turn_on()
        except Exception as e:
            logging.error("Error turning on Reachy: %s", e)
            return
        
        try:
            self.reachy.l_arm.goto(data["l_arm"][0], duration=first_duration)
            self.reachy.r_arm.goto(data["r_arm"][0], duration=first_duration)
            self.reachy.l_arm.gripper.set_opening(data["l_hand"][0])
            self.reachy.r_arm.gripper.set_opening(data["r_hand"][0])
            self.reachy.head.goto(data["head"][0], duration=first_duration, wait=True)
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
        dt = 1/100.0
        t0 = time.time()
        try:
            while not self.stop_event.is_set():
                current_time = time.time() - t0  # elapsed time since playback started

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
                    logging.info(f"Reached end of recording normally")
                    
                    break

                # Locate the right interval in the recorded time array.
                # 'index' is the insertion point which gives us the next timestamp.
                index = bisect.bisect_right(data["time"], current_time)
                logging.info(f"index: {index}, expected index: {current_time/dt}")
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
                
                # logging.info(f"dt: {dt*1000:.0f}, calculation_duration: {calculation_duration*1000:.0f}, margin: {margin*1000:.0f}")
                                    
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
    # allowed_emotions = list_available_emotions(RECORD_FOLDER) # disabled since we have some bad recordings
    # allowed_emotions = ["attentif1", "attentif2", "accueillant", "non_triste1", "oui_triste2", "frustration", "oui_triste1", "oui_excite2", "accueillant2", "oui_excite1", "non_triste2", "non_excite2", "oui_excite3", "amical1", "accueillant3", "non_excite1", "incertain2", "reconnaissant2"]
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
    
    app.run(port=flask_port, host="0.0.0.0")


# ------------------------------------------------------------------------------
# Main entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay Reachy's movements with recorded audio and/or run a Flask server for emotion requests."
    )
    parser.add_argument("--ip", type=str, default="localhost",
                        help="IP address of the robot")
    parser.add_argument("--filename", type=str, default=None,
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
    args = parser.parse_args()
    
    if args.list_audio_devices:
        print(sd.query_devices())
        exit(0)
    
    if args.server:
        run_server_mode(args.ip, args.audio_device, args.audio_offset, args.flask_port)
    else:
        run_cli_mode(args.ip, args.filename, args.audio_device, args.audio_offset)
