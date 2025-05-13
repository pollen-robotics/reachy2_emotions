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
        max_dist = joint_distance_with_new_pose(self.reachy, data)  # better way imo
        logging.info(f"max_dist = {max_dist}")

    except Exception as e:
        logging.error("Error computing distance: %s", e)
        max_dist = 0
    first_duration = max_dist / self.max_joint_speed
    logging.info("Computed initial move duration: %.2f seconds", first_duration)

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
            index_dt = bisect.bisect_right(data["time"], current_time + dt)
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
            rec_speed = 10 * (target_next - target) / dt
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

            # Update left arm joints.
            joints = list(self.reachy.l_arm.joints.values())
            for i in range(len(joints)):
                current_goal = joints[i].goal_position
                new_goal = update_goal(
                    data["l_arm"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i
                )
                joints[i].goal_position = new_goal

            # Update right arm joints.
            joints = list(self.reachy.r_arm.joints.values())
            for i in range(len(joints)):
                current_goal = joints[i].goal_position
                new_goal = update_goal(
                    data["r_arm"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i
                )
                joints[i].goal_position = new_goal

            # Update head joints.
            joints = list(self.reachy.head.joints.values())
            for i in range(len(joints)):
                current_goal = joints[i].goal_position
                new_goal = update_goal(
                    data["head"], current_time, dt_loop, current_goal, self.max_joint_speed, joint_index=i
                )
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

        else:
            logging.info("End of the recording. Replay duration: %.2f seconds", time.time() - t0)
    except Exception as e:
        logging.error("Error during replay: %s", e)
        # print traceback
        traceback.print_exc()
    finally:
        logging.info(
            f"Finally of replay. if self.audio_thread and self.audio_thread.is_alive() = {self.audio_thread and self.audio_thread.is_alive()}"
        )
        if self.audio_thread and self.audio_thread.is_alive():
            # sd.stop()
            self.audio_thread.join()
        logging.info(f"Endend Finally of replay")