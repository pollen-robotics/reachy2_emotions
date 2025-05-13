#!/usr/bin/env python3
import json
import argparse
import logging
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# (PART_JOINT_COUNTS remains the same)
PART_JOINT_COUNTS = {
    "l_arm": 7, "r_arm": 7, "head": 3,
    "l_hand": 1, "r_hand": 1, "l_antenna": 1, "r_antenna": 1,
}
PART_PRETTY_NAMES = {
    "l_arm": "Left Arm", "r_arm": "Right Arm", "head": "Head",
    "l_hand": "Left Hand (Gripper)", "r_hand": "Right Hand (Gripper)",
    "l_antenna": "Left Antenna", "r_antenna": "Right Antenna",
}


DEFAULT_TARGET_HZ = 100
DEFAULT_SG_WINDOW_SECONDS = 0.1
DEFAULT_SG_POLYORDER = 3

def load_raw_recording_data(file_path: str) -> Optional[Dict[str, Any]]:
    # ... (same as before) ...
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"Raw data loaded from {file_path}")
        if "time" not in data or not isinstance(data["time"], list) or len(data["time"]) < 2:
            raise ValueError("Recording must contain a 'time' key with at least two entries.")
        return data
    except Exception as e:
        logging.error(f"Error loading raw data from {file_path}: {e}")
        return None

def process_and_verify_single_joint(
    raw_timestamps: np.ndarray,
    raw_positions: np.ndarray,
    part_name: str,
    joint_idx_0_based: int, # For labeling
    target_hz: int = DEFAULT_TARGET_HZ,
    sg_window_seconds: float = DEFAULT_SG_WINDOW_SECONDS,
    sg_polyorder: int = DEFAULT_SG_POLYORDER,
    plot_verification: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a single joint: interpolates, calculates speed, reconstructs position,
    and optionally plots a verification graph.
    Returns (final_regular_timestamps, regular_positions_for_output, speeds_for_output)
    The returned arrays are on the regular time grid.
    """
    if len(raw_timestamps) != len(raw_positions):
        logging.error("Timestamp and position arrays have different lengths for plotting.")
        return None, None, None
    if len(raw_timestamps) < max(4, sg_polyorder + 2) :
        logging.warning(f"Not enough data points ({len(raw_timestamps)}) for {part_name} joint {joint_idx_0_based+1}. Skipping.")
        return None, None, None

    # 1. Interpolate to a regular time grid
    try:
        t_min_raw, t_max_raw = raw_timestamps.min(), raw_timestamps.max()
        if t_max_raw <= t_min_raw:
             logging.warning(f"Timestamp range invalid for {part_name} joint {joint_idx_0_based+1}. Skipping.")
             return None, None, None

        num_regular_samples = int((t_max_raw - t_min_raw) * target_hz) + 1
        regular_timestamps = np.linspace(t_min_raw, t_max_raw, num=max(2, num_regular_samples))
        dt_regular = regular_timestamps[1] - regular_timestamps[0]

        interp_func_pos_regular = interp1d(raw_timestamps, raw_positions, kind='cubic', fill_value="extrapolate", bounds_error=False)
        regular_positions = interp_func_pos_regular(regular_timestamps)
    except Exception as e:
        logging.error(f"Error during interpolation for {part_name} joint {joint_idx_0_based+1}: {e}")
        return None, None, None

    # 2. Apply Savitzky-Golay Filter for Speed
    sg_window_samples = int(sg_window_seconds / dt_regular)
    if sg_window_samples % 2 == 0: sg_window_samples += 1
    sg_window_samples = max(sg_window_samples, sg_polyorder + 2 if sg_polyorder % 2 == 0 else sg_polyorder + 1)


    if len(regular_positions) < sg_window_samples:
        logging.warning(
            f"Not enough interpolated points ({len(regular_positions)}) for Sav-Gol window "
            f"({sg_window_samples}) on {part_name} joint {joint_idx_0_based+1}. No speed calculated."
        )
        # Return interpolated positions, but no speed
        return regular_timestamps, regular_positions, None

    try:
        speeds_on_regular_time = savgol_filter(
            regular_positions,
            window_length=sg_window_samples,
            polyorder=sg_polyorder,
            deriv=1,
            delta=dt_regular
        )
    except Exception as e:
        logging.error(f"Error applying Sav-Gol for {part_name} joint {joint_idx_0_based+1}: {e}")
        return regular_timestamps, regular_positions, None # Still return regular time/pos

    # 3. Reconstruct Position from Speed (on regular time grid)
    # Anchor the integration with the first regular_position value
    reconstructed_pos_regular = cumulative_trapezoid(
        speeds_on_regular_time, regular_timestamps, initial=0
    ) + regular_positions[0]

    if plot_verification:
        logging.info(f"Plotting verification for {part_name} - Joint {joint_idx_0_based + 1}")
        try:
            if "seaborn-v0_8-darkgrid" in plt.style.available:
                plt.style.use("seaborn-v0_8-darkgrid")
            elif "seaborn-darkgrid" in plt.style.available:
                plt.style.use("seaborn-darkgrid")
            elif "ggplot" in plt.style.available:
                plt.style.use("ggplot")
            else:
                logging.info("Common preferred styles not found, using Matplotlib default.")
        except Exception as e:
            logging.warning(f"Could not set a preferred Matplotlib style, using default. Error: {e}")
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        title_suffix = f"{PART_PRETTY_NAMES.get(part_name, part_name)} - Joint {joint_idx_0_based + 1}"

        # --- Position Plot ---
        axs[0].set_title(f"Position Comparison: {title_suffix}")
        # Plot raw original positions at their exact original timestamps
        axs[0].plot(raw_timestamps, raw_positions, 'o', markersize=3, color='gray', alpha=0.5, label='Raw Original Position Samples')

        # Plot regularized positions (used as input to SavGol) at their regular timestamps
        axs[0].plot(regular_timestamps, regular_positions, '-', color='cornflowerblue', linewidth=1.5, label=f'Regularized Position ({target_hz}Hz - Input to SavGol)')

        # Plot reconstructed positions at their regular timestamps
        axs[0].plot(regular_timestamps, reconstructed_pos_regular, '--', color='orange', linewidth=2, label='Reconstructed Position (from integrated speed)')
        axs[0].set_ylabel("Position")
        axs[0].legend()
        axs[0].grid(True, linestyle=':')

        # --- Speed Plot ---
        axs[1].set_title(f"Calculated Speed: {title_suffix}")
        axs[1].plot(regular_timestamps, speeds_on_regular_time, '-', color='green', linewidth=1.5, label='Calculated Speed (SavGol)')
        axs[1].set_ylabel("Speed")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()
        axs[1].grid(True, linestyle=':')

        fig.tight_layout()
        plt.show()

    return regular_timestamps, regular_positions, speeds_on_regular_time


def preprocess_and_save(
    input_file_path: str,
    output_file_path: str,
    target_hz: int,
    sg_window_seconds: float,
    sg_polyorder: int,
    verify_plot_config: Optional[Tuple[str, int]] = None # (part_name, joint_idx_1_based)
):
    raw_data = load_raw_recording_data(input_file_path)
    if not raw_data:
        return

    processed_data_for_json = {}
    raw_timestamps_np = np.array(raw_data["time"])
    master_regular_timestamps: Optional[np.ndarray] = None

    # Determine which joint to plot if verification is requested
    plot_this_part_name: Optional[str] = None
    plot_this_joint_idx_0_based: Optional[int] = None
    if verify_plot_config:
        plot_this_part_name, joint_idx_1_based = verify_plot_config
        if plot_this_part_name not in PART_JOINT_COUNTS:
            logging.error(f"Verification part '{plot_this_part_name}' not recognized. No plot will be generated.")
            plot_this_part_name = None
        else:
            plot_this_joint_idx_0_based = joint_idx_1_based -1
            if not (0 <= plot_this_joint_idx_0_based < PART_JOINT_COUNTS[plot_this_part_name]):
                logging.error(f"Verification joint index {joint_idx_1_based} for part '{plot_this_part_name}' is out of range. No plot.")
                plot_this_part_name = None


    for part_name, num_joints in PART_JOINT_COUNTS.items():
        if part_name not in raw_data:
            continue
        logging.info(f"Processing part: {part_name}")

        part_data_raw_frames = raw_data[part_name] # list of frames [joint_vals]
        
        # Initialize lists for the final processed data for this part (to be stored in JSON)
        # These will be lists of frames, where each frame is a list of joint values
        output_positions_frames = []
        output_speeds_frames = []

        # Temporary storage for individual joint trajectories on their regular time grids
        # before transposing and potential re-interpolation to master_regular_timestamps
        temp_regular_positions_per_joint: List[Optional[np.ndarray]] = [None] * num_joints
        temp_speeds_per_joint: List[Optional[np.ndarray]] = [None] * num_joints
        temp_regular_timestamps_per_joint: List[Optional[np.ndarray]] = [None] * num_joints


        # Extract individual raw joint trajectories
        if num_joints == 1:
            if part_data_raw_frames and isinstance(part_data_raw_frames[0], list):
                raw_positions_single = np.array([item[0] for item in part_data_raw_frames if item])
            else:
                raw_positions_single = np.array(part_data_raw_frames)
            all_raw_joint_trajectories = [raw_positions_single]
        else:
            try:
                all_raw_joint_trajectories = np.array(part_data_raw_frames).T # Now [joint_idx][time_idx]
                if all_raw_joint_trajectories.shape[0] != num_joints:
                    logging.error(f"Part '{part_name}' expected {num_joints} trajectories, got {all_raw_joint_trajectories.shape[0]}. Skipping.")
                    continue
            except Exception as e:
                logging.error(f"Could not transpose data for {part_name}: {e}. Skipping.")
                continue

        # Process each joint trajectory
        for i in range(num_joints):
            raw_pos_this_joint = all_raw_joint_trajectories[i]
            
            # Determine if this is the specific joint to plot for verification
            do_plot_this_specific_joint = (plot_this_part_name == part_name and plot_this_joint_idx_0_based == i)

            reg_t, reg_p, spd = process_and_verify_single_joint(
                raw_timestamps_np, raw_pos_this_joint,
                part_name, i,
                target_hz, sg_window_seconds, sg_polyorder,
                plot_verification=do_plot_this_specific_joint
            )
            temp_regular_timestamps_per_joint[i] = reg_t
            temp_regular_positions_per_joint[i] = reg_p
            temp_speeds_per_joint[i] = spd

            if reg_t is not None and master_regular_timestamps is None:
                master_regular_timestamps = reg_t # Establish the common timeline

        # After processing all joints for this part, ensure all data is on the master_regular_timestamps
        if master_regular_timestamps is None and any(t is not None for t in temp_regular_timestamps_per_joint):
             # Fallback if first joint failed but others succeeded
            for t in temp_regular_timestamps_per_joint:
                if t is not None:
                    master_regular_timestamps = t
                    break
        
        if master_regular_timestamps is None:
            logging.warning(f"Could not establish a master timeline for part {part_name}. Skipping saving this part.")
            continue

        final_positions_this_part_joints: List[List[float]] = []
        final_speeds_this_part_joints: List[List[float]] = []

        for i in range(num_joints):
            reg_t_joint = temp_regular_timestamps_per_joint[i]
            reg_p_joint = temp_regular_positions_per_joint[i]
            spd_joint = temp_speeds_per_joint[i]

            if reg_t_joint is None or reg_p_joint is None: # This joint failed processing
                # Fill with NaNs or empty lists of the correct length if possible, or skip
                # For simplicity, create lists of NaNs matching master_regular_timestamps length
                nan_list = [np.nan] * len(master_regular_timestamps)
                final_positions_this_part_joints.append(nan_list)
                final_speeds_this_part_joints.append(nan_list) # if spd_joint was also None
                continue

            # Re-interpolate to master timeline if necessary
            if not np.array_equal(reg_t_joint, master_regular_timestamps):
                logging.debug(f"Re-interpolating {part_name} joint {i+1} to master timeline.")
                interp_p = interp1d(reg_t_joint, reg_p_joint, kind='cubic', fill_value="extrapolate", bounds_error=False)
                reg_p_joint_aligned = interp_p(master_regular_timestamps)
                final_positions_this_part_joints.append(reg_p_joint_aligned.tolist())

                if spd_joint is not None:
                    interp_s = interp1d(reg_t_joint, spd_joint, kind='cubic', fill_value="extrapolate", bounds_error=False)
                    spd_joint_aligned = interp_s(master_regular_timestamps)
                    final_speeds_this_part_joints.append(spd_joint_aligned.tolist())
                else:
                    final_speeds_this_part_joints.append([np.nan] * len(master_regular_timestamps))
            else: # Already on master timeline
                final_positions_this_part_joints.append(reg_p_joint.tolist())
                if spd_joint is not None:
                    final_speeds_this_part_joints.append(spd_joint.tolist())
                else:
                    final_speeds_this_part_joints.append([np.nan] * len(master_regular_timestamps))
        
        # Transpose back to frame-based lists for JSON
        if final_positions_this_part_joints:
            # Check if all sublists have data before transposing
            if all(len(p_list) == len(master_regular_timestamps) for p_list in final_positions_this_part_joints):
                 output_positions_frames = np.array(final_positions_this_part_joints).T.tolist()
                 processed_data_for_json[f"{part_name}_positions"] = output_positions_frames
            else:
                logging.warning(f"Inconsistent data lengths for positions in part {part_name}. Not saving positions.")


        if final_speeds_this_part_joints:
            if all(len(s_list) == len(master_regular_timestamps) for s_list in final_speeds_this_part_joints):
                output_speeds_frames = np.array(final_speeds_this_part_joints).T.tolist()
                processed_data_for_json[f"{part_name}_speeds"] = output_speeds_frames
            else:
                logging.warning(f"Inconsistent data lengths for speeds in part {part_name}. Not saving speeds.")


    if master_regular_timestamps is not None:
        processed_data_for_json["time_regular"] = master_regular_timestamps.tolist()
        # Save the processed data
        try:
            with open(output_file_path, "w") as f:
                json.dump(processed_data_for_json, f) # Smaller file without indent for data
            logging.info(f"Processed data saved to {output_file_path}")
        except IOError:
            logging.error(f"Error: Could not write processed data to {output_file_path}")
    else:
        logging.error("Failed to process any data or establish a master timeline.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Reachy recording JSON files to calculate joint speeds and optionally verify.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON recording file.")
    parser.add_argument(
        "-o", "--output_file", type=str, default=None,
        help="Path to save the processed JSON file. Defaults to '<input_file_basename>_processed.json'."
    )
    parser.add_argument(
        "--target_hz", type=int, default=DEFAULT_TARGET_HZ,
        help=f"Target sampling frequency for interpolation (default: {DEFAULT_TARGET_HZ} Hz)."
    )
    parser.add_argument(
        "--sg_window_sec", type=float, default=DEFAULT_SG_WINDOW_SECONDS,
        help=f"Savitzky-Golay filter window size in seconds (default: {DEFAULT_SG_WINDOW_SECONDS} s)."
    )
    parser.add_argument(
        "--sg_polyorder", type=int, default=DEFAULT_SG_POLYORDER,
        help=f"Savitzky-Golay filter polynomial order (default: {DEFAULT_SG_POLYORDER})."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output file if it exists."
    )
    parser.add_argument(
        "--verify_plot", type=str, default=None,
        help="Plot verification for a specific joint. Format: 'part_name:joint_number' (e.g., l_arm:3). "
             "If specified, processing might be slower due to plotting."
    )

    args = parser.parse_args()

    output_file_path = args.output_file
    if output_file_path is None:
        base, ext = os.path.splitext(args.input_file)
        output_file_path = f"{base}_processed{ext}"

    if os.path.exists(output_file_path) and not args.force and not args.verify_plot:
        # If only verifying, don't block on existing output file unless force is also given
        logging.error(
            f"Output file {output_file_path} already exists. Use --force to overwrite or specify --verify_plot to only plot."
        )
        return
    
    verify_plot_config_parsed: Optional[Tuple[str, int]] = None
    if args.verify_plot:
        try:
            part_name, joint_idx_str = args.verify_plot.split(':')
            joint_idx_1_based = int(joint_idx_str)
            verify_plot_config_parsed = (part_name, joint_idx_1_based)
        except ValueError:
            logging.error(f"Invalid format for --verify_plot: '{args.verify_plot}'. Expected 'part_name:joint_number'.")
            return


    preprocess_and_save(
        args.input_file, output_file_path,
        args.target_hz, args.sg_window_sec, args.sg_polyorder,
        verify_plot_config=verify_plot_config_parsed
    )

if __name__ == "__main__":
    main()