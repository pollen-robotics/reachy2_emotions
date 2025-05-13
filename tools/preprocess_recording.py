#!/usr/bin/env python3
import json
import argparse
import logging
import os
import glob # For finding files in a folder
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
DEFAULT_PROCESSED_SUFFIX = ""#"_processed"
DEFAULT_VERIFICATION_PLOT_SUBDIR = "verification_plots"

def load_raw_recording_data(file_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.debug(f"Raw data loaded from {file_path}") # Changed to debug for less verbosity in batch
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
    joint_idx_0_based: int,
    target_hz: int,
    sg_window_seconds: float,
    sg_polyorder: int,
    plot_verification: bool = False,
    plot_save_path: Optional[str] = None,
    show_plot_interactive: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a single joint.
    Returns (regular_timestamps, regular_positions, speeds_on_regular_time)
    """
    # ... (Core logic remains the same as preprocess_recording_v2.py)
    if len(raw_timestamps) != len(raw_positions):
        logging.error("Timestamp and position arrays have different lengths.")
        return None, None, None
    if len(raw_timestamps) < max(4, sg_polyorder + 2) : # Min points for cubic interp and SavGol
        logging.warning(f"Not enough data points ({len(raw_timestamps)}) for {part_name} joint {joint_idx_0_based+1}. Skipping.")
        return None, None, None

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

    sg_window_samples = int(sg_window_seconds / dt_regular)
    if sg_window_samples % 2 == 0: sg_window_samples += 1
    sg_window_samples = max(sg_window_samples, sg_polyorder + 2 if sg_polyorder % 2 == 0 else sg_polyorder + 1)

    if len(regular_positions) < sg_window_samples:
        logging.warning(
            f"Not enough interpolated points ({len(regular_positions)}) for Sav-Gol window "
            f"({sg_window_samples}) on {part_name} joint {joint_idx_0_based+1}. No speed calculated."
        )
        return regular_timestamps, regular_positions, None

    try:
        speeds_on_regular_time = savgol_filter(
            regular_positions, window_length=sg_window_samples, polyorder=sg_polyorder,
            deriv=1, delta=dt_regular
        )
    except Exception as e:
        logging.error(f"Error applying Sav-Gol for {part_name} joint {joint_idx_0_based+1}: {e}")
        return regular_timestamps, regular_positions, None

    if plot_verification:
        reconstructed_pos_regular = cumulative_trapezoid(
            speeds_on_regular_time, regular_timestamps, initial=0
        ) + regular_positions[0]
        logging.info(f"Plotting verification for {part_name} - Joint {joint_idx_0_based + 1}")
        # ... (Plotting code remains the same as v2) ...
        try:
            # Apply a style
            if 'seaborn-v0_8-whitegrid' in plt.style.available: plt.style.use('seaborn-v0_8-whitegrid')
            elif 'seaborn-whitegrid' in plt.style.available: plt.style.use('seaborn-whitegrid')
            elif 'ggplot' in plt.style.available: plt.style.use('ggplot')
        except: pass # Ignore style errors

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        title_suffix = f"{PART_PRETTY_NAMES.get(part_name, part_name)} - Joint {joint_idx_0_based + 1}"
        fig.suptitle(f"Preprocessing Verification ({os.path.basename(plot_save_path).split('_verification.png')[0] if plot_save_path else 'Unnamed'})", fontsize=16)


        axs[0].set_title(f"Position Comparison: {title_suffix}")
        axs[0].plot(raw_timestamps, raw_positions, 'o', markersize=3, color='gray', alpha=0.5, label='Raw Original Samples')
        axs[0].plot(regular_timestamps, regular_positions, '-', color='cornflowerblue', linewidth=1.5, label=f'Regularized Position ({target_hz}Hz - Input to SavGol)')
        axs[0].plot(regular_timestamps, reconstructed_pos_regular, '--', color='orange', linewidth=2, label='Reconstructed Position (from integrated speed)')
        axs[0].set_ylabel("Position")
        axs[0].legend()
        axs[0].grid(True, linestyle=':')

        axs[1].set_title(f"Calculated Speed: {title_suffix}")
        axs[1].plot(regular_timestamps, speeds_on_regular_time, '-', color='green', linewidth=1.5, label='Calculated Speed (SavGol)')
        axs[1].set_ylabel("Speed")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()
        axs[1].grid(True, linestyle=':')

        fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
        if plot_save_path:
            try:
                os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
                plt.savefig(plot_save_path)
                logging.info(f"Verification plot saved to {plot_save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot to {plot_save_path}: {e}")
        if show_plot_interactive:
            plt.show()
        plt.close(fig) # Close the figure to free memory, important in batch mode

    return regular_timestamps, regular_positions, speeds_on_regular_time


def process_single_file(
    input_file_path: str,
    output_file_path: str,
    target_hz: int,
    sg_window_seconds: float,
    sg_polyorder: int,
    force_overwrite: bool,
    verify_plot_joint_config: Optional[Tuple[str, int]] = None, # (part_name, joint_idx_1_based)
    plot_save_dir: Optional[str] = None,
    show_plot_interactive: bool = True
):
    """Processes a single recording file."""
    logging.info(f"--- Starting processing for: {input_file_path} ---")
    if os.path.exists(output_file_path) and not force_overwrite:
        logging.info(f"Output file {output_file_path} already exists. Skipping (use --force to overwrite).")
        # Still generate plot if requested for an existing file, but don't reprocess
        if verify_plot_joint_config:
             logging.info("Plot verification requested for an existing processed file. "
                          "Plot will be generated from raw data, not the existing processed file for consistency.")
             # To plot based on existing processed, would need to load it, but here we focus on verifying the processing itself.
        # else: # If no plot verification, just return
        # return # Modified: Let's allow plot verification to proceed even if output exists.

    raw_data = load_raw_recording_data(input_file_path)
    if not raw_data:
        return

    # This will hold the final data to be saved in the new JSON structure
    # It will use original key names for positions, and new keys for speeds.
    # The 'time' key will store the regularized timestamps.
    output_json_data = {}
    raw_timestamps_np = np.array(raw_data["time"])
    master_regular_timestamps: Optional[np.ndarray] = None

    # Determine which specific joint to plot for verification
    plot_this_part_name: Optional[str] = None
    plot_this_joint_idx_0_based: Optional[int] = None
    do_any_plot_verification = False
    if verify_plot_joint_config:
        plot_this_part_name, joint_idx_1_based = verify_plot_joint_config
        if plot_this_part_name not in PART_JOINT_COUNTS:
            logging.error(f"Verification part '{plot_this_part_name}' not recognized. No plot will be generated for this file.")
        else:
            plot_this_joint_idx_0_based = joint_idx_1_based - 1
            if not (0 <= plot_this_joint_idx_0_based < PART_JOINT_COUNTS[plot_this_part_name]):
                logging.error(f"Verification joint index {joint_idx_1_based} for part '{plot_this_part_name}' is out of range. No plot for this file.")
                plot_this_part_name = None # Invalidate to prevent error later
            else:
                do_any_plot_verification = True # A valid joint is selected for plotting

    # --- Main Processing Loop for Parts ---
    for part_name, num_joints in PART_JOINT_COUNTS.items():
        if part_name not in raw_data:
            continue
        logging.debug(f"Processing part: {part_name}") # Debug for batch verbosity

        part_data_raw_frames = raw_data[part_name]
        
        temp_regular_positions_per_joint: List[Optional[np.ndarray]] = [None] * num_joints
        temp_speeds_per_joint: List[Optional[np.ndarray]] = [None] * num_joints
        temp_regular_timestamps_per_joint: List[Optional[np.ndarray]] = [None] * num_joints

        # Extract individual raw joint trajectories
        # ... (same extraction logic as v2) ...
        if num_joints == 1:
            if part_data_raw_frames and isinstance(part_data_raw_frames[0], list):
                raw_positions_single = np.array([item[0] for item in part_data_raw_frames if item])
            else:
                raw_positions_single = np.array(part_data_raw_frames)
            all_raw_joint_trajectories = [raw_positions_single]
        else:
            try:
                all_raw_joint_trajectories = np.array(part_data_raw_frames).T
                if all_raw_joint_trajectories.shape[0] != num_joints:
                    logging.warning(f"Part '{part_name}' expected {num_joints} trajectories, got {all_raw_joint_trajectories.shape[0]}. Skipping.")
                    continue
            except Exception as e:
                logging.warning(f"Could not transpose data for {part_name}: {e}. Skipping.")
                continue

        for i in range(num_joints): # i is 0-based joint index
            raw_pos_this_joint = all_raw_joint_trajectories[i]
            
            should_plot_this_specific_joint = (do_any_plot_verification and
                                               plot_this_part_name == part_name and
                                               plot_this_joint_idx_0_based == i)
            
            plot_save_path_this_joint = None
            if should_plot_this_specific_joint and plot_save_dir:
                input_basename = os.path.splitext(os.path.basename(input_file_path))[0]
                plot_filename = f"{input_basename}_{part_name}_joint{i+1}_verification.png"
                plot_save_path_this_joint = os.path.join(plot_save_dir, plot_filename)

            reg_t, reg_p, spd = process_and_verify_single_joint(
                raw_timestamps_np, raw_pos_this_joint,
                part_name, i,
                target_hz, sg_window_seconds, sg_polyorder,
                plot_verification=should_plot_this_specific_joint,
                plot_save_path=plot_save_path_this_joint,
                show_plot_interactive=(show_plot_interactive if should_plot_this_specific_joint else False)
            )
            temp_regular_timestamps_per_joint[i] = reg_t
            temp_regular_positions_per_joint[i] = reg_p
            temp_speeds_per_joint[i] = spd

            if reg_t is not None and master_regular_timestamps is None:
                master_regular_timestamps = reg_t

        # Establish master timeline if not set by first processed joint
        if master_regular_timestamps is None:
            for t_reg_candidate in temp_regular_timestamps_per_joint:
                if t_reg_candidate is not None:
                    master_regular_timestamps = t_reg_candidate
                    break
        
        if master_regular_timestamps is None:
            logging.warning(f"Could not establish a master timeline for part {part_name}. Skipping saving data for this part.")
            continue

        # Align to master timeline and prepare for JSON structure
        # ... (same alignment logic as v2, storing into final_positions_this_part_joints, final_speeds_this_part_joints) ...
        final_positions_this_part_joints: List[List[float]] = []
        final_speeds_this_part_joints: List[List[float]] = []

        for i in range(num_joints):
            # ... (alignment as in V2) ...
            reg_t_joint = temp_regular_timestamps_per_joint[i]
            reg_p_joint = temp_regular_positions_per_joint[i]
            spd_joint = temp_speeds_per_joint[i]

            if reg_t_joint is None or reg_p_joint is None:
                nan_list = [float('nan')] * len(master_regular_timestamps)
                final_positions_this_part_joints.append(nan_list)
                final_speeds_this_part_joints.append(nan_list)
                continue

            current_joint_final_positions = reg_p_joint
            current_joint_final_speeds = spd_joint

            if not np.array_equal(reg_t_joint, master_regular_timestamps):
                logging.debug(f"Re-interpolating {part_name} joint {i+1} to master timeline for JSON output.")
                interp_p = interp1d(reg_t_joint, reg_p_joint, kind='cubic', fill_value="extrapolate", bounds_error=False)
                current_joint_final_positions = interp_p(master_regular_timestamps)
                if spd_joint is not None:
                    interp_s = interp1d(reg_t_joint, spd_joint, kind='cubic', fill_value="extrapolate", bounds_error=False)
                    current_joint_final_speeds = interp_s(master_regular_timestamps)
            
            final_positions_this_part_joints.append(current_joint_final_positions.tolist())
            if current_joint_final_speeds is not None:
                final_speeds_this_part_joints.append(current_joint_final_speeds.tolist())
            else:
                final_speeds_this_part_joints.append([float('nan')] * len(master_regular_timestamps))

        # Store in output_json_data using compatible/new keys
        # Positions go under original part_name key, speeds under {part_name}_speed
        if final_positions_this_part_joints and all(len(p_list) == len(master_regular_timestamps) for p_list in final_positions_this_part_joints):
            if num_joints == 1:
                output_json_data[part_name] = final_positions_this_part_joints[0] # Store as flat list
            else:
                output_json_data[part_name] = np.array(final_positions_this_part_joints).T.tolist() # Transpose back to list of frames
        else:
            logging.warning(f"Inconsistent position data for part {part_name} after alignment. Not saving positions.")

        if final_speeds_this_part_joints and all(len(s_list) == len(master_regular_timestamps) for s_list in final_speeds_this_part_joints):
            speed_key = f"{part_name}_speed"
            if num_joints == 1:
                output_json_data[speed_key] = final_speeds_this_part_joints[0] # Store as flat list
            else:
                output_json_data[speed_key] = np.array(final_speeds_this_part_joints).T.tolist() # Transpose back to list of frames
        else:
            logging.warning(f"Inconsistent speed data for part {part_name} after alignment. Not saving speeds.")


    if master_regular_timestamps is not None and output_json_data: # Check if any part data was actually added
        output_json_data["time"] = master_regular_timestamps.tolist() # Use "time" key for regularized time
        
        # Add metadata about the processing
        output_json_data["processing_metadata"] = {
            "original_file": os.path.basename(input_file_path),
            "target_hz": target_hz,
            "sg_window_seconds": sg_window_seconds,
            "sg_polyorder": sg_polyorder,
            "original_time_key_is_now_regularized": True
        }

        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as f:
                json.dump(output_json_data, f) # Minimal indent for smaller file
            logging.info(f"Processed data saved to {output_file_path}")
        except IOError:
            logging.error(f"Error: Could not write processed data to {output_file_path}")
    elif not output_json_data and master_regular_timestamps is not None:
        logging.warning(f"Master timeline established but no part data was successfully processed for {input_file_path}. Output file not saved.")
    else:
        logging.error(f"Failed to process any data or establish a master timeline for {input_file_path}. Output file not saved.")
    logging.info(f"--- Finished processing for: {input_file_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Reachy recording JSON files: regularize time, calculate joint speeds, and optionally verify with plots.")
    parser.add_argument(
        "--mode", choices=["single", "batch_all", "batch_missing"], default="single",
        help="Processing mode: 'single' for one file, 'batch_all' to process all in input_folder, "
             "'batch_missing' to process only new/missing files in input_folder."
    )
    parser.add_argument(
        "-i", "--input", dest="input_path", type=str, required=True,
        help="Path to the input JSON recording file (for single mode) or input folder (for batch modes)."
    )
    parser.add_argument(
        "-o", "--output", dest="output_path", type=str, default=None,
        help="Path to the output JSON file (for single mode, defaults to '<input>_processed.json') "
             "or output folder (for batch modes, required if input is a folder)."
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
        "--force", action="store_true", help="Overwrite output file(s) if they exist."
    )
    parser.add_argument(
        "--verify_plot_joint", type=str, default=None,
        help="Plot verification for a specific joint. Format: 'part_name:joint_number' (e.g., l_arm:3). "
             "Plots are saved in '<output_folder>/verification_plots/' for batch modes, or alongside output for single mode."
    )
    parser.add_argument(
        "--no_show_plot", action="store_true",
        help="Do not show plots interactively, only save them (if --verify_plot_joint is used)."
    )
    parser.add_argument(
        "--processed_suffix", type=str, default=DEFAULT_PROCESSED_SUFFIX,
        help=f"Suffix to add to processed filenames in batch mode (default: '{DEFAULT_PROCESSED_SUFFIX}')"
    )


    args = parser.parse_args()

    # --- Argument Validation and Path Setup ---
    verify_plot_joint_config_parsed: Optional[Tuple[str, int]] = None
    if args.verify_plot_joint:
        try:
            part_name, joint_idx_str = args.verify_plot_joint.split(':')
            joint_idx_1_based = int(joint_idx_str)
            verify_plot_joint_config_parsed = (part_name, joint_idx_1_based)
            # Further validation of part_name and joint_idx will happen per file
        except ValueError:
            parser.error(f"Invalid format for --verify_plot_joint: '{args.verify_plot_joint}'. Expected 'part_name:joint_number'.")


    if args.mode == "single":
        if not os.path.isfile(args.input_path):
            parser.error(f"Input file not found: {args.input_path}")
        
        output_file_path = args.output_path
        if output_file_path is None:
            base, ext = os.path.splitext(args.input_path)
            output_file_path = f"{base}{args.processed_suffix}{ext}"

        plot_save_base_dir = os.path.dirname(output_file_path) if args.verify_plot_joint else None
        plot_save_full_dir = os.path.join(plot_save_base_dir, DEFAULT_VERIFICATION_PLOT_SUBDIR) if plot_save_base_dir else None


        process_single_file(
            args.input_path, output_file_path,
            args.target_hz, args.sg_window_sec, args.sg_polyorder, args.force,
            verify_plot_joint_config=verify_plot_joint_config_parsed, # Pass for the single file
            plot_save_dir=plot_save_full_dir,
            show_plot_interactive=(not args.no_show_plot if args.verify_plot_joint else False)
        )
    else: # Batch modes
        if not os.path.isdir(args.input_path):
            parser.error(f"Input folder not found: {args.input_path}")
        if args.output_path is None or not os.path.isdir(args.output_path): # Output must be a directory for batch
             # Try to create if doesn't exist, but better to require it
            if args.output_path is None:
                parser.error("Output folder (--output) is required for batch modes.")
            try:
                os.makedirs(args.output_path, exist_ok=True)
                logging.info(f"Created output directory: {args.output_path}")
            except OSError:
                parser.error(f"Output path specified but is not a valid directory or cannot be created: {args.output_path}")

        input_files = glob.glob(os.path.join(args.input_path, "*.json"))
        if not input_files:
            logging.info(f"No .json files found in input folder: {args.input_path}")
            return

        processed_count = 0
        skipped_count = 0

        plot_save_full_dir = os.path.join(args.output_path, DEFAULT_VERIFICATION_PLOT_SUBDIR) if args.verify_plot_joint else None


        for input_file in input_files:
            base_name = os.path.basename(input_file)
            name_part, ext_part = os.path.splitext(base_name)
            output_filename = f"{name_part}{args.processed_suffix}{ext_part}"
            output_file_path = os.path.join(args.output_path, output_filename)

            if args.mode == "batch_missing" and os.path.exists(output_file_path) and not args.force:
                logging.debug(f"Output {output_file_path} exists for {input_file}. Skipping in 'batch_missing' mode.")
                skipped_count +=1
                continue
            
            try:
                process_single_file(
                    input_file, output_file_path,
                    args.target_hz, args.sg_window_sec, args.sg_polyorder, args.force,
                    # For batch mode, verify_plot_joint applies to ALL files if set.
                    # This could generate many plots.
                    verify_plot_joint_config=verify_plot_joint_config_parsed,
                    plot_save_dir=plot_save_full_dir,
                    show_plot_interactive=(not args.no_show_plot if args.verify_plot_joint else False)
                )
                processed_count +=1
            except Exception as e:
                logging.error(f"CRITICAL Error processing {input_file} in batch: {e}", exc_info=True)
                skipped_count +=1
        
        logging.info(f"Batch processing complete. Processed: {processed_count}, Skipped/Failed: {skipped_count}")


if __name__ == "__main__":
    main()