#!/usr/bin/env python3
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define part names and their expected number of joints (or 1 if single value)
PART_JOINT_COUNTS = {
    "l_arm": 7,
    "r_arm": 7,
    "head": 3,
    "l_hand": 1,
    "r_hand": 1,
    "l_antenna": 1,
    "r_antenna": 1,
}

# Define pretty names for plots if desired
PART_PRETTY_NAMES = {
    "l_arm": "Left Arm",
    "r_arm": "Right Arm",
    "head": "Head",
    "l_hand": "Left Hand (Gripper)",
    "r_hand": "Right Hand (Gripper)",
    "l_antenna": "Left Antenna",
    "r_antenna": "Right Antenna",
}

def load_recording_data(file_path: str) -> Dict[str, Any]:
    """Load the JSON recording."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"Data loaded from {file_path}")

        if "time" not in data or not isinstance(data["time"], list) or len(data["time"]) < 2:
            raise ValueError("Recording must contain a 'time' key with at least two timestamp entries.")

        # Validate other parts (optional but good for robustness)
        for part_name in PART_JOINT_COUNTS.keys():
            if part_name not in data:
                logging.warning(f"Part '{part_name}' not found in the recording.")
                continue
            if not isinstance(data[part_name], list):
                raise ValueError(f"Data for part '{part_name}' must be a list.")
            if len(data[part_name]) != len(data["time"]):
                raise ValueError(
                    f"Mismatch in data length for part '{part_name}' ({len(data[part_name])}) "
                    f"and 'time' ({len(data['time'])})."
                )
        return data
    except FileNotFoundError:
        logging.error(f"Error: Recording file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        raise
    except ValueError as ve:
        logging.error(f"Data validation error: {ve}")
        raise


def prepare_plot_data(
    recording_data: Dict[str, Any], parts_to_plot: List[str]
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
    """
    Extracts and prepares time and joint data for specified parts.
    Returns timestamps and a dictionary of part_name -> list of joint_data arrays.
    """
    timestamps = np.array(recording_data["time"])
    plot_data: Dict[str, List[np.ndarray]] = {}

    for part_name in parts_to_plot:
        if part_name not in recording_data or part_name not in PART_JOINT_COUNTS:
            logging.warning(f"Part '{part_name}' requested for plotting but not found in data or config. Skipping.")
            continue

        part_data_raw = recording_data[part_name]
        num_joints = PART_JOINT_COUNTS[part_name]

        if num_joints == 1:
            # Single value per time step (e.g., hand, antenna)
            # Ensure it's a flat list of numbers, not list of lists
            if part_data_raw and isinstance(part_data_raw[0], list):
                # If it's like [[val1], [val2], ...], flatten it.
                joint_values = np.array([item[0] for item in part_data_raw if item])
            else:
                joint_values = np.array(part_data_raw)
            plot_data[part_name] = [joint_values]
        else:
            # Multiple joints per time step (e.g., arm, head)
            # Data should be list of lists. Transpose to get trajectories per joint.
            # e.g., [[j1_t1, j2_t1], [j1_t2, j2_t2]] -> [[j1_t1, j1_t2], [j2_t1, j2_t2]]
            try:
                # Ensure all sub-lists have the correct number of joints
                for i, frame_joints in enumerate(part_data_raw):
                    if len(frame_joints) != num_joints:
                        raise ValueError(
                            f"Part '{part_name}', frame {i}: expected {num_joints} joints, "
                            f"got {len(frame_joints)}."
                        )
                
                joint_trajectories = np.array(part_data_raw).T
                plot_data[part_name] = [joint_trajectory for joint_trajectory in joint_trajectories]
            except ValueError as e:
                logging.error(f"Error processing multi-joint part '{part_name}': {e}. Ensure data is list of lists with consistent joint counts.")
                continue
            except TypeError as e: # Handles cases where sub-elements aren't lists/numbers
                logging.error(f"Type error processing multi-joint part '{part_name}': {e}. Ensure data is list of lists of numbers.")
                continue


    return timestamps, plot_data


def plot_joint_data(
    timestamps: np.ndarray,
    prepared_data: Dict[str, List[np.ndarray]],
    title: str = "Joint Positions Over Time",
):
    """Plots the prepared joint data."""
    num_parts = len(prepared_data)
    if num_parts == 0:
        logging.warning("No data to plot.")
        return

    # Dynamic subplot layout (e.g., 2 columns, adjust rows as needed)
    # Or one plot per part if many joints, or one subplot per joint if fewer total joints.
    # For now, let's do one subplot per part.

    # Determine total number of individual joint series to plot for color cycling
    total_joint_series = sum(len(trajectories) for trajectories in prepared_data.values())
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, total_joint_series))) # Use viridis or other colormap
    color_idx = 0

    fig, axes = plt.subplots(num_parts, 1, figsize=(12, 3 * num_parts), sharex=True, squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    for i, (part_name, joint_trajectories_list) in enumerate(prepared_data.items()):
        ax = axes[i]
        part_pretty_name = PART_PRETTY_NAMES.get(part_name, part_name.replace("_", " ").title())
        ax.set_title(part_pretty_name)
        ax.set_ylabel("Joint Position (degrees/units)") # Adjust unit if known
        ax.grid(True, linestyle=":", alpha=0.7)

        num_joints_in_part = len(joint_trajectories_list)

        for j, joint_trajectory in enumerate(joint_trajectories_list):
            if timestamps.shape[0] != joint_trajectory.shape[0]:
                logging.warning(
                    f"Skipping plot for {part_name} joint {j}: "
                    f"Time dimension mismatch ({timestamps.shape[0]}) vs "
                    f"Joint data dimension ({joint_trajectory.shape[0]})"
                )
                continue
            
            label = f"Joint {j+1}" if num_joints_in_part > 1 else part_pretty_name
            ax.plot(timestamps, joint_trajectory, label=label, color=colors[color_idx % len(colors)], linewidth=1.5)
            color_idx += 1
        
        if num_joints_in_part > 1 : # or always show legend
             ax.legend(loc="upper right", fontsize="small")
        
        # Improve x-axis ticks for time
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
        ax.tick_params(axis='x', rotation=30)


    if num_parts > 0:
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(title, fontsize=16, y=0.99) # Adjust y to avoid overlap with top subplot title
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot joint positions from a Reachy recording JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON recording file.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=list(PART_JOINT_COUNTS.keys()) + ["all"],
        default=["all"],
        help=(
            "Space-separated list of parts to plot (e.g., l_arm head l_hand). "
            f"Choices: {', '.join(list(PART_JOINT_COUNTS.keys()) + ['all'])}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Joint Positions Over Time",
        help="Title for the plot."
    )

    args = parser.parse_args()

    try:
        recording_data = load_recording_data(args.json_file)
    except Exception:
        # Error already logged by load_recording_data
        return

    parts_to_plot_requested = args.parts
    if "all" in parts_to_plot_requested:
        parts_to_plot = [part for part in PART_JOINT_COUNTS.keys() if part in recording_data]
    else:
        parts_to_plot = [part for part in parts_to_plot_requested if part in recording_data]

    if not parts_to_plot:
        logging.error("No valid parts selected or found in the data for plotting.")
        return

    logging.info(f"Preparing data for parts: {', '.join(parts_to_plot)}")
    timestamps, prepared_plot_data = prepare_plot_data(recording_data, parts_to_plot)

    if not prepared_plot_data:
        logging.error("Failed to prepare any data for plotting.")
        return
        
    plot_joint_data(timestamps, prepared_plot_data, title=args.title)


if __name__ == "__main__":
    plt.style.use('seaborn-darkgrid') 


    main()