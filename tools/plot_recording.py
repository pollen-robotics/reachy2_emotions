#!/usr/bin/env python3
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configure logging
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


def parse_part_selection(selection_str: str, available_parts: List[str]) -> Tuple[str, Optional[int]]:
    """
    Parses a part selection string like 'l_arm' or 'l_arm:3'.
    Returns (part_name, joint_index) where joint_index is 0-based or None.
    """
    if ":" in selection_str:
        try:
            part_name, joint_idx_str = selection_str.split(":")
            joint_idx = int(joint_idx_str) - 1  # User provides 1-based index
            if joint_idx < 0:
                raise ValueError("Joint index must be 1 or greater.")
        except ValueError:
            logging.warning(
                f"Invalid joint index in '{selection_str}'. "
                "Format should be 'part_name:joint_number' (e.g., l_arm:3). "
                "Plotting all joints for this part if valid."
            )
            part_name = selection_str.split(":")[0]  # Take part before colon
            joint_idx = None
    else:
        part_name = selection_str
        joint_idx = None

    if part_name not in available_parts:
        logging.warning(f"Part '{part_name}' from selection '{selection_str}' not available in data. Skipping.")
        return None, None  # Indicate invalid part

    # Validate joint_idx against actual number of joints for the part
    if joint_idx is not None and part_name in PART_JOINT_COUNTS:
        if joint_idx >= PART_JOINT_COUNTS[part_name]:
            logging.warning(
                f"Joint index {joint_idx + 1} for part '{part_name}' is out of range "
                f"(max {PART_JOINT_COUNTS[part_name]}). Plotting all joints for this part."
            )
            joint_idx = None  # Fallback to plotting all joints for this part
    return part_name, joint_idx


def prepare_plot_data(
    recording_data: Dict[str, Any], parts_to_plot_config: List[Tuple[str, Optional[int]]]
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, List[int]]]:
    """
    Extracts and prepares time and joint data for specified parts and specific joints.
    Returns:
        - timestamps (np.ndarray)
        - plot_data (Dict[str_part_name, List[np.ndarray_joint_trajectory]])
        - joint_indices_to_plot (Dict[str_part_name, List[int_0_based_indices]])
    """
    timestamps = np.array(recording_data["time"])
    plot_data: Dict[str, List[np.ndarray]] = {}
    joint_indices_to_plot: Dict[str, List[int]] = {}

    for part_name, specific_joint_idx_to_plot in parts_to_plot_config:
        if part_name not in recording_data or part_name not in PART_JOINT_COUNTS:
            # Already logged by parse_part_selection if it came from there
            continue

        part_data_raw = recording_data[part_name]
        num_total_joints_in_part = PART_JOINT_COUNTS[part_name]

        current_part_trajectories: List[np.ndarray] = []
        current_part_joint_indices: List[int] = []

        if num_total_joints_in_part == 1:
            # Single value per time step (e.g., hand, antenna)
            if part_data_raw and isinstance(part_data_raw[0], list):
                joint_values = np.array([item[0] for item in part_data_raw if item])
            else:
                joint_values = np.array(part_data_raw)

            if specific_joint_idx_to_plot is None or specific_joint_idx_to_plot == 0:
                current_part_trajectories.append(joint_values)
                current_part_joint_indices.append(0)
            else:
                logging.warning(
                    f"Part '{part_name}' is single-valued; specific joint index {specific_joint_idx_to_plot + 1} ignored or invalid."
                )
                # Still plot if no specific index was asked (None) or if 0 was asked.
                if (
                    specific_joint_idx_to_plot is None
                ):  # only add if no specific joint was requested (it means plot all for the part)
                    current_part_trajectories.append(joint_values)
                    current_part_joint_indices.append(0)

        else:  # Multiple joints
            try:
                for i, frame_joints in enumerate(part_data_raw):
                    if len(frame_joints) != num_total_joints_in_part:
                        raise ValueError(
                            f"Part '{part_name}', frame {i}: expected {num_total_joints_in_part} joints, "
                            f"got {len(frame_joints)}."
                        )
                all_joint_trajectories_for_part = np.array(part_data_raw).T

                if specific_joint_idx_to_plot is not None:
                    if 0 <= specific_joint_idx_to_plot < num_total_joints_in_part:
                        current_part_trajectories.append(all_joint_trajectories_for_part[specific_joint_idx_to_plot])
                        current_part_joint_indices.append(specific_joint_idx_to_plot)
                    else:
                        # This case should be caught by parse_part_selection, but good to be safe
                        logging.warning(
                            f"Invalid specific joint index {specific_joint_idx_to_plot} for part '{part_name}'. Skipping."
                        )
                else:  # Plot all joints for this part
                    for idx in range(num_total_joints_in_part):
                        current_part_trajectories.append(all_joint_trajectories_for_part[idx])
                        current_part_joint_indices.append(idx)

            except ValueError as e:
                logging.error(f"Error processing multi-joint part '{part_name}': {e}.")
                continue
            except TypeError as e:
                logging.error(f"Type error processing multi-joint part '{part_name}': {e}.")
                continue

        if current_part_trajectories:
            if part_name not in plot_data:
                plot_data[part_name] = []
                joint_indices_to_plot[part_name] = []
            plot_data[part_name].extend(current_part_trajectories)
            joint_indices_to_plot[part_name].extend(current_part_joint_indices)

    return timestamps, plot_data, joint_indices_to_plot


def plot_joint_data(
    timestamps: np.ndarray,
    prepared_data: Dict[str, List[np.ndarray]],
    joint_indices_plotted: Dict[str, List[int]],  # 0-based indices of joints plotted for each part
    main_title: str,
    show_points: bool = False,
):
    """Plots the prepared joint data."""
    num_parts_with_data = len(prepared_data)
    if num_parts_with_data == 0:
        logging.warning("No data to plot.")
        return

    # Use a perceptually distinct colormap like 'tab10', 'tab20', or 'Paired'
    # 'tab10' has 10 distinct colors, 'tab20' has 20.
    # We'll cycle through 'tab10' for better distinction if fewer than 10 lines per plot.
    # If more, we might need a different strategy or accept some color reuse.

    fig, axes = plt.subplots(num_parts_with_data, 1, figsize=(14, 3.5 * num_parts_with_data), sharex=True, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0  # Index for iterating through axes
    for part_name, joint_trajectories_list in prepared_data.items():
        if not joint_trajectories_list:  # Should not happen if prepare_plot_data is correct
            continue

        ax = axes[plot_idx]
        part_pretty_name = PART_PRETTY_NAMES.get(part_name, part_name.replace("_", " ").title())
        ax.set_title(part_pretty_name, fontsize=12)
        ax.set_ylabel("Joint Position", fontsize=10)  # Removed (degrees/units) for now
        ax.grid(True, linestyle="--", alpha=0.6)

        # Use 'tab10' for distinct colors, cycle if more than 10 joints in one subplot
        prop_cycle = plt.cycler(color=plt.cm.get_cmap("tab10").colors)
        ax.set_prop_cycle(prop_cycle)

        actual_joint_indices_for_this_part = joint_indices_plotted.get(part_name, [])

        for i, joint_trajectory in enumerate(joint_trajectories_list):
            if timestamps.shape[0] != joint_trajectory.shape[0]:
                logging.warning(
                    f"Skipping plot for {part_name} (data index {i}): "
                    f"Time dimension mismatch ({timestamps.shape[0]}) vs "
                    f"Joint data dimension ({joint_trajectory.shape[0]})"
                )
                continue

            # Determine the true joint number (1-based for label)
            # This relies on joint_indices_plotted being correctly populated by prepare_plot_data
            true_joint_num_1_based = (
                (actual_joint_indices_for_this_part[i] + 1) if i < len(actual_joint_indices_for_this_part) else (i + 1)
            )

            label = f"Joint {true_joint_num_1_based}" if PART_JOINT_COUNTS[part_name] > 1 else part_pretty_name

            (line,) = ax.plot(timestamps, joint_trajectory, label=label, linewidth=1.5, alpha=0.8)
            if show_points:
                ax.plot(
                    timestamps, joint_trajectory, marker="o", linestyle="None", markersize=3, color=line.get_color(), alpha=0.6
                )

        if any(PART_JOINT_COUNTS[p] > 1 for p in prepared_data.keys() if p == part_name) or len(joint_trajectories_list) > 1:
            ax.legend(loc="best", fontsize="small")  # 'best' can sometimes be slow or overlap

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, prune="both"))
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        plot_idx += 1

    if num_parts_with_data > 0:
        axes[-1].set_xlabel("Time (seconds)", fontsize=10)
        fig.suptitle(main_title, fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust rect for suptitle and xlabel

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot joint positions from a Reachy recording JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON recording file.")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["all"],
        help=(
            "Space-separated list of parts to plot (e.g., l_arm head:1 l_hand r_arm:3). "
            "To plot a specific joint, use 'part_name:joint_number' (1-based index). "
            "Example: 'l_arm:3' for the 3rd joint of the left arm. "
            "Use 'all' to plot all joints of all available parts. Default: all."
        ),
    )

    args = parser.parse_args()

    file_basename = os.path.splitext(os.path.basename(args.json_file))[0]
    plot_main_title = f"Joint Positions: {file_basename}"

    try:
        recording_data = load_recording_data(args.json_file)
    except Exception:
        return

    parts_config_to_plot: List[Tuple[str, Optional[int]]] = []
    available_parts_in_data = [p for p in PART_JOINT_COUNTS.keys() if p in recording_data]

    if "all" in args.parts:
        for part_name in available_parts_in_data:
            parts_config_to_plot.append((part_name, None))  # None means all joints for this part
    else:
        for selection_str in args.parts:
            part_name, joint_idx = parse_part_selection(selection_str, available_parts_in_data)
            if part_name:  # If valid part was parsed
                # Avoid duplicates if user specifies 'l_arm' and 'l_arm:3'
                # Current logic will plot l_arm (all joints) then l_arm (joint 3)
                # For simplicity, we'll allow this for now. User should be specific.
                # Or, we could process "part_name" first, then specific joints if "part_name:joint_idx" is also present.
                # For now, if 'l_arm' is present, it implies all joints.
                # If 'l_arm:3' is also present, the current `prepare_plot_data` might plot joint 3 twice
                # if it's not careful with how it accumulates. Let's refine prepare_plot_data.
                parts_config_to_plot.append((part_name, joint_idx))

    if not parts_config_to_plot:
        logging.error("No valid parts selected or found in the data for plotting.")
        return

    logging.info(f"Plotting configuration: {parts_config_to_plot}")
    timestamps, prepared_plot_data, joint_indices_plotted = prepare_plot_data(recording_data, parts_config_to_plot)

    if not prepared_plot_data:
        logging.error("Failed to prepare any data for plotting.")
        return

    plot_joint_data(timestamps, prepared_plot_data, joint_indices_plotted, plot_main_title, True)


if __name__ == "__main__":
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

    # plt.rcParams['lines.markersize'] = 4 # Example for global marker size if needed
    # plt.rcParams['lines.linewidth'] = 1.8 # Example for global line width
    main()
