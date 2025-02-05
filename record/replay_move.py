import argparse
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
from reachy2_sdk import ReachySDK  # type: ignore


def get_last_recording(folder: str) -> str:
    """Retrieve the most recent recording file from a specified folder.

    Args:
        folder (str): Path to the folder containing recording files.

    Returns:
        str: The name of the most recently created recording file.
    """
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    files.sort(key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return files[-1]


def load_data(folder: str, filename: str) -> Tuple[dict, float]:
    """Load recording data and extract the time interval between frames.

    Args:
        folder (str): Path to the folder containing the recording.
        filename (str): Name of the file to load.

    Returns:
        Tuple[dict, float]: A tuple containing the loaded data dictionary and
            the time interval between frames.
    """
    file_path = os.path.join(folder, filename)
    with open(file_path, "r") as f:
        data = json.load(f)
        print(f"Data loaded from {filename}")
    timeframe = data["time"][1] - data["time"][0]
    return data, timeframe


def distance_with_new_pose(reachy: ReachySDK, data: dict) -> float:
    """Calculate the maximum distance between the current arm positions
        and the first positions from the data.

    Args:
        reachy (ReachySDK): The Reachy robot object.
        data (dict): The recording data containing arm positions.

    Returns:
        float: The maximum distance between the current and initial positions
            of the arms.
    """
    first_l_arm_pose = reachy.l_arm.forward_kinematics(data["l_arm"][0])
    first_r_arm_pose = reachy.r_arm.forward_kinematics(data["r_arm"][0])

    current_l_arm_pose = reachy.l_arm.forward_kinematics()
    current_r_arm_pose = reachy.r_arm.forward_kinematics()

    distance_l_arm = np.linalg.norm(
        first_l_arm_pose[:3, 3] - current_l_arm_pose[:3, 3]
    )
    distance_r_arm = np.linalg.norm(
        first_r_arm_pose[:3, 3] - current_r_arm_pose[:3, 3]
    )

    return np.max([distance_l_arm, distance_r_arm])


def main(ip: str, filename: Optional[str]):
    """Main function to connect to Reachy, load a recorded movement,
        and replay it on the robot.

    Args:
        ip (str): IP address of the Reachy robot.
        filename (Optional[str]): Name of the recording file to replay.
            If not provided, the most recent recording will be used.
    """

    # connect to Reachy
    reachy = ReachySDK(host=ip)

    # get the last recording file if no filename is given
    folder = "recordings"
    filename = get_last_recording(folder) if filename is None\
        else filename + ".json"

    # load the data and the timeframe from the file
    data, timeframe = load_data(folder, filename)

    # check the distance between the current position of arms
    # and the first position and adapt the duration of the first move
    max_dist = distance_with_new_pose(reachy, data)
    first_duration = max_dist * 10 if max_dist > 0.2 else 2

    # wait for the user to press enter
    input("Is Reachy ready to move ? Press Enter to continue.")

    # set Reachy on the first position with a goto
    reachy.l_arm.goto(data["l_arm"][0], duration=first_duration)
    reachy.r_arm.goto(data["r_arm"][0], duration=first_duration)
    reachy.l_arm.gripper.set_opening(data["l_hand"][0])
    reachy.r_arm.gripper.set_opening(data["r_hand"][0])
    reachy.head.goto(data["head"][0], duration=first_duration, wait=True)

    print("First position reached.")

    t0 = time.time()

    # replay the data
    try:
        for ite in range(len(data["time"])):
            start_t = time.time() - t0

            for joint, goal in zip(
                reachy.l_arm.joints.values(), data["l_arm"][ite]
            ):
                joint.goal_position = goal
            for joint, goal in zip(
                reachy.r_arm.joints.values(), data["r_arm"][ite]
            ):
                joint.goal_position = goal
            for joint, goal in zip(
                reachy.head.joints.values(), data["head"][ite]
            ):
                joint.goal_position = goal

            reachy.l_arm.gripper.goal_position = data["l_hand"][ite]
            reachy.r_arm.gripper.goal_position = data["r_hand"][ite]

            reachy.send_goal_positions(check_positions=False)

            left_time = timeframe - (time.time() - t0 - start_t)
            if left_time > 0:
                time.sleep(left_time)
        else:
            print(
                "End of the recording. Time of the replaying : ",
                time.time() - t0
            )

    except KeyboardInterrupt:
        print("Replay stopped by the user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="IP address of the robot",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional name of the file to replay",
    )

    args = parser.parse_args()

    main(args.ip, args.filename)