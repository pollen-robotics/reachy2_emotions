import argparse
import datetime
import json
import os
import time

from reachy2_sdk import ReachySDK  # type: ignore


def main(ip: str, filename: str, freq: int):
    reachy = ReachySDK(host=ip)

    data: dict = {
        "time": [],
        "l_arm": [],
        "l_hand": [],
        "r_arm": [],
        "r_hand": [],
        "head": [],
        "l_antenna": [],
        "r_antenna": [],
    }

    input("Press Enter to start the recording.")

    try:
        t0 = time.time()
        print("Recording in progress, press Ctrl+C to stop")

        while True:
            l_arm = reachy.l_arm.get_current_positions()
            r_arm = reachy.r_arm.get_current_positions()
            head = reachy.head.get_current_positions()
            l_hand = reachy.l_arm.gripper.get_current_opening()
            r_hand = reachy.r_arm.gripper.get_current_opening()
            l_antenna = reachy.head.l_antenna.present_position
            r_antenna = reachy.head.r_antenna.present_position

            data["time"].append(time.time() - t0)
            data["l_arm"].append(l_arm)
            data["l_hand"].append(l_hand)
            data["r_arm"].append(r_arm)
            data["r_hand"].append(r_hand)
            data["head"].append(head)
            data["l_antenna"].append(l_antenna)
            data["r_antenna"].append(r_antenna)

            time.sleep(1 / freq)

    except KeyboardInterrupt:
        directory = "recordings"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(
            f"Data saved to {filename}, time of recording :",
            time.time() - t0
        )


if __name__ == "__main__":
    d = datetime.datetime.now()
    default_filename = f'recording_{d.strftime("%m%d_%H%M")}.json'
    parser = argparse.ArgumentParser(
        description="Record the movements of Reachy."
    )

    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="IP address of the robot",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=default_filename,
        help="Name of the file to save the data",
    )

    parser.add_argument(
        "--freq",
        type=int,
        default=100,
        help="Frequency of the recording",
    )

    args = parser.parse_args()

    main(args.ip, args.filename, args.freq)