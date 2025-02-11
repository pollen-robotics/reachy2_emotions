
import pygame
import time
import math
import numpy as np
import traceback
import threading
import queue
from reachy2_sdk import ReachySDK
import numpy.typing as npt
from reachy2_sdk.parts.joints_based_part import JointsBasedPart


import os

# To be able to use pygame in "headless" mode
# set SDL to use the dummy NULL video driver, so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"

msg = """
This node takes inputs from a controller and publishes them
as Twist messages in SI units. Tested on a SONY Dual shock 4 controller
and an XBOX controller.

Left joy: holonomic translations
Right joy: rotation

L2/L1 : increase/decrease only linear speed (additive) +-0.05m/s
R2/R1 : increase/decrease only angular speed (additive) +-0.2rad/s

CTRL-C  or press CIRCLE on the controller to quit
"""

# PS4 controller:
# Button  0 = X
# Button  1 = O
# Button  2 = Triangle
# Button  3 = Square
# Button  4 = l1
# Button  5 = r1
# Button  6 = l2
# Button  7 = r2
# Button  8 = share
# Button  9 = options
# Button 10 = ps_button
# Button 11 = joy_left
# Button 12 = joy_right

# XBOX controller:
# Button  0 = A
# Button  1 = B
# Button  2 = X
# Button  3 = Y
# Button  4 = LB
# Button  5 = RB
# Button  6 = back
# Button  7 = start
# Button  8 = big central button
# LT and RT are axis (like a joy)

# When using the XBOX controller, most of it is the same,
# except that you must use Start and Back to increase the max speeds.


# These are integer values between 0 and 100
TORQUE_LIMIT=100
SPEED_LIMIT=100
MAX_ANTENNA_ANGLE = 130 # to be checked
## Bunch of utility functions

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1



def hello(reachy: ReachySDK) -> None:
    reachy.r_arm.goto([0, -40, -70, -120, 0, 0, 90], duration=1.5)
    reachy.head.goto([10, 0, 0], duration=1.5)
    time.sleep(1.5)
    x = threading.Thread(target=reachy.head.happy)
    x.start()
    reachy.r_arm.goto([0, -40, -70, -115, 0, 20, 90], duration=0.5, interpolation_mode="linear")
    reachy.r_arm.goto([0, -40, -70, -120, 0, -20, 90], duration=0.5, interpolation_mode="linear")
    reachy.r_arm.goto([0, -40, -70, -115, 0, 20, 90], duration=0.5, interpolation_mode="linear", wait=True)

def sweeting(reachy: ReachySDK) -> None:
    reachy.r_arm.gripper.close()
    reachy.head.goto([10, 15, 0])
    x = threading.Thread(target=reachy.head.sad)
    x.start()
    reachy.r_arm.goto([-75, -8, 20, -120, 42, -3, 65])
    reachy.r_arm.translate_by(0, -0.05, 0, duration=1)
    reachy.r_arm.translate_by(0, 0.05, 0, duration=1, wait=True)
    reachy.head.goto_posture()
    reachy.r_arm.goto([0, 10, -10, 0, 0, 0, 0])
    time.sleep(0.5)
    reachy.r_arm.gripper.open()

def discover(reachy: ReachySDK) -> None:
    reachy.head.look_at(0.5, -0.1, -0.2)
    reachy.r_arm.goto([5, 7, 9, -95, 0, 0, 0])

    upward_gotoid = reachy.r_arm.translate_by(x=0.05, y=0, z=0, frame="gripper")
    forward_gotoid = reachy.r_arm.translate_by(x=0, y=0, z=-0.05, frame="gripper", duration=1)

    while not reachy.is_goto_finished(upward_gotoid):
        time.sleep(0.1)
    reachy.head.look_at(0.55, -0.1, -0.2, duration=1)
    x = threading.Thread(target=reachy.head.surprised)
    x.start()

    back_gotoid = reachy.r_arm.translate_by(x=0, y=0, z=0.05, frame="gripper", duration=1)
    reachy.head.look_at(0.4, -0.1, -0.2, duration=1)

    arm_gotoid = reachy.r_arm.goto([-9, 3, -8, -125, 3, -16, 38])
    while not reachy.r_arm.get_goto_playing() == arm_gotoid:
        time.sleep(0.1)
    reachy.head.look_at(0.5, -0.25, -0.1)
    reachy.head.goto([-7, 15, -27], duration=1)
    first_rotation_gotoid = reachy.r_arm.wrist.yaw.goto(80, duration=1)
    second_rotation_gotoid = reachy.r_arm.wrist.yaw.goto(-20, duration=1)
    reachy.r_arm.wrist.yaw.goto(80, duration=1)
    reachy.r_arm.wrist.yaw.goto(-20, duration=1)
    lastarmrot_gotoid = reachy.r_arm.wrist.yaw.goto(80, duration=1)

    while not reachy.is_goto_finished(first_rotation_gotoid):
        time.sleep(0.1)
    x = threading.Thread(target=reachy.head.surprised2)
    x.start()

    while not reachy.is_goto_finished(second_rotation_gotoid):
        time.sleep(0.1)

    reachy.head.goto([15, 15, -27])
    reachy.r_arm.gripper.close()
    time.sleep(1)
    reachy.r_arm.gripper.open()
    time.sleep(0.5)
    reachy.r_arm.gripper.close()
    time.sleep(0.5)
    reachy.r_arm.gripper.open()

    while reachy.r_arm.gripper.is_moving():
        time.sleep(0.1)

    ok_gotoid = reachy.r_arm.goto([-20, 7, -8, -130, 3, -16, 0], duration=1)
    while not reachy.is_goto_finished(lastarmrot_gotoid):
        time.sleep(0.1)
    reachy.head.goto([0, 0, 0], duration=1)
    reachy.head.goto([0, 10, 0], duration=0.3)
    reachy.head.goto([0, 0, 0], duration=0.3)
    x = threading.Thread(target=reachy.head.happy)
    x.start()
    while not reachy.is_goto_finished(ok_gotoid):
        time.sleep(0.1)
    reachy.r_arm.gripper.close()
    reachy.r_arm.goto([-15, 7, -8, -125, 3, -27, 0], duration=0.3)
    reachy.r_arm.goto([-20, 7, -8, -130, 3, -16, 0], duration=0.3, wait=True)

def check_grippers(reachy: ReachySDK) -> None:
    reachy.head.look_at(0.5, -0.1, -0.2, duration=1.5)
    r_arm_gotoid = reachy.r_arm.goto([5, 7, 9, -115, 0, 0, 0], duration=1.5)
    while not reachy.is_goto_finished(r_arm_gotoid):
        time.sleep(0.1)

    reachy.head.goto([-15, 26, 11], duration=2)
    l_arm_gotoid = reachy.l_arm.goto([5, -7, -9, -115, 0, 0, 0], duration=1.5)

    x = threading.Thread(target=reachy.head.surprised2)
    x.start()

    while not reachy.is_goto_finished(l_arm_gotoid):
        time.sleep(0.1)

    reachy.r_arm.gripper.close()
    reachy.l_arm.gripper.close()
    time.sleep(0.2)
    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()
    time.sleep(0.2)
    reachy.r_arm.gripper.close()
    reachy.l_arm.gripper.close()
    time.sleep(0.2)
    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()

def happy(reachy: ReachySDK) -> None:
    reachy.head.goto([20, 0, 0], duration=0.7)
    time.sleep(0.2)
    x = threading.Thread(target=reachy.head.happy)
    x.start()
    time.sleep(1)
    reachy.head.goto([-10, -7, 0], duration=1)
    reachy.head.goto([0, 0, 0], wait=True)
    x.join()

def sad(reachy: ReachySDK) -> None:
    reachy.head.goto([0, 25, 0])
    time.sleep(0.5)
    x = threading.Thread(target=reachy.head.sad)
    x.start()
    reachy.head.goto([-4, 25, 5], duration=1)
    reachy.head.goto([3, 25, -5], duration=1)
    reachy.head.goto([0, 25, 5], duration=1)
    time.sleep(4.5)
    reachy.head.goto([-5, 10, 0])
    time.sleep(3)
    reachy.head.goto([0, 0, 0], wait=True)
    x.join()

def curious(reachy: ReachySDK) -> None:
    reachy.head.goto([10, 0, 0], duration=0.7)
    time.sleep(0.2)
    x = threading.Thread(target=reachy.head.surprised)
    x.start()
    reachy.head.goto([17, 0, 0], duration=1)
    time.sleep(1)
    reachy.head.goto([0, 0, 0], duration=1, wait=True)
    x.join()

def curious2(reachy: ReachySDK) -> None:
    reachy.head.goto([-16, -2, -3], duration=1)
    time.sleep(0.2)
    x = threading.Thread(target=reachy.head.surprised2)
    x.start()
    reachy.head.goto([12, 0, 0], duration=1)
    time.sleep(1)
    reachy.head.goto([0, 0, 0], duration=1, wait=True)
    x.join()

def play_happy(reachy):
    print("Play happy")
    happy(reachy)

def play_sad(reachy):
    print("Play sad")
    sad(reachy)

def play_curious(reachy):
    print("Play curious")
    curious(reachy)

def play_curious2(reachy):
    print("Play curious 2")
    curious2(reachy)

def play_hello(reachy):
    print("Play Hello ...")
    hello(reachy)
    reachy.goto_posture("default", wait=True)

def play_sweeting(reachy):
    print("Play sweeting ...")
    sweeting(reachy)
    reachy.goto_posture("default", wait=True)

def play_discover(reachy):
    print("Play discover ...")
    discover(reachy)
    reachy.goto_posture("default", wait=True)

def play_check_grippers(reachy):
    print("Play check grippers ...")
    check_grippers(reachy)
    reachy.goto_posture("default", wait=True)


class JoyTeleop():
    def __init__(self):
        print("Starting zuuu_teleop_joy!")

        pygame.init()
        pygame.display.init()
        pygame.joystick.init()

        self.nb_joy = pygame.joystick.get_count()
        if self.nb_joy < 1:
            self.get_logger().error("No controller detected.")
            self.emergency_shutdown()
        print("nb joysticks: {}".format(self.nb_joy))
        self.j = pygame.joystick.Joystick(0)
        self.lin_speed_ratio = 0.15
        self.rot_speed_ratio = 1.5
        # The joyticks dont come back at a perfect 0 position when released.
        # Any abs(value) below min_joy_position will be assumed to be 0
        self.min_joy_position = 0.03
        self.current_command = None
        self.command_lock = threading.Lock()
        self.antenna_vibration_offset = 0.0
        
        self.emergency_reachy = ReachySDK(host="localhost")

        if not self.emergency_reachy.is_connected:
            exit("Reachy is not connected.")
        
        # self.emergency_reachy.mobile_base._set_drive_mode("cmd_vel")
        # self.emergency_reachy.mobile_base.reset_odometry()
        self.emergency_reachy.turn_on()
        self.emergency_reachy.head.r_antenna.turn_on()
        self.emergency_reachy.head.l_antenna.turn_on()
        print("ReachySDK initialized and turned on.")
        # set_speed_and_torque_limits(self.emergency_reachy, torque_limit=TORQUE_LIMIT, speed_limit=SPEED_LIMIT)
        
        self.prev_joy2 = -1
        self.prev_joy5 = -1
        self.left_joy = 0.0
        self.right_joy = 0.0
        self.prev_hat = (0, 0)
                
        self.sdk_command_queue = queue.Queue()

        # Start ReachySDK in a separate thread
        self.sdk_thread = threading.Thread(target=self.run_reachy_sdk, daemon=True)
        self.sdk_thread.start()
        
        print(msg)
        
        
    def set_command(self, command):
        """Set a new command if none is currently being executed."""
        with self.command_lock:
            if self.current_command is None:
                self.current_command = command
            else:
                self.get_logger().warn("Cannot set command; one is already being executed.")

    def clear_command(self):
        """Clear the current command, e.g., during an emergency stop."""
        with self.command_lock:
            self.current_command = None
            
    def _vibration_loop(self):
        dur = 2
        t = np.linspace(0, dur, dur * 100)
        pos = 10 * np.sin(2 * np.pi * 5 * t)
        
        for p in pos:
            self.antenna_vibration_offset = p
            time.sleep(0.01)


    def start_vibration(self):
        """Starts the vibration thread if not already running."""
        self._vibration_active = True
        self._vibration_thread = threading.Thread(target=self._vibration_loop)
        self._vibration_thread.daemon = True
        self._vibration_thread.start()


    def run_reachy_sdk(self):
        """Thread for managing ReachySDK client."""
        try:
            reachy = ReachySDK(host="localhost")

            if not reachy.is_connected:
                exit("Reachy is not connected.")


            while True:
                time.sleep(0.01)  # Avoid busy waiting
                # Gachettes
                # reachy.head.l_antenna.goal_position = (self.prev_joy2 + 1)*MAX_ANTENNA_ANGLE
                # reachy.head.r_antenna.goal_position = (self.prev_joy5 + 1)*MAX_ANTENNA_ANGLE
                # joy
                new_l = -(self.left_joy)*MAX_ANTENNA_ANGLE - self.antenna_vibration_offset
                new_r = (self.right_joy)*MAX_ANTENNA_ANGLE + self.antenna_vibration_offset
                if abs(new_l - reachy.head.l_antenna.goal_position) > 0.1 or abs(new_r - reachy.head.r_antenna.goal_position) > 0.1:  
                    reachy.head.l_antenna.goal_position = new_l
                    reachy.head.r_antenna.goal_position = new_r
                    reachy.send_goal_positions(check_positions=False)
                # print(f"antenna L, R: {reachy.head.l_antenna.goal_position}, {reachy.head.r_antenna.goal_position}")
                
                with self.command_lock:
                    command = self.current_command
                    self.current_command = None  # Clear the command after reading

                if command is None:
                    continue
                


                # Execute the command
                try:
                    if command == "play_hello":
                        play_hello(reachy)
                    elif command == "play_sweeting":
                        play_sweeting(reachy)
                    elif command == "play_discover":
                        play_discover(reachy)
                    elif command == "play_check_grippers":
                        play_check_grippers(reachy)
                    elif command == "play_happy":
                        play_happy(reachy)
                    elif command == "play_curious":
                        play_curious(reachy)
                    elif command == "play_curious2":
                        play_curious2(reachy)
                    elif command == "play_sad":
                        play_sad(reachy)
                    else:
                        self.get_logger().error(f"Unknown command: {command}")
                except Exception as e:
                    self.get_logger().error(f"Error executing ReachySDK command: {e}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize ReachySDK: {e}")

    def tick_controller(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if self.j.get_button(6):  # l2
                    self.lin_speed_ratio = min(3.0, self.lin_speed_ratio + 0.05)
                    print(
                        "max translational speed: {:.1f}m/s, max rotational speed: {:.1f}rad/s".format(
                            self.lin_speed_ratio * 100, self.rot_speed_ratio * 100
                        )
                    )
                if self.j.get_button(7):  # r2
                    self.rot_speed_ratio = min(12.0, self.rot_speed_ratio + 0.2)
                    print(
                        "max translational speed: {:.1f}m/s, max rotational speed: {:.1f}rad/s".format(
                            self.lin_speed_ratio * 100, self.rot_speed_ratio * 100
                        )
                    )
                if self.j.get_button(4):  # l1
                    self.lin_speed_ratio = max(0.0, self.lin_speed_ratio - 0.05)
                    print(
                        "max translational speed: {:.1f}m/s, max rotational speed: {:.1f}rad/s".format(
                            self.lin_speed_ratio * 100, self.rot_speed_ratio * 100
                        )
                    )
                if self.j.get_button(5):  # r1
                    self.rot_speed_ratio = max(0.0, self.rot_speed_ratio - 0.2)
                    print(
                        "max translational speed: {:.1f}m/s, max rotational speed: {:.1f}rad/s".format(
                            self.lin_speed_ratio * 100, self.rot_speed_ratio * 100
                        )
                    )
                if self.j.get_button(3):  # Y
                    self.set_command("play_hello")
                if self.j.get_button(2):  # X
                    # self.set_command("play_check_grippers")
                    self.start_vibration()
                if self.j.get_button(0):  # A
                    self.set_command("play_discover")
                if self.j.get_button(1):
                    self.set_command("play_sweeting")
                
            elif event.type == pygame.JOYAXISMOTION:
                curr_joy2 = self.j.get_axis(2) # left (LT). -1 when not pressed, 1 when pressed
                curr_joy5 = self.j.get_axis(5) # right (RT). -1 when not pressed, 1 when pressed
                self.left_joy = self.j.get_axis(1)
                self.right_joy = self.j.get_axis(4)
                # print(f"LT: {curr_joy2}, RT: {curr_joy5}, left: {left_joy}, right: {right_joy}")
                # print(f"left: {self.left_joy}, right: {self.right_joy}")
                # if curr_joy2 > 0 and self.prev_joy2 <= 0:
                #     self.emergency_shutdown()
                # if curr_joy5 > 0 and self.prev_joy5 <=0:
                #     pass
                self.prev_joy2 = curr_joy2
                self.prev_joy5 = curr_joy5
            
            elif event.type == pygame.JOYHATMOTION:
                curr_hat = self.j.get_hat(0)
                if curr_hat == (0, 1) and self.prev_hat!=(0, 1):
                    self.set_command("play_happy")
                if curr_hat == (0, -1) and self.prev_hat!=(0, -1):
                    self.set_command("play_sad")
                if curr_hat == (1, 0) and self.prev_hat!=(1, 0):
                    self.set_command("play_curious")
                if curr_hat == (-1, 0) and self.prev_hat!=(-1, 0):
                    self.set_command("play_curious2")
                self.prev_hat = curr_hat

        if self.nb_joy != pygame.joystick.get_count():
            self.get_logger().warn("Controller disconnected!")

    def rumble(self, duration):
        self.rumble_start = time.time()
        self.is_rumble = True
        self.rumble_duration = duration
        # Duration doesn't work, have to do it ourselves
        self.j.rumble(1, 1, 1000)

    def print_controller(self):
        # Get the name from the OS for the controller/joystick.
        name = self.j.get_name()
        print("Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = self.j.get_numaxes()
        # print("Number of axes: {}".format(axes))

        for i in range(axes):
            axis = self.j.get_axis(i)
            print("Axis {} value: {:>6.3f}".format(i, axis))
        # time.sleep(0.5)

        buttons = self.j.get_numbuttons()
        # print("Number of buttons: {}".format(buttons))

        for i in range(buttons):
            button = self.j.get_button(i)
            print("Button {:>2} value: {}".format(i, button))
        
        time.sleep(0.5)

    def speeds_from_joystick(self):
        cycle_max_t = self.lin_speed_ratio  # 0.2*factor
        cycle_max_r = self.rot_speed_ratio  # 0.1*factor

        if abs(self.j.get_axis(1)) < self.min_joy_position:
            x = 0.0
        else:
            x = -self.j.get_axis(1) * cycle_max_t

        if abs(self.j.get_axis(0)) < self.min_joy_position:
            y = 0.0
        else:
            y = -self.j.get_axis(0) * cycle_max_t

        if abs(self.j.get_axis(3)) < self.min_joy_position:
            rot = 0.0
        else:
            rot = -self.j.get_axis(3) * cycle_max_r

        # Making sure that the xy_speed doesn't go beyond a fixed maximum: (some controllers give (1, 1) when pressed diagonaly instead of (0.5, 0.5)
        xy_speed = math.sqrt(x**2 + y**2)
        max_speed_xy = cycle_max_t
        if xy_speed > max_speed_xy:
            # This formula guarantees that the ratio x/y remains the same, while ensuring the xy_speed is equal to max_speed_xy
            new_x = math.sqrt(max_speed_xy**2 / (1 + (y**2) / (x**2)))
            new_y = new_x * y / x
            # The formula can mess up the signs, fixing them here
            x = sign(x) * new_x / sign(new_x)
            y = sign(y) * new_y / sign(new_y)
        return x, y, rot

    def main_tick(self):
        # print("Tick!!")
        self.tick_controller()


def main():
    node = JoyTeleop()
    
    while True:
        node.main_tick()
        time.sleep((0.01))


if __name__ == "__main__":
    main()
    
    