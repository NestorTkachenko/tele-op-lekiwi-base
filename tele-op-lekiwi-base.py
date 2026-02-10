

import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(
        remote_ip="10.0.0.95",
        id="my_lekiwi",
        teleop_keys={
            "forward": "w",
            "backward": "s",
            "left": "j",
            "right": "l",
            "rotate_left": "a",
            "rotate_right": "d",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "q",
        },
    )
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    keyboard.connect()


    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Keyboard
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)


        action = base_action

        # Send action to robot
        _ = robot.send_action(action)


        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()