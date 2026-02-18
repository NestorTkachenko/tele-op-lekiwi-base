from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AAF2627211",
    id="my_awesome_follower_arm2",
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AAF2641301",
    id="my_awesome_leader_arm2",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()

    robot.send_action(action)