from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AAF2627211",
    id="my_awesome_follower_arm2",
)

robot = SO101Follower(robot_config)

robot.connect()


while True:
    action = {'shoulder_pan.pos': -9.396709323583181, 'shoulder_lift.pos': -20.20159596808064, 'elbow_flex.pos': 57.065706570657056, 'wrist_flex.pos': -62.86083584661783, 'wrist_roll.pos': 6.203995793901157, 'gripper.pos': 1.4263074484944533}

    robot.send_action(action)