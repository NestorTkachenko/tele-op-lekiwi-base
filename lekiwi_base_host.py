import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import numpy as np
import zmq

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from lerobot.robots.robot import Robot

SPEED_SCALE = 2.0


@dataclass
class LeKiwiBaseServerConfig:
    """Configuration for the LeKiwi base-only host script."""

    robot: LeKiwiConfig = field(default_factory=LeKiwiConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwiBase(Robot):
    config_class = LeKiwiConfig
    name = "lekiwi_base"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base_left_wheel": Motor(8, "sts3215", norm_mode_body),
                "base_back_wheel": Motor(9, "sts3215", norm_mode_body),
                "base_right_wheel": Motor(7, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )
        self.base_motors = list(self.bus.motors.keys())
        self.cameras = {}

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(("x.vel", "y.vel", "theta.vel"), float)

    @property
    def observation_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logging.info("No calibration file found; running base calibration.")
            self.calibrate()
        self.configure()
        logging.info("%s connected.", self)

    def calibrate(self) -> None:
        homing_offsets = dict.fromkeys(self.base_motors, 0)
        range_mins = dict.fromkeys(self.base_motors, 0)
        range_maxes = dict.fromkeys(self.base_motors, 4095)

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        return raw_speed / steps_per_deg

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 6000,
    ) -> dict[str, int]:
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        angles = np.radians(np.array([240, 0, 120]) - 90)
        matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        wheel_linear_speeds = matrix.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed: int,
        back_wheel_speed: int,
        right_wheel_speed: int,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, float]:
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius

        angles = np.radians(np.array([240, 0, 120]) - 90)
        matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        velocity_vector = np.linalg.inv(matrix).dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)

        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }

    def get_observation(self) -> RobotObservation:
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)
        return self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

    def send_action(self, action: RobotAction) -> RobotAction:
        base_goal_vel = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
        base_goal_vel.update({k: v for k, v in action.items() if k.endswith(".vel")})

        base_goal_vel = {k: v * SPEED_SCALE for k, v in base_goal_vel.items()}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"],
            base_goal_vel["y.vel"],
            base_goal_vel["theta.vel"],
        )
        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)
        return base_goal_vel

    def stop_base(self) -> None:
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logging.info("Base motors stopped")

    def disconnect(self) -> None:
        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        logging.info("%s disconnected.", self)


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self) -> None:
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: LeKiwiBaseServerConfig) -> None:
    logging.info("Configuring LeKiwi base-only host")
    robot = LeKiwiBase(cfg.robot)

    logging.info("Connecting LeKiwi base")
    robot.connect()

    logging.info("Starting HostAgent")
    host = LeKiwiHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        while True:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as exc:
                logging.error("Message fetching failed: %s", exc)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    "Command not received for more than %s milliseconds. Stopping the base.",
                    host.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down LeKiwi base host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi base-only host cleanly")


if __name__ == "__main__":
    main()
