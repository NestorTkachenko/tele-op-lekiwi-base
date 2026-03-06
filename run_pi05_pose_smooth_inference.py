import argparse
import threading
import time

import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import OBS_STR, build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pose-based PI05 on SO101 with smooth control: low-rate inference + high-rate interpolated control."
        )
    )
    parser.add_argument("--policy-path", required=True, help="Path or HF repo to trained PI05 policy checkpoint")
    parser.add_argument("--dataset-repo-id", required=True, help="Dataset repo used for policy features/stats")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root")
    parser.add_argument("--port", default="/dev/ttyACM0", help="SO101 serial port")
    parser.add_argument("--robot-id", default="so101_pi05_smooth")
    parser.add_argument("--wrist-cam-index", type=int, default=0)
    parser.add_argument("--context-cam-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--inference-hz", type=float, default=10.0, help="Measured policy inference rate")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Robot command rate")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing on inferred target (0 disables, 1 uses only latest target)",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=8.0,
        help="Max allowed target change per inference step per joint (deg) to limit aggressive jumps",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-seconds", type=int, default=0, help="0 = run until Ctrl+C")
    return parser.parse_args()


def clip_step(target: np.ndarray, previous: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - previous
    delta = np.clip(delta, -max_step, max_step)
    return previous + delta


def main() -> int:
    args = parse_args()

    if args.inference_hz <= 0 or args.control_hz <= 0:
        raise ValueError("--inference-hz and --control-hz must be > 0")
    if not (0.0 <= args.smoothing_alpha <= 1.0):
        raise ValueError("--smoothing-alpha must be in [0, 1]")

    inference_period = 1.0 / args.inference_hz
    control_period = 1.0 / args.control_hz

    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root, download_videos=False)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    robot_cfg = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        cameras={
            "camera_wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_cam_index,
                fps=args.camera_fps,
                width=args.width,
                height=args.height,
            ),
            "camera_top": OpenCVCameraConfig(
                index_or_path=args.context_cam_index,
                fps=args.camera_fps,
                width=args.width,
                height=args.height,
            ),
        },
    )
    robot = make_robot_from_config(robot_cfg)

    action_names = dataset.features["action"]["names"]

    print("Connecting robot and cameras...")
    robot.connect()

    initial_obs = robot.get_observation()
    current_action = np.array([float(initial_obs[name]) for name in action_names], dtype=np.float32)

    shared = {
        "prev_target": current_action.copy(),
        "target": current_action.copy(),
        "seq": 0,
        "stop": False,
    }
    model_lock = threading.Lock()
    io_lock = threading.Lock()

    device = get_safe_torch_device(args.device)

    def inference_worker() -> None:
        while True:
            t0 = time.perf_counter()
            with model_lock:
                should_stop = shared["stop"]
            if should_stop:
                return

            with io_lock:
                raw_obs = robot.get_observation()

            obs_frame = build_dataset_frame(dataset.features, raw_obs, prefix=OBS_STR)
            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(getattr(policy.config, "use_amp", False)),
                robot_type=robot.robot_type,
            )
            predicted_action_dict = make_robot_action(action_values, dataset.features)
            raw_target = np.array([float(predicted_action_dict[name]) for name in action_names], dtype=np.float32)

            with model_lock:
                prev_target = shared["target"].copy()
                bounded = clip_step(raw_target, prev_target, args.max_joint_step_deg)
                smoothed = (1.0 - args.smoothing_alpha) * prev_target + args.smoothing_alpha * bounded
                shared["prev_target"] = prev_target
                shared["target"] = smoothed
                shared["seq"] += 1

            elapsed = time.perf_counter() - t0
            sleep_s = max(inference_period - elapsed, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()

    print(
        f"Running smooth inference: inference={args.inference_hz:.2f}Hz, control={args.control_hz:.2f}Hz, "
        f"alpha={args.smoothing_alpha}, max_step={args.max_joint_step_deg}deg"
    )

    segment_start = current_action.copy()
    segment_end = current_action.copy()
    segment_t0 = time.perf_counter()
    seen_seq = -1
    start_time = time.perf_counter()

    try:
        while True:
            loop_t0 = time.perf_counter()

            if args.max_seconds > 0 and (loop_t0 - start_time) >= args.max_seconds:
                break

            with model_lock:
                seq = int(shared["seq"])
                latest_target = shared["target"].copy()

            if seq != seen_seq:
                segment_start = current_action.copy()
                segment_end = latest_target
                segment_t0 = loop_t0
                seen_seq = seq

            u = min(max((loop_t0 - segment_t0) / inference_period, 0.0), 1.0)
            cmd_vec = (1.0 - u) * segment_start + u * segment_end
            cmd = {name: float(value) for name, value in zip(action_names, cmd_vec.tolist(), strict=True)}

            with io_lock:
                robot.send_action(cmd)

            current_action = cmd_vec

            dt = time.perf_counter() - loop_t0
            sleep_s = max(control_period - dt, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Stopping inference (Ctrl+C).")
    finally:
        with model_lock:
            shared["stop"] = True
        inference_thread.join(timeout=2.0)
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
