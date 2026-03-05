import argparse
import time

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
        description="Run a delta-action policy live on SO101 by converting predicted delta actions to absolute joint targets."
    )
    parser.add_argument("--policy-path", required=True, help="Path or HF repo for trained checkpoint")
    parser.add_argument("--dataset-repo-id", required=True, help="Delta dataset repo used for policy feature/stats")
    parser.add_argument("--dataset-root", default=None, help="Optional local root for dataset cache")
    parser.add_argument("--port", default="/dev/ttyACM0", help="SO101 serial port")
    parser.add_argument("--robot-id", default="so101_delta_runner")
    parser.add_argument("--wrist-cam-index", type=int, default=0)
    parser.add_argument("--context-cam-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10, help="Control/inference loop frequency")
    parser.add_argument("--inference-frames", type=int, default=20, help="Policy n_action_steps override")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-seconds", type=int, default=0, help="0 means run until Ctrl+C")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=False,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = args.inference_frames

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
                fps=args.fps,
                width=args.width,
                height=args.height,
            ),
            "camera_top": OpenCVCameraConfig(
                index_or_path=args.context_cam_index,
                fps=args.fps,
                width=args.width,
                height=args.height,
            ),
        },
    )
    robot = make_robot_from_config(robot_cfg)

    observation_state_names = dataset.features["observation.state"]["names"]

    print("Connecting robot and cameras...")
    robot.connect()
    print("Running delta inference. Press Ctrl+C to stop.")

    start_time = time.perf_counter()
    device = get_safe_torch_device(args.device)

    try:
        while True:
            t0 = time.perf_counter()

            if args.max_seconds > 0 and (t0 - start_time) >= args.max_seconds:
                break

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

            delta_action = make_robot_action(action_values, dataset.features)
            obs_state = obs_frame["observation.state"]
            obs_state_map = dict(zip(observation_state_names, obs_state.tolist(), strict=True))

            absolute_action = {}
            for key, value in delta_action.items():
                if key.endswith(".pos") and key in obs_state_map:
                    absolute_action[key] = float(obs_state_map[key] + value)
                else:
                    absolute_action[key] = float(value)

            robot.send_action(absolute_action)

            dt = time.perf_counter() - t0
            sleep_s = max(1.0 / args.fps - dt, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Stopping inference (Ctrl+C).")
    finally:
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
