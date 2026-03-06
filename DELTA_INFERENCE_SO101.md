# SO101 Delta Inference

Use this script to run a delta-trained policy on SO101 hardware.

The script converts model output from delta joint actions to absolute joint targets before sending commands:

`absolute_action = observation.state + predicted_delta_action`

## Delta Inference Command

```bash
python run_so101_delta_inference.py \
  --policy-path <PATH_OR_HF_REPO_TO_TRAINED_DELTA_POLICY> \
  --dataset-repo-id nuffnuff/pi05pnptest_delta \
  --port /dev/ttyACM0 \
  --wrist-cam-index 0 \
  --context-cam-index 1 \
  --inference-frames 20 \
  --fps 10 \
  --device cuda
```

## Smooth Pose Inference Command (Requested)

```bash
python run_pi05_pose_smooth_inference.py \
  --policy-path nuffnuff/pi05pnpcorrect \
  --dataset-repo-id <YOUR_POSE_DATASET_REPO> \
  --port /dev/ttyACM0 \
  --wrist-cam-index 0 \
  --context-cam-index 2 \
  --inference-hz 10 \
  --control-hz 30 \
  --camera-fps 30 \
  --smoothing-alpha 0.35 \
  --max-joint-step-deg 8 \
  --device cuda
```

If you are on Apple Silicon, switch `--device` to `mps`.
