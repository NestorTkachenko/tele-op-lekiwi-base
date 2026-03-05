# SO101 Delta Inference

Use this script to run a delta-trained policy on SO101 hardware.

The script converts model output from delta joint actions to absolute joint targets before sending commands:

`absolute_action = observation.state + predicted_delta_action`

## Command (your requested setup)

- Robot port: `/dev/ttyACM0`
- Wrist camera index: `0`
- Context (top) camera index: `1`
- Inference frames: `20`
- Training/inference FPS: `10`

```bash
python run_so101_delta_inference.py \
  --policy-path <PATH_OR_HF_REPO_TO_TRAINED_POLICY> \
  --dataset-repo-id nuffnuff/pi05pnptest_delta \
  --port /dev/ttyACM0 \
  --wrist-cam-index 0 \
  --context-cam-index 1 \
  --inference-frames 20 \
  --fps 10 \
  --device cpu
```

If you are on GPU/MPS, change `--device` to `cuda` or `mps`.
