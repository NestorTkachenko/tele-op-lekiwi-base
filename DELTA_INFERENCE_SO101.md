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
  --policy-path nuffnuff/pi05pnpdelta \
  --dataset.push_to_hub=False\
  --port /dev/ttyACM0 \
  --wrist-cam-index 2 \
  --context-cam-index 0 \
  --inference-frames 30 \
  --fps 10 \
  --device cuda \
  --dataset-repo-id nuffnuff/pi05pnptest_delta1
```

If you are on GPU/MPS, change `--device` to `cuda` or `mps`.
