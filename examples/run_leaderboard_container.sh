#!/usr/bin/env bash
# Run the full EEG-FM-zoo leaderboard inside emeg-fm's NGC 26.06 container
# (braindecode 1.5.2 + emeg-fm + fmscope), with neuro-dynadojo mounted at /ndd.
set -euo pipefail
IMAGE="${EEGFM_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
EMEG_FM="${EMEG_FM:-$HOME/dev/emeg-fm}"; T9="${T9:-/mnt/t9}"
NDD="${NDD:-$HOME/Workspace/neuro-dynadojo}"
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
PYPATH="$T9/tokfix:/emeg-fm:/emeg-fm/fmscope:$T9/moabblibs:$T9/eegfm_libs_2606:$T9/tsfmlibs:/ndd/src"
exec docker run --rm --gpus all --ipc=host \
  -v "$EMEG_FM:/emeg-fm" -v "$T9:$T9" -v "/data:/data:ro" -v "$NDD:/ndd" \
  ${NDD_SCRATCH:+-v "$NDD_SCRATCH:/scratch"} \
  -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$PYPATH" -e HF_HOME="$T9/hf" \
  -e HF_TOKEN="$HF_TOKEN" -e NDD_DEV="${NDD_DEV:-cpu}" -e NDD_SKIP_ZOO="${NDD_SKIP_ZOO:-}" -e NDD_NSUBJ="${NDD_NSUBJ:-40}" -e MNE_DATA="$T9/moabb_data" \
  -e NDD_JSON="${NDD_JSON:-}" -e NDD_FMS="${NDD_FMS:-}" -e NDD_OSL="${NDD_OSL:-}" -e NDD_SEEDS="${NDD_SEEDS:-12}" -e NDD_NPER="${NDD_NPER:-40}" -e NDD_LEADFIELD="${NDD_LEADFIELD:-radial}" -e NDD_PROBE="${NDD_PROBE:-linear}" -w /emeg-fm \
  "$IMAGE" python "${1:-/ndd/examples/multi_fm_leaderboard_container.py}"
