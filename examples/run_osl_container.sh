#!/usr/bin/env bash
# Run the osl-dynamics TDE-HMM contender inside the neurojax/oracle-osl container
# (osl-dynamics 3.x + TensorFlow), with neuro-dynadojo mounted at /ndd.
set -euo pipefail
IMAGE="${OSL_IMAGE:-neurojax/oracle-osl:latest}"
NDD="${NDD:-$HOME/Workspace/neuro-dynadojo}"
exec docker run --rm --gpus all --ipc=host \
  -v "$NDD:/ndd" ${NDD_SCRATCH:+-v "$NDD_SCRATCH:/scratch"} \
  -e PYTHONPATH=/ndd/src -e NDD_NPER="${NDD_NPER:-40}" \
  -e NDD_JSON="${NDD_JSON:-}" -e NDD_OSL="${NDD_OSL:-}" -e NDD_SEEDS="${NDD_SEEDS:-12}" \
  -e TF_CPP_MIN_LOG_LEVEL=3 -w /ndd \
  "$IMAGE" python "${1:-/ndd/examples/osl_dynamics_scenarios.py}"
