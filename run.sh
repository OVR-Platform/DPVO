#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/video.mp4 [additional run.py args]" >&2
    exit 1
fi

VIDEO_PATH="$1"
shift || true

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Input video '$VIDEO_PATH' not found" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the DPVO pipeline inside a Docker container while saving outputs on the host

VIDEO_ABS="$(realpath "$VIDEO_PATH")"
OUTPUT_HOST_DIR="${DPVO_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
IMAGE_NAME="${DPVO_IMAGE_NAME:-dpvo-runtime}"

mkdir -p "$OUTPUT_HOST_DIR"

DOCKER_CMD=(
    docker run --rm --gpus all --ipc=host
    -v "$VIDEO_ABS":/data/input.mp4:ro
    -v "$OUTPUT_HOST_DIR":/app/outputs
    "$IMAGE_NAME"
    python run.py
    --imagedir=/data/input.mp4
    --calib=calib/tesla.txt
    --stride=5
    --plot
    --save_camera_poses
    --save_motion
    --save_trajectory
)

if [ $# -gt 0 ]; then
    DOCKER_CMD+=("$@")
fi

"${DOCKER_CMD[@]}"
