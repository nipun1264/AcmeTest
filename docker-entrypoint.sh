#!/bin/sh
# Generates the synthetic feed on first run (the starter's own generator
# has no __main__ block, so it's invoked as a function, not a script),
# then hands off to the packaged CLI entry point.
set -eu

CONFIG_PATH="${CONFIG_PATH:-configs/default.json}"
VIDEO_PATH="${VIDEO_PATH:-synthetic_pitch_feed.mp4}"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "generating synthetic video at $VIDEO_PATH"
    python -c "from synthetic_generator import generate_synthetic_video; generate_synthetic_video(outputPath='$VIDEO_PATH')"
fi

exec pitch-engine --config "$CONFIG_PATH" "$@"
