#!/bin/bash
export LIVEKIT_AGENTS_DISABLE_INFERENCE_EXECUTOR=1
cd "$(dirname "$0")"
uv run python src/agent.py dev
