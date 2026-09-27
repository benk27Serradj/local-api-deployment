#!/bin/bash
# Serves the quantized model locally via an OpenAI-compatible API.
# Usage: ./serve.sh [model-file]  (defaults to the Q4_K_M quantized version)

MODEL="${1:-./models/qwen-q4_k_m.gguf}"
PORT="${PORT:-8080}"

echo "Serving $MODEL on http://localhost:$PORT (OpenAI-compatible API)"
./llama.cpp/build/bin/llama-server -m "$MODEL" --port "$PORT"
