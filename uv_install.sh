#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (modern pip replacement)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo ""
fi


uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt