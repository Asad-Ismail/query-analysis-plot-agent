#!/bin/bash
# Modern Setup with uv - Fast & Auto-fixes Dependencies

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (modern pip replacement)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo ""
fi

echo "✅ uv installed"
echo ""

# Install dependencies using uv (will auto-resolve numpy conflict!)
echo "⚡ Installing dependencies with uv (fast!)..."
echo ""

uv pip install flask flask-cors python-dotenv pyyaml

echo "  ✓ Flask and utilities"

uv pip install "numpy>=1.26.0,<2.0.0" pandas matplotlib seaborn

echo "  ✓ Data science libraries"

uv pip install langchain==0.2.16 langchain-openai==0.1.23 langgraph==0.2.28 langfuse==2.35.0

echo "  ✓ LangChain stack"

uv pip install openai==1.109.1 tiktoken==0.12.0