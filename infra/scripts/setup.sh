#!/usr/bin/env bash
# Swara automated setup and dependency verification script
set -euo pipefail

echo "=================================================="
echo "Swara - Automated Production Setup"
echo "CSKY - 'Every Voice Has an Identity.'"
echo "=================================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed." >&2
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[+] Detected Python $PY_VER"

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "[+] Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

echo "[+] Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -e .

echo "[+] Verifying model artifacts..."
swara doctor

echo "[+] Downloading default model checkpoints..."
swara models download

echo "[+] Running local inference benchmark..."
swara benchmark

echo "=================================================="
echo "Swara installation successfully completed!"
echo "Run 'swara --help' or 'swara serve' to start API."
echo "=================================================="
