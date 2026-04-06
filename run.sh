#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Find a suitable Python (prefer 3.13, fall back to 3.12, 3.11, 3.10, then generic)
PYTHON=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        ver=$("$py" -c "import sys; print(sys.version_info[:2])")
        major=$("$py" -c "import sys; print(sys.version_info[0])")
        minor=$("$py" -c "import sys; print(sys.version_info[1])")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10-3.13 required (Open3D/pyvista compatibility)"
    echo "Install with: brew install python@3.13"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

# Create output directories
mkdir -p splat/depth_frames logs

# Run tests first
echo ""
echo "Running component tests..."
python test_components.py

echo ""
echo "Starting 3D World Mapper..."
echo "Press 'q' in any OpenCV window or Ctrl+C to quit"
echo ""

python main.py
