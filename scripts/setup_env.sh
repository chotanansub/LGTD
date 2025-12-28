#!/bin/bash

# LGTD Environment Setup Script
# This script sets up the development environment for LGTD

set -e  # Exit on error

echo "=================================="
echo "LGTD Environment Setup"
echo "=================================="

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: $MACHINE"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
echo ""
echo "Checking for Python..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_VERSION=$(python --version)
    echo "✓ Found: $PYTHON_VERSION"
    PYTHON_CMD=python
else
    echo "✗ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VER=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VER"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "env" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf env
        $PYTHON_CMD -m venv env
        echo "✓ Virtual environment recreated"
    fi
else
    $PYTHON_CMD -m venv env
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [ "$MACHINE" = "Windows" ]; then
    source env/Scripts/activate
else
    source env/bin/activate
fi
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "1. Installing core dependencies..."
pip install -r requirements.txt
echo "✓ Core dependencies installed"

# Ask if user wants experiment dependencies
echo ""
read -p "Install experiment dependencies (baselines)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing experiment dependencies..."
    pip install -r requirements-experiments.txt
    echo "✓ Experiment dependencies installed"
fi

# Install package in development mode
echo ""
echo "Installing LGTD in development mode..."
pip install -e .
echo "✓ LGTD installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment in the future, run:"
if [ "$MACHINE" = "Windows" ]; then
    echo "  env\\Scripts\\activate"
else
    echo "  source env/bin/activate"
fi
echo ""
echo "To run experiments:"
echo "  python experiments/scripts/run_synthetic.py"
echo ""
echo "To run the demo notebook:"
echo "  jupyter notebook notebooks/LGTD_Quick_Demo.ipynb"
