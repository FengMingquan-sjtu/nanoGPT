#!/bin/bash

# nanoGPT Dependencies Installation Script
# This script automates the installation of all required packages for nanoGPT training

set -e  # Exit on any error

echo "=========================================="
echo "nanoGPT Dependencies Installation Script"
echo "=========================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if pip is available
if ! command_exists pip; then
    echo "Error: pip is not installed or not in PATH"
    echo "Please install pip first"
    exit 1
fi

echo "✓ pip is available"
echo ""

# Function to install package with error handling
install_package() {
    local package="$1"
    local description="$2"
    
    echo "Installing $description..."
    echo "Command: $package"
    
    if eval "$package"; then
        echo "✓ Successfully installed $description"
    else
        echo "✗ Failed to install $description"
        echo "Please check the error above and try again"
        exit 1
    fi
    echo ""
}

# Install PyTorch
install_package "pip install torch==2.3.0" "PyTorch 2.3.0"

# Install core libraries with strict versioning for training
install_package "pip install transformers[torch]==4.41.2" "Transformers 4.41.2 with PyTorch support"
install_package "pip install deepspeed==0.15.4" "DeepSpeed 0.15.4"
install_package "pip install accelerate==1.1.1" "Accelerate 1.1.1"
install_package "pip install datasets==2.19.2" "Datasets 2.19.2"
install_package "pip install datatrove==0.3.0 fire matplotlib seaborn wandb" "Additional utilities (datatrove, fire, matplotlib, seaborn, wandb)"

# Install Flash Attention for optimized performance
echo "Installing Flash Attention (this may take a while)..."
echo "Command: MAX_JOBS=8 pip install flash-attn --no-build-isolation"

if MAX_JOBS=8 pip install flash-attn --no-build-isolation; then
    echo "✓ Successfully installed Flash Attention"
else
    echo "✗ Failed to install Flash Attention"
    echo "Note: Flash Attention installation can fail due to CUDA version compatibility"
    echo "You may need to install it manually or skip this step if not using CUDA"
    echo "Continuing with other packages..."
fi

echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "- PyTorch 2.3.0"
echo "- Transformers 4.41.2 (with PyTorch support)"
echo "- DeepSpeed 0.15.4"
echo "- Accelerate 1.1.1"
echo "- Datasets 2.19.2"
echo "- datatrove 0.3.0"
echo "- fire"
echo "- matplotlib"
echo "- seaborn"
echo "- wandb"
echo "- flash-attn (if successful)"
echo ""
echo "You can now run nanoGPT training scripts!"
