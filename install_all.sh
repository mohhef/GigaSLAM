#!/bin/bash
# GigaSLAM Installation Script
# Large-Scale Monocular SLAM with Hierarchical Gaussian Splats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GigaSLAM Installation ==="

# Determine conda location
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Error: Could not find conda installation"
    exit 1
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^gigaslam "; then
    echo "Conda environment 'gigaslam' already exists, activating..."
    conda activate gigaslam
else
    echo "Creating conda environment 'gigaslam'..."
    conda create -n gigaslam python=3.10 -y
    conda activate gigaslam
fi

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install CUDA toolkit 11.8 for building extensions
echo "Installing CUDA toolkit 11.8..."
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y

# Install torch_scatter separately (requires PyTorch to be installed first)
echo "Installing torch_scatter..."
wget -nc https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_scatter-2.1.2%2Bpt22cu118-cp310-cp310-linux_x86_64.whl || true
pip install ./torch_scatter-2.1.2+pt22cu118-cp310-cp310-linux_x86_64.whl

# Install xformers for CUDA 11.8 (before requirements.txt to avoid version mismatch)
echo "Installing xformers for CUDA 11.8..."
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118

# Install Python requirements (torch_scatter and xformers already installed above)
echo "Installing Python requirements..."
pip install -r requirements.txt

# Build 3D GS Rendering Module
echo "Building 3D Gaussian Splatting modules..."
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization

# Install OpenCV C++ API (needed for DBoW2)
echo "Installing OpenCV C++ dependencies..."
sudo apt-get install -y libopencv-dev || echo "Warning: Could not install libopencv-dev. You may need to install it manually."

# Build DBoW2
echo "Building DBoW2..."
cd DBoW2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install || echo "Warning: Could not install DBoW2 system-wide. Continuing..."
cd "$SCRIPT_DIR"

# Build DPRetrieval
echo "Building DPRetrieval..."
pip install ./DPRetrieval

# Build loop closure correction module
echo "Building loop closure correction module..."
python setup.py install

# Download ORB vocabulary
echo "Downloading ORB vocabulary..."
if [ ! -f ORBvoc.txt ]; then
    wget -c https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
    tar -xzvf ORBvoc.txt.tar.gz
    rm -f ORBvoc.txt.tar.gz
else
    echo "ORBvoc.txt already exists"
fi

echo ""
echo "=== GigaSLAM Installation Complete ==="
echo ""
echo "To use GigaSLAM:"
echo "  conda activate gigaslam"
echo "  python slam.py --config configs/kitti_06.yaml"
echo ""
echo "Note: UniDepth weights will be downloaded automatically on first run."
echo ""
