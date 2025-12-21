#!/bin/bash
# Mac M4 Setup Script for Qwen2-Audio Experiments
# Run this on your Mac while Windows runs LLaVA experiments

echo "🍎 Setting up Mac M4 for Qwen2-Audio experiments..."

# Create virtual environment
python3 -m venv venv_mac
source venv_mac/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with MPS (Metal) support for M4
pip3 install torch torchvision torchaudio

# Install transformers and dependencies
pip3 install transformers==4.40.0
pip3 install accelerate
pip3 install pandas pillow gtts pyyaml tqdm soundfile
pip3 install sentencepiece einops

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "✅ ffmpeg already installed"
fi

echo ""
echo "✅ Mac setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy dataset from Windows: scp or USB drive"
echo "2. Update config_mac.yaml with Qwen2 settings"
echo "3. Run: python main.py --mode full"
echo ""

