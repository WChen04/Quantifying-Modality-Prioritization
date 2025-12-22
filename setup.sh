#!/bin/bash
# Setup script for Quantifying Modality Prioritization

set -e

echo "Setting up Quantifying Modality Prioritization..."
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml from template..."
    cp config.yaml.template config.yaml
    echo "Created config.yaml - please edit it with your settings"
else
    echo "config.yaml already exists"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo ""
    echo "Docker detected - you can use Docker for cross-platform support"
    echo "   Note: No virtual environment needed when using Docker!"
    echo "   Run: docker-compose up gpu  (for NVIDIA GPU)"
    echo "   Run: docker-compose up cpu  (for Mac/CPU)"
else
    echo ""
    echo "Docker not found - using local installation"
    echo ""
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo "Python $PYTHON_VERSION detected"
    else
        echo "ERROR: Python 3.11+ required but not found"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "Virtual environment created"
    else
        echo "Virtual environment already exists"
    fi
    
    echo ""
    echo "Installing dependencies..."
    echo "   Activate venv first: source venv/bin/activate"
    echo "   Then install PyTorch and dependencies"
    echo ""
    echo "   For NVIDIA GPU:"
    echo "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo "     pip install bitsandbytes"
    echo ""
    echo "   For Mac/CPU:"
    echo "     pip install torch torchvision torchaudio"
    echo ""
    echo "   Then:"
    echo "     pip install -r requirements.txt"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your model settings"
echo "2. Run experiments: python main.py --mode full"
echo "   Or use Docker: docker-compose up gpu (or cpu)"

