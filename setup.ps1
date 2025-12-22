# Setup script for Quantifying Modality Prioritization (Windows PowerShell)

Write-Host "Setting up Quantifying Modality Prioritization..." -ForegroundColor Cyan
Write-Host ""

# Check if config.yaml exists
if (-not (Test-Path "config.yaml")) {
    Write-Host "Creating config.yaml from template..." -ForegroundColor Yellow
    Copy-Item "config.yaml.template" "config.yaml"
    Write-Host "Created config.yaml - please edit it with your settings" -ForegroundColor Green
} else {
    Write-Host "config.yaml already exists" -ForegroundColor Green
}

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>$null
    Write-Host ""
    Write-Host "Docker detected - you can use Docker for cross-platform support" -ForegroundColor Green
    Write-Host "   Note: No virtual environment needed when using Docker!" -ForegroundColor Yellow
    Write-Host "   Run: docker-compose up gpu  (for NVIDIA GPU)" -ForegroundColor Cyan
    Write-Host "   Run: docker-compose up cpu  (for Mac/CPU)" -ForegroundColor Cyan
} catch {
    Write-Host ""
    Write-Host "Docker not found - using local installation" -ForegroundColor Yellow
    Write-Host ""
    
    # Check Python version
    try {
        $pythonVersion = python --version 2>$null
        Write-Host "$pythonVersion detected" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Python 3.11+ required but not found" -ForegroundColor Red
        exit 1
    }
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "Virtual environment already exists" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    Write-Host "   Activate venv first: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "   Then install PyTorch and dependencies" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   For NVIDIA GPU:" -ForegroundColor Cyan
    Write-Host "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor White
    Write-Host "     pip install bitsandbytes" -ForegroundColor White
    Write-Host ""
    Write-Host "   For CPU:" -ForegroundColor Cyan
    Write-Host "     pip install torch torchvision torchaudio" -ForegroundColor White
    Write-Host ""
    Write-Host "   Then:" -ForegroundColor Cyan
    Write-Host "     pip install -r requirements.txt" -ForegroundColor White
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit config.yaml with your model settings" -ForegroundColor White
Write-Host "2. Run experiments: python main.py --mode full" -ForegroundColor White
Write-Host "   Or use Docker: docker-compose up gpu (or cpu)" -ForegroundColor White

