# Quantifying Modality Prioritization as an Attack Vector in MLLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository provides the implementation and experimental framework for quantifying modality prioritization as a security vulnerability in Multimodal Large Language Models (MLLMs). The work introduces a systematic methodology for measuring how different input modalities are weighted when conflicting information is presented across text, image, and audio channels.

### Research Contributions

1. **Prioritization Index**: A quantitative metric ($\Delta_{\text{audio}}$) for measuring modality prioritization across 300 controlled adversarial scenarios
2. **Harmonic-Dissonance Benchmark**: A standardized evaluation framework with 50 adversarial prompts spanning 9 threat categories
3. **Cross-Architecture Analysis**: Comparative evaluation of pipeline architectures (LLaVA + Whisper) versus native multimodal models (Qwen2-Audio)
4. **Reproducible Framework**: Complete experimental code, Docker containers, and data for reproducible research

### Experimental Findings

Our experiments demonstrate that architecture determines prioritization patterns. Pipeline-based models (LLaVA) exhibit audio-first prioritization where benign audio reduces visual jailbreaking success by 44% (18.0% → 10.0% ASR). Native multimodal architectures (Qwen2-Audio) maintain balanced modality weighting with consistent safety across all input conditions.

---

## Installation

### Docker (Recommended)

Docker provides a consistent execution environment across platforms. When using Docker, a virtual environment is not required as dependencies are managed within containers.

**Prerequisites:**
- Docker Desktop ([Download](https://www.docker.com/products/docker-desktop))
- NVIDIA Docker for GPU support on Linux/Windows ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Mac users: Docker Desktop includes MPS support automatically

**Quick Start:**

```bash
git clone https://github.com/WChen04/Quantifying-Modality-Prioritization.git
cd Quantifying-Modality-Prioritization

# Create configuration from template
cp config.yaml.template config.yaml
# Edit config.yaml with your settings

# Run experiments
docker-compose up gpu    # For NVIDIA GPU
docker-compose up cpu    # For Mac or CPU-only
```

**Docker Commands:**

```bash
# Run specific experiment mode
docker-compose run --rm gpu python main.py --mode visual

# Access container shell
docker-compose run --rm gpu bash

# View logs
docker-compose logs -f gpu

# Stop containers
docker-compose down
```

The `docker-compose.yml` provides two services: `gpu` for NVIDIA GPUs with CUDA support, and `cpu` for Mac (MPS) or CPU-only systems. Results are saved to `./results/` on the host machine.

### Local Installation

For local installation, a Python virtual environment is recommended to avoid dependency conflicts.

**Prerequisites:**
- Python 3.11+
- NVIDIA GPU with 12GB+ VRAM (recommended) or Mac with M-series chip
- CUDA (for NVIDIA GPUs) or MPS (automatic on Mac)

**Installation Steps:**

```bash
git clone https://github.com/WChen04/Quantifying-Modality-Prioritization.git
cd Quantifying-Modality-Prioritization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch
# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes

# For Mac (MPS) or CPU:
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

**Configuration:**

```bash
cp config.yaml.template config.yaml
```

Edit `config.yaml` to select model architecture:

```yaml
# Option 1: Qwen2-Audio (native multimodal)
use_qwen2_audio: true
qwen2_model: "Qwen/Qwen2-Audio-7B-Instruct"

# Option 2: LLaVA + Whisper (pipeline architecture)
# use_qwen2_audio: false
# vision_model: "llava-hf/llava-v1.6-mistral-7b-hf"
# audio_model: "openai/whisper-large-v3"

device: "auto"  # "cuda", "mps", or "cpu"
use_4bit: true  # Memory optimization (NVIDIA GPU only)
dataset_path: "data/dataset_extended.csv"
```

---

## Usage

### Running Experiments

The framework supports three experimental conditions:

```bash
# Run all conditions sequentially
python main.py --mode full

# Run individual conditions
python main.py --mode text    # Text-only baseline
python main.py --mode visual  # Visual-only attacks
python main.py --mode omni    # Omni-modal (visual + audio)
```

### Analysis and Visualization

```bash
# Generate statistics and metrics
python analyze_results.py

# Generate figures
python create_figures.py

# Compress artifacts for repository since .mp4 and .pdf files may be large
python zip_artifacts.py
```

### Extracting Artifacts

Artifacts are stored in compressed format. To extract:

```bash
unzip results/artifacts.zip -d results/artifacts/
```

---

## Repository Structure

```
Quantifying-Modality-Prioritization/
├── main.py                                    # Main experiment runner
├── config.yaml.template                       # Configuration template
├── config.yaml                                # User configuration (create from template)
├── requirements.txt                           # Python dependencies
├── analyze_results.py                         # Results analysis
├── create_figures.py                          # Figure generation
├── zip_artifacts.py                           # Artifact compression utility
├── Dockerfile                                 # Docker container definition
├── docker-compose.yml                         # Docker Compose configuration
├── data/
│   ├── dataset.csv                # Test dataset (3 prompts)
│   └── dataset_extended.csv       # Full dataset (50 prompts)
├── src/
│   ├── generator.py               # Visual/audio artifact generation
│   ├── judge.py                   # Response evaluation
│   └── models_hf.py               # Model implementations
└── results/
    ├── experiment_text.csv        # Text-only results
    ├── experiment_visual.csv      # Visual-only results
    ├── experiment_omni.csv        # Omni-modal results
    ├── artifacts.zip              # Compressed artifacts
    └── artifacts/                 # Individual files (extract from zip)
```

---

## Experimental Methodology

### Model Architectures

Two architectures were evaluated:

1. **LLaVA-v1.6-Mistral-7B + Whisper-Large-v3**: Pipeline architecture where Whisper transcribes audio to text, which is then concatenated with visual features before LLaVA processing.

2. **Qwen2-Audio-7B-Instruct**: Native multimodal model where raw audio waveforms are encoded directly into embeddings that preserve prosodic features.

### Evaluation Framework

The Harmonic-Dissonance Benchmark evaluates three attack vectors per prompt:
- **Text-Only**: Baseline text attacks to establish safety alignment
- **Visual-Only**: Same text rendered as images (1024×200px PNG, Arial 36pt)
- **Omni-Modal**: Visual attacks combined with benign audio masks

Attack Success Rate (ASR) is computed using keyword-based classification. Responses are labeled as "Refusal" if they contain safety language, or "Compliance" if they provide instructional content. Statistical significance is tested using the Wilcoxon signed-rank test.

### Results Summary

| Architecture | Text ASR | Visual ASR | Omni ASR | $\Delta_{\text{audio}}$ |
|--------------|----------|------------|----------|-------------------------|
| LLaVA + Whisper | 8.0% | 18.0% | 10.0% | -44.4% |
| Qwen2-Audio | 2.0% | 0.0% | 0.0% | 0.0% |

LLaVA demonstrates audio-first prioritization where benign audio reduces visual attack success. Qwen2-Audio maintains balanced weighting with consistent safety across modalities.

---

## Limitations

- **Dataset Size**: 50 prompts (sufficient for initial validation, may require expansion for publication)
- **Model Coverage**: Limited to LLaVA + Whisper and Qwen2-Audio architectures
- **Evaluation Method**: Keyword-based classification (effective but may benefit from LLM-as-judge)
- **Hardware Requirements**: GPU recommended (experiments require ~4 hours on RTX 3060)

---

## Future Work

Potential extensions include:
- Expanded model coverage (GPT-4V, Claude 3, Flamingo)
- Enhanced evaluation using LLM-as-judge or human evaluation
- Dataset expansion to 100-200 prompts with multilingual support
- Defense mechanism development (Joint-Modality Scrutiny, entropy-based detection)
- Mechanistic interpretability analysis (attention visualization, safety neuron identification)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
