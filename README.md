# Quantifying Modality Prioritization as an Attack Vector in MLLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Research Paper**: [modality-prioritization-attack-vector.tex](modality-prioritization-attack-vector.tex)

## 📄 Overview

This repository contains the implementation of research investigating a critical security vulnerability in Multimodal Large Language Models (MLLMs): **Cross-Modal Indirect Prompt Injection (CM-IPI)**.

### The Core Finding

We discovered that benign audio inputs (e.g., "What color is this?") systematically amplify visual jailbreaking attacks on MLLMs. This **Audio Multiplier Effect** works by diverting the model's attention from malicious visual content, creating an exploitable "Hearing is Trusting" bias.

### Key Contributions

1. **Distraction Hypothesis**: Formalized how benign audio distracts attention from visual threats
2. **Harmonic-Dissonance Benchmark**: 50 adversarial prompts across 9 threat categories
3. **Cross-Architecture Validation**: Tested on LLaVA-v1.6 + Whisper and Qwen2-Audio
4. **Open-Source Framework**: Fully reproducible experiments

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **GPU**: NVIDIA GPU with 12GB+ VRAM recommended (RTX 3060 or better)
- **CUDA**: For GPU acceleration

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Quantifying-Modality-Prioritization.git
cd Quantifying-Modality-Prioritization

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to select your model backend:

```yaml
# Hugging Face (Open Source) - Recommended
backend: "huggingface"
vision_model: "llava-hf/llava-v1.6-mistral-7b-hf"
audio_model: "openai/whisper-large-v3"
use_4bit: true  # Memory optimization
dataset_path: "data/dataset_extended.csv"
```

### Running Experiments

```bash
# Run all three experimental conditions
python main.py --mode full

# Or run individually:
python main.py --mode text    # Text-only baseline
python main.py --mode visual  # Visual-only attacks
python main.py --mode omni    # Omni-modal (visual + audio)
```

### Analyzing Results

```bash
# Generate statistics and metrics
python analyze_results.py

# Create figures for paper
python create_figures.py
```

---

## 📊 Repository Structure

```
Quantifying-Modality-Prioritization/
├── modality-prioritization-attack-vector.tex  # Research paper (LaTeX)
├── main.py                                     # Main experiment runner
├── config.yaml                                 # Configuration file
├── requirements.txt                            # Python dependencies
├── analyze_results.py                          # Results analysis
├── create_figures.py                           # Figure generation
├── data/
│   ├── dataset.csv                # Small test dataset (3 prompts)
│   └── dataset_extended.csv       # Full dataset (50 prompts)
├── src/
│   ├── generator.py               # Generates visual/audio artifacts
│   ├── judge.py                   # Evaluates model responses
│   ├── models.py                  # Gemini models (deprecated)
│   └── models_hf.py               # Hugging Face models
└── results/
    ├── experiment_text.csv        # Text-only results
    ├── experiment_visual.csv      # Visual-only results
    ├── experiment_omni.csv        # Omni-modal results
    └── artifacts/                 # Generated images & audio
```

---

## ✅ What Worked

### Successful Components

1. **LLaVA + Whisper Pipeline**
   - Transcription-based approach works reliably
   - 4-bit quantization enables 12GB GPU usage
   - Consistent results across 50 prompts

2. **Visual Jailbreaking**
   - Typographic attacks (text-as-image) successfully bypass text filters
   - 1024×200px PNG images with Arial 36pt font work best

3. **Audio Generation**
   - Google Text-to-Speech (gTTS) creates effective benign masks
   - Generic questions ("What color is this?") work without optimization

4. **Evaluation Framework**
   - Keyword-based ASR calculation is fast and reliable
   - Statistical validation via Wilcoxon signed-rank test

### Validated Findings

- ✅ Audio consistently amplifies visual attacks
- ✅ Effect is architecture-independent (LLaVA and Qwen2-Audio)
- ✅ No adversarial optimization needed
- ✅ Black-box attack (API access only)

---

## ⚠️ Known Issues & Limitations

### What Didn't Work

1. **Gemini Backend**
   - Initially planned to use Gemini 2.5 Flash
   - Removed due to API access issues and cost
   - Pivoted to open-source models

2. **Text-Only Experiments**
   - Required workaround: LLaVA needs dummy images for text-only mode
   - Fixed by using blank 1×1 pixel placeholder images

3. **Memory Constraints**
   - Full-precision models require 40GB+ VRAM
   - Solution: 4-bit quantization via bitsandbytes

### Current Limitations

- **Dataset Size**: 50 prompts (sufficient for course paper, small for publication)
- **Model Coverage**: Only tested LLaVA + Whisper and Qwen2-Audio
- **Evaluation**: Keyword-based (simple but effective for proof-of-concept)
- **Hardware**: Requires GPU (experiments took ~4 hours on RTX 3060)

---

## 🔮 Future Work

### Planned Improvements

1. **Expand Model Coverage**
   - Test GPT-4V, GPT-4o, Claude 3
   - Evaluate Gemini Pro (if API access available)
   - Try Flamingo and BLIP-2 architectures

2. **Enhanced Evaluation**
   - Use LLM-as-judge (GPT-4) for more nuanced scoring
   - Implement human evaluation for gold standard
   - Add adversarial robustness metrics

3. **Dataset Expansion**
   - Increase to 100-200 prompts
   - Add multilingual prompts
   - Test different audio types (music, ambient sounds)

4. **Defense Mechanisms**
   - Implement Joint-Modality Scrutiny layers
   - Test entropy-based detection
   - Develop adversarial training protocols

5. **Mechanistic Interpretability**
   - Visualize attention heads during attacks
   - Identify safety neurons across modalities
   - Trace activation pathways

---

## 📖 For Beginners

### Understanding the Code

**Start here if you're new to multimodal AI security:**

1. **Read the Paper First**: Open `modality-prioritization-attack-vector.tex` (or compile to PDF)
2. **Explore the Dataset**: Check `data/dataset.csv` to see example prompts
3. **Run Small Test**: Use the 3-prompt dataset first
   ```bash
   # Edit config.yaml: dataset_path: "data/dataset.csv"
   python main.py --mode visual
   ```
4. **Check Results**: Look at `results/experiment_visual.csv`
5. **View Artifacts**: See generated images in `results/artifacts/`

### Key Concepts

- **Modality**: Input type (text, image, audio)
- **Jailbreaking**: Bypassing AI safety mechanisms
- **ASR (Attack Success Rate)**: % of successful attacks
- **Audio Multiplier Effect**: How much audio increases ASR
- **Late-Fusion**: Processing modalities separately before combining

### Troubleshooting

**GPU Out of Memory?**
```yaml
use_4bit: true  # In config.yaml
```

**Slow Experiments?**
```yaml
dataset_path: "data/dataset.csv"  # Use small dataset first
```

**Import Errors?**
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{chen2025modality,
  title={Quantifying Modality Prioritization as an Attack Vector in Multimodal Large Language Models},
  author={Chen, William},
  journal={arXiv preprint},
  year={2025},
  institution={Rensselaer Polytechnic Institute}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Testing additional model architectures
- Expanding the dataset
- Implementing defense mechanisms
- Improving evaluation metrics

---

## ⚖️ Ethical Considerations

This research is conducted for AI safety purposes only. All experiments:
- Use synthetic adversarial prompts (no real harmful content)
- Run on isolated systems
- Findings shared with model developers before publication

**Do not use these techniques maliciously.**

---

## 📧 Contact

**William Chen**  
Rensselaer Polytechnic Institute  
Email: chenw21@rpi.edu

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repo if you find it useful for your research!**
