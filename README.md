# Quantifying Modality Prioritization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_Format-blue)](paper/main.pdf)
[![Benchmark](https://img.shields.io/badge/Benchmark-Harmonic_Dissonance-red)](src/)

**Quantifying Modality Prioritization** is a research framework investigating a critical security vulnerability in Multimodal Large Language Models (MLLMs): the asymmetry of trust between sensory inputs.

We introduce the **Harmonic-Dissonance Benchmark**, a testing suite that validates the _Distraction Hypothesis_. This hypothesis posits that MLLMs exhibit a "Safe-Modality Bias," preferentially trusting benign audio inputs (low entropy) over malicious visual threats (high entropy) when resolving safety conflicts.

[Image of System Architecture Diagram]

## 🧪 The "Harmonic-Dissonance" Benchmark

This repository contains the official implementation of the Harmonic-Dissonance framework, designed to measure the **Audio Multiplier Effect**—the increase in Attack Success Rate (ASR) when audio is introduced to visual attacks.

### Key Features

- **Procedural Injection:** Automatically renders typographic visual threats and synthesizes benign audio masks using `gTTS`.
- **LLM-as-a-Judge:** Uses an independent, clean GPT-4o instance to rigorously evaluate jailbreak success on a 5-point Likert scale (Refusal $\to$ Full Compliance).
- **Modality Isolation:** Supports A/B testing across Text-Only, Visual-Only, and Omni-Modal (Audio+Visual) vectors.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API Key (or local LLaVA endpoint)
- `ffmpeg` (required for audio processing)

### Installation

```bash
git clone [https://github.com/yourusername/quantifying-modality-prioritization.git](https://github.com/yourusername/quantifying-modality-prioritization.git)
cd quantifying-modality-prioritization
pip install -r requirements.txt
```
