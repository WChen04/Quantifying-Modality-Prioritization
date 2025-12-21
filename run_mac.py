#!/usr/bin/env python3
"""
Mac M4 Runner for Qwen2-Audio Experiments
Uses config_mac.yaml with MPS (Metal) support
"""

import subprocess
import sys

def main():
    print("🍎 Starting Qwen2-Audio experiments on Mac M4...")
    print("=" * 60)
    
    # Use config_mac.yaml instead of default config.yaml
    import yaml
    import os
    
    # Load Mac config
    with open('config_mac.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily replace config.yaml
    import shutil
    if os.path.exists('config.yaml'):
        shutil.copy('config.yaml', 'config.yaml.backup')
    shutil.copy('config_mac.yaml', 'config.yaml')
    
    try:
        # Run full experiment
        subprocess.run([sys.executable, 'main.py', '--mode', 'full'], check=True)
    finally:
        # Restore original config
        if os.path.exists('config.yaml.backup'):
            shutil.move('config.yaml.backup', 'config.yaml')
    
    print("=" * 60)
    print("✅ Qwen2-Audio experiments complete!")
    print("Results saved to: results_qwen2/")

if __name__ == "__main__":
    main()

