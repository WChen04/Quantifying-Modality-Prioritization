#!/usr/bin/env python3
"""
Compress artifacts directory into a zip file for GitHub.
This reduces repository size while preserving all generated artifacts.
"""

import os
import zipfile
from pathlib import Path

def zip_artifacts():
    """Compress artifacts directory into a zip file."""
    artifacts_dir = Path("results/artifacts")
    output_zip = Path("results/artifacts.zip")
    
    if not artifacts_dir.exists():
        print(f"Error: {artifacts_dir} does not exist")
        return False
    
    # Count files
    files = list(artifacts_dir.glob("*.png")) + list(artifacts_dir.glob("*.mp3"))
    if not files:
        print(f"No artifacts found in {artifacts_dir}")
        return False
    
    print(f"Found {len(files)} artifact files")
    print(f"Creating {output_zip}...")
    
    # Create zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            # Store files with just their name (not full path)
            zipf.write(file, file.name)
            print(f"  Added: {file.name}")
    
    # Get file sizes
    total_size = sum(f.stat().st_size for f in files)
    zip_size = output_zip.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print(f"\nCompression complete!")
    print(f"  Original size: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed size: {zip_size / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}%")
    print(f"\nZip file created: {output_zip}")
    
    return True

if __name__ == "__main__":
    zip_artifacts()

