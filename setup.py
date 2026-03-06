#!/usr/bin/env python3
"""Setup script for the attention visualization project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def main():
    """Setup the project."""
    print("🚀 Setting up Attention Visualization Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("✗ Failed to install package")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "assets", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Generate synthetic data
    if not run_command("python scripts/generate_data.py", "Generating synthetic data"):
        print("⚠️  Failed to generate synthetic data (optional)")
    
    # Run basic tests
    if not run_command("python -c \"import sys; sys.path.append('src'); from attention_viz.core import AttentionVisualizer; print('✓ Core module imports successfully')\"", "Testing core imports"):
        print("✗ Core module import failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Run the example script:")
    print("     python scripts/example.py")
    print("  2. Launch the interactive demo:")
    print("     streamlit run demo/app.py")
    print("  3. Run tests:")
    print("     pytest tests/ -v")
    print("  4. Install pre-commit hooks (optional):")
    print("     pre-commit install")


if __name__ == "__main__":
    main()
