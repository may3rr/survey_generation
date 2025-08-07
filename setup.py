#!/usr/bin/env python3
"""
Setup script for Survey Generation System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed",
        "outputs",
        "logs",
        "configs/custom"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_env_template():
    """Create environment variable template"""
    env_content = """# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
QWEN_API_KEY=your_qwen_api_key_here
GPT_API_KEY=sk-gakaBMw0kKz3KaTSC559F71c481e4417A6Dc732c33AbAc90

# Data Configuration
DATA_PATH=./data/raw/original_survey_df.pkl
PROCESSED_DATA_DIR=./data/processed
OUTPUT_DIR=./outputs

# Model Configuration
DEFAULT_MODEL=gpt-4o-mini
QWEN_MODEL=qwen3-14b"""
    
    with open(".env.template", "w") as f:
        f.write(env_content)
    print("✅ Environment template created (.env.template)")

def main():
    """Main setup function"""
    print("🚀 Setting up Survey Generation System...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("You can install dependencies manually: pip install -r requirements.txt")
    
    # Setup directories
    setup_directories()
    
    # Download NLTK data
    if not download_nltk_data():
        print("You can download NLTK data manually: python -c \"import nltk; nltk.download('punkt')\"")
    
    # Create environment template
    create_env_template()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and fill in your API keys")
    print("2. Place your raw survey data in data/raw/original_survey_df.pkl")
    print("3. Run: python run_experiments.py --setup")
    print("4. Start experiments: python run_experiments.py gpt_main")

if __name__ == "__main__":
    main()