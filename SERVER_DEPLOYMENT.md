# Server Deployment Guide

## Overview
This document describes the detailed steps for migrating the AutoSurvey experimental environment from a local M3 Pro MacBook Pro to a server. The QFiD model has been successfully integrated into `run_baselines.py` and device compatibility issues have been resolved.

## System Requirements

### Recommended Server Configuration
- **GPU**: NVIDIA GPU (recommended A100/V100/RTX 4090) with at least 16GB VRAM
- **CPU**: 8 cores or more
- **Memory**: 32GB or more (64GB recommended)
- **Storage**: 100GB or more available space
- **Operating System**: Linux (Ubuntu 20.04+ recommended)

### Software Dependencies
- Python 3.8+
- PyTorch 1.12+ (CUDA version)
- CUDA 11.3+
- Git

## Files and Directories to Upload

### Core Code Files
```
AutoSurvey/lhb_survey_generation/
├── experiments/
│   └── baselines/
│       └── run_baselines.py                 # Main experimental script (QFiD integrated)
├── data/
│   └── raw/
│       └── original_survey_df.pkl           # Original dataset
├── SciReviewGen-main/                       # QFiD model code
│   └── qfid/
│       ├── qfid.py                          # QFiD model implementation (device compatibility fixed)
│       └── run_summarization.py             # QFiD training script
├── requirements.txt                         # Python dependencies
└── SERVER_DEPLOYMENT.md                     # This document
```

## Deployment Steps

### 1. Server Environment Preparation

#### 1.1 Install Python and pip
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### 1.2 Install CUDA (if using NVIDIA GPU)
```bash
# Download and install CUDA Toolkit 11.3+
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 1.3 Create Virtual Environment
```bash
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate
```

### 2. Upload Code Files

#### 2.1 Upload using rsync (recommended)
```bash
# From local machine
rsync -av --progress \
  /Users/jackielyu/AutoSurvey/lhb_survey_generation/ \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/

# Upload specific large files separately
rsync -av --progress \
  /Users/jackielyu/AutoSurvey/lhb_survey_generation/data/raw/original_survey_df.pkl \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/data/raw/
```

#### 2.2 Or use scp to upload core files
```bash
# Upload core experimental script
scp experiments/baselines/run_baselines.py \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/experiments/baselines/

# Upload dataset
scp data/raw/original_survey_df.pkl \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/data/raw/

# Upload QFiD model
scp -r SciReviewGen-main/qfid \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/SciReviewGen-main/

# Upload configuration files
scp requirements.txt .env \
  username@server:/path/to/AutoSurvey/lhb_survey_generation/
```

### 3. Install Dependencies

#### 3.1 Create requirements.txt
Create requirements.txt file on the server:

```bash
cat > requirements.txt << 'EOF'
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
torch>=1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.15.0

# Text processing
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
rouge-score>=0.1.2
nltk>=3.8
scikit-learn>=1.1.0
rank-bm25>=0.2.2

# API and configuration
requests>=2.28.0
python-dotenv>=0.19.0
pyyaml>=6.0
tqdm>=4.64.0
EOF
```

#### 3.2 Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 4. Configure Environment Variables
Create `.env` file:

```bash
cat > .env << 'EOF'
# CUDA settings
CUDA_VISIBLE_DEVICES=0

# Experiment configuration
NUM_SURVEYS=5
MAX_WORKERS=4

# Model cache directory
TRANSFORMERS_CACHE=/path/to/cache
HF_HOME=/path/to/cache
EOF
```

### 5. Run Experiments

#### 5.1 Basic Running
```bash
cd /path/to/AutoSurvey/lhb_survey_generation
source venv/bin/activate

# Run baseline experiments
python experiments/baselines/run_baselines.py
```

#### 5.2 Specify Number of Surveys to Process
```bash
NUM_SURVEYS=3 python experiments/baselines/run_baselines.py
```

#### 5.3 Use Specific GPU
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/baselines/run_baselines.py
```

## Important Fix Notes

### 1. Device Compatibility Fixes
The following issues in the QFiD model have been fixed:
- Hardcoded CUDA device → Automatic device detection (CUDA > MPS > CPU)
- Device mismatch errors → Dynamic device allocation
- MPS compatibility issues → Apple Silicon GPU support

### 2. Performance Optimizations
- Improved generation parameters (max_length=512, num_beams=6)
- Better text formatting and cleaning
- Error handling and logging

### 3. Integration Features
- QFiD fully integrated into baseline system
- Consistent interface with other models (LEAD, LexRank, BigBird, FiD)
- Automatic model download and caching

## Running Configuration

### 1. Environment Variable Settings
```bash
# Set number of surveys to process
export NUM_SURVEYS=5

# Set maximum number of parallel workers
export MAX_WORKERS=4

# Set GPU device
export CUDA_VISIBLE_DEVICES=0
```

### 2. Model Cache Settings
```bash
# Set HuggingFace model cache directory
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_cache
```

### 3. Log Configuration
```bash
# Enable detailed logging
export PYTHONPATH=/path/to/AutoSurvey/lhb_survey_generation:$PYTHONPATH
```

## Monitoring and Debugging

### 1. Monitor GPU Usage
```bash
nvidia-smi -l 1
```

### 2. Monitor Memory Usage
```bash
htop
```

### 3. View Log Files
```bash
tail -f experiments/baselines/baseline.log
```

## Performance Optimization Suggestions

1. **Batch Processing**: Adjust batch size according to GPU memory
2. **Parallel Processing**: Use multi-GPU when available
3. **Model Cache**: Pre-download models to avoid network delays
4. **Data Preprocessing**: Preprocess data to reduce runtime overhead

## Backup and Recovery

### 1. Code Backup
```bash
tar -czf AutoSurvey_backup_$(date +%Y%m%d).tar.gz \
  --exclude=venv --exclude=__pycache__ \
  /path/to/AutoSurvey/lhb_survey_generation/
```

### 2. Result Backup
```bash
rsync -av /path/to/AutoSurvey/lhb_survey_generation/outputs/ \
  backup_server:/backup/AutoSurvey_outputs/
```

## Security Considerations

1. **API Key Management**: Use environment variables, do not hardcode
2. **Network Security**: Ensure server network security
3. **Data Privacy**: Comply with data usage agreements
4. **Access Control**: Set appropriate file permissions

## Contact and Support

If you encounter any issues, please check:
1. Log files for detailed error information
2. GPU memory usage and CUDA version compatibility
3. Network connection and model download status

This deployment guide ensures the stable operation of the AutoSurvey experimental environment in a server environment.