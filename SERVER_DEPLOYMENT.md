# 服务器部署指南

## 概述
本文档描述了将 AutoSurvey 实验环境从本地 M3 Pro MacBook Pro 迁移到服务器的详细步骤。QFiD 模型已成功集成到 `run_baselines.py` 中，并修复了设备兼容性问题。

## 系统要求

### 服务器配置建议
- **GPU**: NVIDIA GPU (推荐 A100/V100/RTX 4090) 至少 16GB 显存
- **CPU**: 8 核心以上
- **内存**: 32GB 以上 (推荐 64GB)
- **存储**: 100GB 以上可用空间
- **操作系统**: Linux (Ubuntu 20.04+ 推荐)

### 软件依赖
- Python 3.8+
- PyTorch 1.12+ (CUDA 版本)
- CUDA 11.3+
- Git

## 需要上传的文件和目录

### 核心代码文件
```
AutoSurvey/lhb_survey_generation/
├── experiments/
│   └── baselines/
│       └── run_baselines.py                 # 主要实验脚本（已集成 QFiD）
├── data/
│   └── raw/
│       └── original_survey_df.pkl           # 原始数据集
├── SciReviewGen-main/                       # QFiD 模型代码
│   └── qfid/
│       ├── __init__.py
│       ├── qfid.py                          # QFiD 模型实现（已修复设备兼容性）
│       └── run_summarization.py             # QFiD 训练脚本
├── requirements.txt                         # Python 依赖
└── SERVER_DEPLOYMENT.md                     # 本文档
```

## 部署步骤

### 1. 服务器环境准备

#### 1.1 安装 Python 和 pip
```bash
sudo apt update
sudo apt install python3.8 python3.8-pip python3.8-venv
```

#### 1.2 安装 CUDA (如果使用 NVIDIA GPU)
```bash
# 下载并安装 CUDA Toolkit 11.3+
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run
```

#### 1.3 创建虚拟环境
```bash
python3.8 -m venv venv
source venv/bin/activate
```

### 2. 上传代码文件

#### 2.1 使用 rsync 上传（推荐）
```bash
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  /Users/jackielyu/AutoSurvey/lhb_survey_generation/ \
  user@server:/path/to/AutoSurvey/lhb_survey_generation/
```

#### 2.2 或者使用 scp 上传核心文件
```bash
scp -r /Users/jackielyu/AutoSurvey/lhb_survey_generation/experiments \
  user@server:/path/to/AutoSurvey/lhb_survey_generation/

scp -r /Users/jackielyu/AutoSurvey/lhb_survey_generation/data \
  user@server:/path/to/AutoSurvey/lhb_survey_generation/

scp -r /Users/jackielyu/AutoSurvey/lhb_survey_generation/SciReviewGen-main \
  user@server:/path/to/AutoSurvey/lhb_survey_generation/

scp /Users/jackielyu/AutoSurvey/lhb_survey_generation/SERVER_DEPLOYMENT.md \
  user@server:/path/to/AutoSurvey/lhb_survey_generation/
```

### 3. 安装依赖

#### 3.1 创建 requirements.txt
在服务器上创建 requirements.txt 文件：
```txt
torch>=1.12.0+cu113
transformers>=4.21.0
datasets>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
rouge-score>=0.1.2
sumy>=0.8.1
accelerate>=0.20.0
sentencepiece>=0.1.96
protobuf>=3.20.0
nltk>=3.7
tqdm>=4.64.0
```

#### 3.2 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：
```bash
# CUDA 设置
CUDA_VISIBLE_DEVICES=0

# 实验配置
NUM_SURVEYS_TO_PROCESS=30
PYTHONPATH=/path/to/AutoSurvey/lhb_survey_generation:$PYTHONPATH

# 模型缓存目录
TRANSFORMERS_CACHE=/path/to/model_cache
HF_HOME=/path/to/huggingface_cache
```

### 5. 运行实验

#### 5.1 基本运行
```bash
cd /path/to/AutoSurvey/lhb_survey_generation
python experiments/baselines/run_baselines.py
```

#### 5.2 指定处理的调研数量
```bash
NUM_SURVEYS_TO_PROCESS=50 python experiments/baselines/run_baselines.py
```

#### 5.3 使用特定 GPU
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/baselines/run_baselines.py
```

## 重要修复说明

### 1. 设备兼容性修复
已修复 QFiD 模型中的以下问题：
- 硬编码 CUDA 设备 → 自动检测设备 (CUDA > MPS > CPU)
- 设备不匹配错误 → 动态设备分配
- MPS 兼容性问题 → 支持 Apple Silicon GPU

### 2. 性能优化
- 改进的生成参数 (max_length=512, num_beams=6)
- 更好的文本格式化和清理
- 错误处理和日志记录

### 3. 集成特性
- QFiD 已完全集成到 baseline 系统中
- 与其他模型 (LEAD, LexRank, BigBird, FiD) 一致的接口
- 自动模型下载和缓存

## 运行配置

### 1. 环境变量设置
```bash
# 设置处理的调查数量（可选，默认为30）
export NUM_SURVEYS_TO_PROCESS=30

# 设置 CUDA 设备（如果使用 GPU）
export CUDA_VISIBLE_DEVICES=0
```

### 2. 运行基准测试
```bash
cd experiments/baselines
python run_baselines.py
```

### 3. 预期输出
- 生成文件：`baseline_chapter_outputs.json`
- 包含所有基准线的生成结果：LEAD, LexRank, BigBird, FiD, QFiD
- QFiD 模型会自动下载并缓存到 `~/.cache/huggingface/`

## 常见问题和解决方案

### 1. 内存不足
如果遇到内存不足错误：
- 减少 `NUM_SURVEYS_TO_PROCESS` 环境变量
- 使用更小的批处理大小

### 2. GPU 相关问题
如果遇到 CUDA 错误：
- 检查 CUDA 版本兼容性
- 确保 PyTorch 安装了正确的 CUDA 版本
- 可以设置 `CUDA_VISIBLE_DEVICES=""` 强制使用 CPU

### 3. QFiD 模型问题
如果 QFiD 加载失败：
- 检查网络连接（需要下载预训练模型）
- 确保有足够的磁盘空间缓存模型（约 2GB）
- 检查 `SciReviewGen-main/qfid/` 文件夹是否完整

### 4. 依赖包冲突
如果遇到依赖包冲突：
- 建议使用新的虚拟环境
- 按照文档中的顺序安装依赖包
- 检查 Python 版本兼容性

## 性能优化建议

### 1. GPU 优化
- 使用多 GPU（设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`）
- 调整批处理大小
- 使用混合精度训练（如果支持）

### 2. 内存优化
- 定期清理 Python 缓存
- 监控内存使用情况
- 考虑使用数据流式处理

### 3. 存储优化
- 确保有足够的磁盘空间存储模型和结果
- 考虑使用 SSD 提高I/O性能

## 监控和日志

### 1. 日志输出
脚本会输出详细的运行日志，包括：
- 模型加载状态
- 处理进度
- 错误和警告信息

### 2. 结果验证
运行完成后，检查 `baseline_chapter_outputs.json`：
- 确认所有基准线都有结果
- 检查结果格式是否正确
- 验证处理数量是否符合预期

## 联系支持

如果遇到部署问题：
1. 检查本文档的常见问题部分
2. 查看错误日志获取详细信息
3. 确认服务器配置满足最低要求