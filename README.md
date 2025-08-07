# Survey Generation System

A comprehensive system for generating academic survey papers using Large Language Models with different experimental configurations.

## Project Structure

```
lhb_survey_generation/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   ├── data/                     # Data processing modules
│   └── utils/                    # Utility functions
├── experiments/                  # Experimental scripts
│   ├── main/                     # Main experimental scripts
│   ├── qwen_ablations/           # Ablation studies
│   └── evaluations/              # Evaluation scripts
├── configs/                      # Configuration files
├── data/                         # Data storage
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
├── outputs/                      # Generated outputs
├── tests/                        # Test scripts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your actual API keys
nano .env
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Quick Start(example Qwen)

### 1. Main Survey Generation
```bash
cd experiments/main
python survey_generation_qwen.py
```

### 2. Qwen3-14B Survey Generation
```bash
cd experiments/main
python survey_generation_qwen.py
```

### 3. Ablation Studies

**Without RAG:**
```bash
cd experiments/qwen_ablations
python qwen_ablation_no_rag.py
```

**Single Agent (No RAG + No MultiAgent):**
```bash
cd experiments/qwen_ablations
python qwen_ablation_single_agent.py
```

### 4. Evaluation
```bash
cd experiments/evaluations
python evaluate_qwen_outputs.py
```

## Data Requirements

- Place your raw survey data at: `data/raw/original_survey_df.pkl`
- Processed data will be automatically generated in: `data/processed/`

## Configuration

Edit `configs/config.yaml` to customize:
- API keys and endpoints
- Model parameters
- Generation settings
- Evaluation metrics

## Output Structure

Generated surveys are saved in:
- `outputs/survey_{i}_generated.json` - Complete survey
- `outputs/survey_{i}_allocations.json` - Paper allocations
- `outputs/survey_{i}_evaluation.json` - Evaluation results

## Key Features

1. **Multi-Agent Architecture**: Separate agents for paper allocation, abstract writing, and section generation
2. **Vector Search**: Semantic search for relevant papers using Sentence Transformers
3. **Context-Aware Generation**: Sections consider the overall survey structure
4. **Comprehensive Evaluation**: ROUGE-based evaluation metrics
5. **Configurable Pipeline**: Easy to modify models, parameters, and settings