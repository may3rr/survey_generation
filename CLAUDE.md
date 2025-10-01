# CLAUDE.md - AutoSurvey LHB Survey Generation Repository

## Repository Overview

This is a repository for an LLM-driven scientific literature survey generation framework based on multi-agents and RAG technology. The project implements an automated framework that combines multi-agent architecture with Retrieval-Augmented Generation (RAG) technology to simulate the collaborative workflow of human research teams in decomposing the survey generation process.

## Core Architecture

### 1. Multi-Agent Architecture
- **PaperAllocationAgent**: Responsible for allocating relevant literature to each section
- **AbstractWriterAgent**: Generates the abstract section of literature surveys
- **SectionWriterAgent**: Generates specific content for each section

### 2. RAG Retrieval System
- Hybrid Retrieval Strategy: Combines vectorized retrieval and BM25 keyword retrieval
- Reranking Mechanism: Uses CrossEncoder for result reranking
- FAISS Vector Index: Efficient semantic similarity search

## Detailed Module Description

### `/src` Core Source Code Directory

#### `/src/agents` Agent Implementation
- **`paper_allocation.py`**: Literature allocation agent
  - Implements hybrid search and reranking pipeline
  - Supports fusion of semantic search and BM25 search
  - Contains deduplication and candidate literature filtering logic
  - Key methods: `allocate()`, `_search_bm25_in_pool()`, `_merge_and_deduplicate()`

- **`abstract_writer.py`**: Abstract generation agent
  - Generates academic abstracts based on allocated literature results
  - Supports configurable token count and prompt templates
  - Key methods: `generate()`, `_create_abstract_prompt()`

- **`section_writer.py`**: Section content generation agent
  - Parallel generation of multiple section contents
  - Supports custom prompt templates and citation formats
  - Contains content formatting and reference management
  - Key methods: `generate()`, `generate_multiple_sections()`, `_format_references()`

- **`prompt_writer.py`**: Prompt generator
  - Manages prompt templates for various generation tasks

#### `/src/data` Data Processing Module
- **`data_processor.py`**: Core data processing class
  - Supports vector storage creation and management (FAISS)
  - Implements BM25 index construction
  - Provides semantic search and keyword search functionality
  - Contains CrossEncoder reranking
  - Key methods: `create_vector_store()`, `search_similar_abstracts()`, `search_in_papers()`, `rerank_papers()`

#### `/src/utils` Utility Module
- **`config_loader.py`**: Configuration loader
  - Supports YAML configuration files and environment variables
  - Provides secure API key management
  - Contains configuration validation functionality
  - Key methods: `load_config()`, `validate_config()`, `get_api_config()`

- **`api_client.py`**: API client
  - Unified LLM API call interface
  - Supports retry mechanism and usage statistics
  - Compatible with OpenAI and Qwen API formats
  - Key methods: `generate_text()`

### `/experiments` Experiment Scripts Directory

#### `/experiments/main` Main Experiments
- **`survey_generation_gpt.py`**: Generate surveys using GPT-4o-mini
- **`survey_generation_qwen.py`**: Generate surveys using Qwen3-14B

#### `/experiments/qwen_ablations` Ablation Experiments
- **`qwen_ablation_no_rag.py`**: Ablation experiment without RAG
- **`qwen_ablation_single_agent.py`**: Single-agent architecture ablation experiment

#### `/experiments/evaluations` Evaluation Module
- **`evaluate_qwen_outputs.py`**: Evaluate generation results using ROUGE metrics

#### `/experiments/baselines` Baseline Models
- **`run_baselines.py`**: Run baseline model comparisons

### `/SciReviewGen-main` QFiD Model Support
- **`qfid/qfid.py`**: QFiD (Query-Focused Input Duplication) model implementation
- **`qfid/run_summarization.py`**: Run QFiD summarization generation
- Supports query-based document summarization tasks

### Configuration and Data

#### `/configs` Configuration Files
- System configuration files (YAML format)
- Model parameters and path settings

#### `/data` Data Directory
- Raw data storage location
- Need to download `original_survey_df.pkl` file

#### `/processed_data` Processed Data
- FAISS vector index files
- Preprocessed literature data
- BM25 index files

#### `/outputs` Output Directory
- Generated literature survey results
- Evaluation reports and log files

## Main Running Scripts

### `run_experiments.py` Experiment Runner
Unified experiment entry point, supporting the following experiment types:
- `gpt_main`: GPT-4o-mini main experiment
- `qwen_main`: Qwen3-14B main experiment  
- `qwen_no_rag`: No RAG ablation experiment
- `qwen_single_agent`: Single agent ablation experiment
- `evaluate_qwen`: Evaluate Qwen generation results

Usage:
```bash
python run_experiments.py <experiment_name> --num-surveys <number>
```

### `run_hybrid_experiment.py` Hybrid Experiment
Dedicated experiment script for running hybrid retrieval and reranking pipeline

### `survey_generation_gpt.py` GPT Generation Script
Independent GPT literature survey generation script

## Environment Configuration

### Dependencies (requirements.txt)
Main dependencies include:
- `pandas>=1.5.0`: Data processing
- `sentence-transformers>=2.2.0`: Text embedding
- `faiss-cpu>=1.7.0`: Vector indexing
- `rouge-score>=0.1.2`: Evaluation metrics
- `rank-bm25>=0.2.2`: BM25 retrieval
- `pyyaml>=6.0`: Configuration file parsing
- `python-dotenv>=0.19.0`: Environment variable management

### Environment Variables (.env)
Need to configure the following API keys:
- `GPT_API_KEY`: OpenAI API key
- `GPT_BASE_URL`: OpenAI API base URL
- `GPT_MODEL`: GPT model name
- `QWEN_API_KEY`: Qwen API key  
- `QWEN_BASE_URL`: Qwen API base URL
- `QWEN_MODEL`: Qwen model name

## Data Requirements

### Raw Data
- Need to download `original_survey_df.pkl` from Google Drive
- Based on SciReviewGen dataset (CC BY-NC 4.0 license)

### Processed Data
- Preprocessed data package containing FAISS index files
- Vector embeddings and retrieval indexes

## Core Features

1. **Hybrid Retrieval Pipeline**: Combines vectorized semantic search and BM25 keyword search
2. **Intelligent Reranking**: Uses CrossEncoder to rerank retrieval results
3. **Multi-Agent Collaboration**: Different agents handle different generation tasks
4. **Parallel Processing**: Supports parallel generation of multiple sections
5. **Flexible Configuration**: Supports multiple LLM models and parameter adjustment
6. **Evaluation Mechanism**: Built-in ROUGE metrics evaluation
7. **Error Handling**: Comprehensive logging and exception handling

## Development and Extension

### Adding New Agents
1. Create new agent class in `/src/agents` directory
2. Inherit base agent interface
3. Implement core methods like `generate()`

### Supporting New LLMs
1. Add new API client in `api_client.py`
2. Add configuration mapping in `config_loader.py`
3. Create corresponding experiment scripts

### Extending Retrieval Strategies
1. Add new retrieval methods in `data_processor.py`
2. Update retrieval pipeline in `paper_allocation.py`
3. Add corresponding configuration options

## Important Notes

1. **Data License**: Ensure compliance with SciReviewGen dataset's CC BY-NC 4.0 license
2. **API Key Security**: Do not commit API keys to version control systems
3. **Resource Consumption**: Vector indexing and large language model calls require significant computational resources
4. **Network Dependency**: Requires stable network connection to access LLM APIs

## Troubleshooting

### Common Issues
1. **API Call Failures**: Check API key configuration and network connection
2. **Data Loading Errors**: Confirm data file paths and formats are correct
3. **Vector Index Errors**: Check if preprocessed data is complete
4. **Insufficient Memory**: Consider using smaller embedding models or reducing data volume

### Log Files
- `data_processor.log`: Data processing logs
- `section_writer.log`: Section generation logs
- `hybrid_experiment.log`: Hybrid experiment logs

This framework provides a complete, scalable solution for automated scientific literature survey generation, supporting multiple experimental settings and evaluation methods.