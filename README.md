# An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG

This repository contains the official implementation for the paper, **"An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG"**.

## Overview

The rapid growth of scientific publications presents a significant challenge for traditional literature survey methods. To address this, we propose an automated framework that synergizes a multi-agent architecture with Retrieval-Augmented Generation (RAG). This approach mimics the collaborative workflow of human research teams by decomposing the survey generation process into specialized, collaborative sub-tasks, ensuring global coherence across long documents. The integrated RAG mechanism grounds the generated content in factual academic literature, mitigating the risk of LLM hallucination and ensuring academic integrity, even when citation information is incomplete.

Our framework provides a reproducible pipeline for generating high-quality academic surveys, evaluating their quality, and comparing the performance of different Large Language Models (LLMs).

## Framework Architecture (Methodology)

Our framework replicates the human expert workflow for composing literature surveys through four core modules:

1.  **Retrieval-Augmented Citation Allocation**: A hybrid RAG strategy is employed to allocate citations. For sections with predefined references, it uses direct matching. For sections without, it performs context-aware semantic search (using FAISS) to retrieve the most relevant papers from a candidate pool.
2.  **Progressive Abstract Generation**: A dedicated `AbstractWriterAgent` generates a concise and informative summary of the entire survey based on the key papers identified for each section.
3.  **Structured Content Generation**: A `SectionWriterAgent` generates the content for each section. It is provided with the full survey outline to maintain contextual awareness, ensuring smooth transitions and thematic consistency across the document.
4.  **Quality Assessment**: The quality of the generated surveys is evaluated using standard ROUGE metrics (ROUGE-1, ROUGE-2, and ROUGE-L) to measure lexical coverage, phrasal accuracy, and semantic coherence against reference texts.

For a detailed explanation of the methodology, please refer to our full paper.

## Key Features

-   **Multi-Agent Architecture**: A modular system featuring specialized agents for `Paper Allocation`, `Abstract Writing`, and `Section Generation` to ensure structured and coherent output.
-   **Retrieval-Augmented Generation (RAG)**: Integrates a hybrid retrieval mechanism that uses direct citation matching and dense vector search (via FAISS) to ground generated content in relevant academic literature.
-   **Configurable Experiments**: Supports various experimental setups, including different LLMs (e.g., GPT-4o-mini, Qwen3-14B) and ablation studies (e.g., without RAG, single-agent).
-   **Quantitative Evaluation**: Includes scripts for assessing the quality of generated surveys using standard ROUGE metrics.

## Repository Structure and Code Description

```
.
├── src/                    # Core source code for the framework
│   ├── agents/             # Implementation of the different agents (allocation, abstract, section)
│   └── data/               # Data processing scripts
├── experiments/            # Scripts to run experiments and evaluations
│   ├── main/               # Main survey generation experiments
│   ├── qwen_ablations/     # Ablation studies for the Qwen model
│   └── evaluations/        # Evaluation script for generated outputs
├── configs/                # System configuration files (e.g., model params, paths)
├── data/                   # Directory for raw data (e.g., original_survey_df.pkl)
├── processed_data/         # Directory for preprocessed data (e.g., FAISS index)
├── outputs/                # Default directory for generated surveys and evaluation reports
├── .env.example            # Template for environment variables (API keys)
├── requirements.txt        # Python package dependencies
├── run_experiments.py      # Main entry point to run all experiments
└── README.md               # This file
```

## Reproducibility Guide

### 1. Requirements

-   **OS**: macOS / Windows 11 / Linux
-   **Python Version**: 3.9+
-   **Hardware**: A multi-core CPU is recommended. A CUDA-enabled GPU is optional but can accelerate embedding processes.

### 2. Setup Environment

First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/may3rr/survey_generation.git
cd survey_generation

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download required NLTK data for sentence splitting
python -c "import nltk; nltk.download('punkt')"
```

### 3. Dataset and Preprocessing

#### Dataset Information and License
This work is built upon the **SciReviewGen** dataset. For full details, please refer to the official dataset repository and paper:
-   **Repository**: [https://github.com/tetsu9923/SciReviewGen](https://github.com/tetsu9923/SciReviewGen)
-   **License**: The dataset is released under the CC BY-NC 4.0 license, which restricts its use to non-commercial purposes.

#### Data Download
The raw and preprocessed data files must be downloaded separately. Our experiments use the `original_survey_df` version.

-   **Raw Data (`original_survey_df.pkl`)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1J83Ku6Qe63Tu6iDX9ek-G5SLxceDTXIC/view?usp=drive_link)
    -   Place the file in the `data/raw/` directory.

-   **Processed Data (FAISS index, etc.)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1zzKO5cS7YAD4tjZ6K7QxhrYfqydpuopX/view?usp=drive_link)
    -   Unzip the archive and place its contents into the `processed_data/` directory.

#### Preprocessing
The `processed_data/` directory contains files generated from the raw data. The key preprocessing step was the creation of a semantic search index for the RAG module. We generated vector embeddings for the abstracts of all cited papers using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings were then used to build a `FAISS` index (`abstract_index.faiss`), enabling efficient semantic retrieval.

### 4. Configuration

Create a `.env` file from the provided template to store your API keys.

```bash
# Create the .env file from the example template
cp .env.example .env

# Edit the file to add your API keys for OpenAI (GPT) and/or Alibaba (Qwen)
nano .env  # or use your favorite text editor
```

### 5. Usage Instructions (Running Experiments)

All scripts are designed to be executed from the project's root directory via the `run_experiments.py` script. Generated surveys will be saved in the `outputs/` directory.

#### Main Experiments
-   **To generate surveys with Qwen3-14B:**
    ```bash
    python run_experiments.py qwen_main
    ```
-   **To generate surveys with GPT-4o-mini:**
    ```bash
    python run_experiments.py gpt_main
    ```

#### Ablation Studies
-   **To run the experiment without RAG:**
    ```bash
    python run_experiments.py qwen_no_rag
    ```
-   **To run the experiment with a single-agent architecture:**
    ```bash
    python run_experiments.py qwen_single_agent
    ```

### 6. Evaluation

To evaluate the generated outputs (e.g., from the Qwen experiment) against the reference texts using ROUGE scores:

```bash
python run_experiments.py evaluate_qwen
```
The evaluation results will be printed to the console, and detailed JSON reports will be saved in the `outputs/` directory.

## Citations

If you use this work, please cite our paper (details to be added upon publication).

Please also consider citing the original **SciReviewGen** dataset paper:

[SciReviewGen: A Large-scale Dataset for Automatic Literature Review Generation](https://aclanthology.org/2023.findings-acl.418/) (Kasanishi et al., Findings 2023)
## License

The code in this repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details. Note that the underlying dataset has its own license (CC BY-NC 4.0).

## Contribution Guidelines

We welcome contributions and suggestions! If you find a bug or have an idea for an improvement, please open an issue on the GitHub repository.