# An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG

This repository contains the official implementation of the framework proposed in the paper, *"An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG"*.

Our system automates the generation of academic survey papers using a multi-agent architecture combined with Retrieval-Augmented Generation (RAG). It supports multiple LLMs and experimental configurations for comprehensive analysis.

## Key Features

-   **Multi-Agent Architecture**: A modular system featuring specialized agents for `Paper Allocation`, `Abstract Writing`, and `Section Generation` to ensure a structured and coherent output.
-   **Retrieval-Augmented Generation (RAG)**: Integrates a hybrid retrieval mechanism that uses direct citation matching and dense vector search (via FAISS) to ground the generated content in relevant academic literature.
-   **Configurable Experiments**: Supports various experimental setups, including different LLMs (GPT-4o-mini, Qwen3-14B) and ablation studies to evaluate the framework's components.
-   **Quantitative Evaluation**: Includes scripts for assessing the quality of generated surveys using standard ROUGE metrics.

## Quick Start Guide

This guide provides the necessary steps to set up the environment and reproduce the experiments described in the paper.

### 1. Setup Environment

First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/may3rr/survey_generation.git
cd survey_generation

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Download Datasets

The raw and processed datasets are necessary but not included in this repository due to their size.

-   **Raw Data (`original_survey_df.pkl`)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1J83Ku6Qe63Tu6iDX9ek-G5SLxceDTXIC/view?usp=drive_link)
    -   Place the file in the `data/raw/` directory.

-   **Processed Data (FAISS index, etc.)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1zzKO5cS7YAD4tjZ6K7QxhrYfqydpuopX/view?usp=drive_link)
    -   Unzip the archive and place its contents into the `.processed_data/` directory.

After this step, your `data/` directory should have the following structure:
```
processed_data/
├──abstract_index.faiss
├──bib_abstracts.json
├──... (other processed files)
data/
├── raw/
│   └── original_survey_df.pkl
```

### 3. Configure API Keys

Create a `.env` file from the provided template to store your API credentials.

```bash
# Create the .env file
cp .env.example .env

# Edit the file to add your API keys
nano .env
```

### 4. Run Experiments

You can now run any of the experiments. All scripts are designed to be executed from the project's root directory.

#### **Main Experiments**

The main experiments use the full multi-agent and RAG framework.

-   **To generate surveys with Qwen3-14B:**
    ```bash
    python run_experiments.py qwen_main
    ```
-   **To generate surveys with GPT-4o-mini:**
    ```bash
    python run_experiments.py gpt_main
    ```

#### **Evaluation**

To evaluate the generated outputs (e.g., from Qwen) against the reference texts using ROUGE scores:
```bash
python run_experiments.py evaluate_qwen
```

## Repository Structure

```
.
├── src/
│   ├── agents/                   # Core agent implementations
│   ├── data/                     # Data processing modules
│   └── utils/                    # Utility functions (API client, config loader)
├── experiments/
│   ├── main/                     # Scripts for main experiments
│   ├── qwen_ablations/           # Scripts for ablation studies
│   └── evaluations/              # Evaluation script
├── configs/                      # System configuration files
├── data/                         # Data storage (ignored by Git)
├── processed_data/
│   ├── abstract_index.faiss
│   ├── bib_abstracts.json
│   ├── ... (other processed files)
├── outputs/                      # Generated outputs (ignored by Git)
├── .env.example                  # Environment variable template
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

## Acknowledgements

This work utilizes the **SciReviewGen** dataset. We thank the authors for making this resource publicly available.

> Kasanishi, T., Isonuma, M., Mori, J., & Sakata, I. (2023). SciReviewGen: A Large-scale Dataset for Automatic Literature Review Generation. In *Findings of the Association for Computational Linguistics: ACL 2023*.