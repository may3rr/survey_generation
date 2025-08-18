# An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG

This repository contains the official implementation for the paper, *"An LLM-driven Scientific Literature Surveys Generation Framework Based on Multi-Agents and RAG"*.

Our framework automates the generation of academic literature surveys by synergizing a multi-agent architecture with Retrieval-Augmented Generation (RAG). It provides a reproducible pipeline for generating surveys, evaluating their quality, and comparing the performance of different Large Language Models (LLMs).

## Key Features

-   **Multi-Agent Architecture**: A modular system featuring specialized agents for `Paper Allocation`, `Abstract Writing`, and `Section Generation` to ensure structured and coherent output.
-   **Retrieval-Augmented Generation (RAG)**: Integrates a hybrid retrieval mechanism that uses direct citation matching and dense vector search (via FAISS) to ground generated content in relevant academic literature.
-   **Configurable Experiments**: Supports various experimental setups, including different LLMs (e.g., GPT-4o-mini, Qwen3-14B) and ablation studies.
-   **Quantitative Evaluation**: Includes scripts for assessing the quality of generated surveys using standard ROUGE metrics.

## Reproducibility Guide

### 1. Requirements

-   **OS**: macOS / Windows 11 / Linux
-   **Python Version**: 3.9+
-   **Hardware**: A multi-core CPU is recommended. A CUDA-enabled GPU is optional but can accelerate certain embedding processes if `faiss-gpu` is used.

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

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```
*Note: The versions specified in `requirements.txt` are the minimum required versions. The project has been tested with the versions listed in the file.*

### 3. Dataset and Preprocessing

#### Dataset Information and License
This work is built upon the **SciReviewGen** dataset. For full details, please refer to the official dataset repository: [**https://github.com/tetsu9923/SciReviewGen**](https://github.com/tetsu9923/SciReviewGen).


#### Data Download
Due to their large size, the raw and preprocessed data files used in our experiments must be downloaded separately. Our experiments use the `original_survey_df` version of the dataset.

-   **Raw Data (`original_survey_df.pkl`)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1J83Ku6Qe63Tu6iDX9ek-G5SLxceDTXIC/view?usp=drive_link)
    -   Place the file in the `data/raw/` directory.

-   **Processed Data (FAISS index, etc.)**:
    -   Download from: [Google Drive](https://drive.google.com/file/d/1zzKO5cS7YAD4tjZ6K7QxhrYfqydpuopX/view?usp=drive_link)
    -   Unzip the archive and place its contents into the `processed_data/` directory.

#### Data Processing Methodology
The `processed_data/` directory contains files generated from the raw data. The key preprocessing step was the creation of a semantic search index for the RAG module. We generated vector embeddings for the abstracts of all cited papers using the **`sentence-transformers/all-MiniLM-L6-v2`** model. These embeddings were then used to build a `FAISS` index (`abstract_index.faiss`), enabling efficient semantic retrieval.

### 4. Configure API Keys

Create a `.env` file from the provided template to store your API credentials for services like OpenAI or Alibaba Cloud.

```bash
# Create the .env file
cp .env.example .env

# Edit the file to add your API keys
nano .env  # or use your favorite text editor
```

#### Using OpenRouter as API Provider

If you want to use OpenRouter instead of the default API providers, you can easily switch by modifying the configuration:

**Step 1: Update your `.env` file**

Replace the API configuration in your `.env` file:

```bash
# OpenRouter API Configuration
GPT_API_KEY=your_openrouter_api_key_here
GPT_BASE_URL=https://openrouter.ai/api/v1
GPT_MODEL=qwen/qwen3-14b

# You can also use other models available on OpenRouter
# GPT_MODEL=anthropic/claude-3-haiku
# GPT_MODEL=openai/gpt-4o-mini
# GPT_MODEL=meta-llama/llama-3.1-8b-instruct
```

**Step 2: Update the configuration file (Optional)**

You can also modify `configs/config.yaml` to set OpenRouter as default:

```yaml
api:
  gpt:
    base_url: "https://openrouter.ai/api/v1"
    model: "qwen/qwen3-14b"  # or any other model from OpenRouter
    max_tokens: 1500
    temperature: 0.7
    max_retries: 3
    retry_delay: 1.0
```



**Available Models on OpenRouter:**
- `qwen/qwen3-14b` - Qwen 3 14B model
- `openai/gpt-4o-mini` - OpenAI GPT-4o Mini
- And many more available at [OpenRouter Models](https://openrouter.ai/models)

### 5. Run Experiments

All scripts are designed to be executed from the project's root directory. The generated survey texts will be saved in the `outputs/` directory.

#### **Main Experiments**
-   **To generate surveys with Qwen3-14B:**
    ```bash
    python run_experiments.py qwen_main
    ```
-   **To generate surveys with GPT-4o-mini:**
    ```bash
    python run_experiments.py gpt_main
    ```

#### **Evaluation**
To evaluate the generated outputs (e.g., from the Qwen experiment) against the reference texts using ROUGE scores:
```bash
python run_experiments.py evaluate_qwen
```
The evaluation results will be printed to the console and saved in the `outputs/evaluations/` directory.

### Repository Structure

```
.
├── src/                    # Core source code (agents, data handlers, utils)
├── experiments/            # Experiment-running scripts
├── configs/                # System configuration files
├── data/                   # For raw data (ignored by Git)
├── processed_data/         # For preprocessed data like FAISS index
├── outputs/                # For generated outputs and evaluations (ignored by Git)
├── .env.example            # Environment variable template
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```



### License

The code in this repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details.

Note that the underlying dataset (**SciReviewGen**) used by this project has its own license (CC BY-NC 4.0), which restricts its use to non-commercial purposes only.