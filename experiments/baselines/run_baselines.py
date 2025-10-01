import json
import os
import re
from pathlib import Path
from typing import Dict, List, Callable

import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Set up HuggingFace mirror endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Import QFiD model
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "SciReviewGen-main" / "qfid"))
from qfid import BartForConditionalGeneration as QFiDModel

# Optional import guard for LexRank baseline
try:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _LEXRANK_AVAILABLE = True
except ImportError:
    _LEXRANK_AVAILABLE = False

# Optional import guard for QFiD baseline
try:
    from transformers import BartTokenizer
    _QFID_AVAILABLE = True
except ImportError:
    _QFID_AVAILABLE = False

NUM_SURVEYS_TO_PROCESS = 30
OUTPUT_FILENAME = "baseline_chapter_outputs.json"


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _resolve_num_surveys() -> int:
    """Resolve how many surveys to process, allowing ENV override."""
    override = os.getenv("NUM_SURVEYS_TO_PROCESS")
    if override is None:
        return NUM_SURVEYS_TO_PROCESS
    try:
        value = int(override)
        return value if value > 0 else NUM_SURVEYS_TO_PROCESS
    except ValueError:
        return NUM_SURVEYS_TO_PROCESS


def _split_sentences(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def run_lead_baseline(documents: List[str], num_sentences: int = 5) -> str:
    """Return the first N sentences after concatenating all documents."""
    combined = " ".join(documents).strip()
    if not combined:
        return ""
    sentences = _split_sentences(combined)
    if not sentences:
        return ""
    return " ".join(sentences[:num_sentences])


def run_lexrank_baseline(documents: List[str], num_sentences: int = 5) -> str:
    """Run LexRank summarization over concatenated documents."""
    if not _LEXRANK_AVAILABLE:
        raise ImportError("sumy is required for the LexRank baseline. Install via `pip install sumy`. ")

    combined = " ".join(documents).strip()
    if not combined:
        return ""

    parser = PlaintextParser.from_string(combined, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


def run_bigbird_baseline(query: str, documents: List[str]) -> str:
    print("INFO: Skipping BigBird generation (placeholder).")
    return "[BigBird output placeholder]"


def run_fid_baseline(query: str, documents: List[str]) -> str:
    print("INFO: Skipping FiD generation (placeholder).")
    return "[FiD output placeholder]"


# Global QFiD model and tokenizer to avoid reloading for each inference
_qfid_model = None
_qfid_tokenizer = None
_qfid_device = None


def _load_qfid_model():
    """Load QFiD model and tokenizer lazily."""
    global _qfid_model, _qfid_tokenizer, _qfid_device

    if _qfid_model is not None:
        return

    if not _QFID_AVAILABLE:
        raise ImportError("transformers is required for the QFiD baseline. Install via `pip install transformers`. ")

    print("Loading QFiD model and tokenizer...")

    # Set device
    if torch.cuda.is_available():
        _qfid_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _qfid_device = torch.device("mps")
    else:
        _qfid_device = torch.device("cpu")
    print(f"Using device: {_qfid_device}")

    # Load standard BART model first, then use QFiD architecture
    model_name = "facebook/bart-large-cnn"

    # Load tokenizer
    _qfid_tokenizer = BartTokenizer.from_pretrained(model_name)

    # Load QFiD model with standard BART weights
    _qfid_model = QFiDModel.from_pretrained(model_name)

    # Move model to device
    _qfid_model.to(_qfid_device)
    _qfid_model.eval()

    print("QFiD model loaded successfully.")


def _format_input_for_qfid(query: str, documents: List[str]) -> str:
    """
    Format input for QFiD model according to SciReviewGen format.
    Format: "survey_title <s> chapter_title <s> abstract1 <s> BIB001 <s> survey_title <s> chapter_title <s> abstract2 <s> BIB002 <s> ..."
    """
    if not documents:
        return ""

    # Split query into title and chapter if possible
    parts = query.split(" - ", 1)
    survey_title = parts[0] if len(parts) > 1 else query
    chapter_title = parts[1] if len(parts) > 1 else ""

    # Format according to QFiD requirements
    # Each document should be formatted as: survey_title <s> chapter_title <s> abstract <s> BIBXXX
    formatted_parts = []
    for i, doc in enumerate(documents[:100]):  # Limit to 100 documents as in original code
        bib_id = f"BIB{i+1:03d}"
        # Clean document text
        doc = doc.strip().replace('\n', ' ').replace('\r', ' ')
        formatted_part = f"{survey_title} <s> {chapter_title} <s> {doc} <s> {bib_id}"
        formatted_parts.append(formatted_part)

    # Join all parts with <s> separator
    return " <s> ".join(formatted_parts)


def run_qfid_baseline(query: str, documents: List[str]) -> str:
    """Run QFiD generation for query and documents."""
    try:
        _load_qfid_model()

        if not documents:
            return ""

        # Format input
        formatted_input = _format_input_for_qfid(query, documents)

        # Tokenize
        inputs = _qfid_tokenizer(
            formatted_input,
            max_length=2048,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(_qfid_device) for k, v in inputs.items()}

        # Generate with better parameters for scientific summarization
        with torch.no_grad():
            outputs = _qfid_model.generate(
                **inputs,
                max_length=512,  # Increased for longer summaries
                num_beams=6,     # Increased beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=4,
                length_penalty=2.5,
                temperature=0.8,
                repetition_penalty=1.2,
                do_sample=False  # Use deterministic generation for consistency
            )

        # Decode
        generated_text = _qfid_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return generated_text.strip()

    except Exception as e:
        print(f"WARN: QFiD generation failed: {e}")
        return ""


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Original dataset not found at {path}. Please prepare the data before running baselines."
        )
    return pd.read_pickle(path)


def _collect_section_documents(survey_row: pd.Series, section_index: int) -> List[str]:
    abstracts_by_section = survey_row.get("bib_abstracts") or []
    documents: List[str] = []
    if isinstance(abstracts_by_section, list) and section_index < len(abstracts_by_section):
        abstracts_dict = abstracts_by_section[section_index] or {}
        if isinstance(abstracts_dict, dict):
            documents = [abstract.strip() for abstract in abstracts_dict.values() if isinstance(abstract, str) and abstract.strip()]
    return documents


def _prepare_baseline_outputs(survey_df: pd.DataFrame, num_surveys: int) -> List[Dict]:
    available = len(survey_df)
    if num_surveys > available:
        raise ValueError(
            f"Requested {num_surveys} surveys but only {available} are available. Please adjust NUM_SURVEYS_TO_PROCESS."
        )

    baseline_functions: Dict[str, Callable[[str, List[str]], str]] = {
        "LEAD": lambda query, docs: run_lead_baseline(docs),
        "LexRank": lambda query, docs: run_lexrank_baseline(docs),
        "BigBird": run_bigbird_baseline,
        "FiD": run_fid_baseline,
        "QFiD": run_qfid_baseline,
    }

    results: List[Dict] = []

    for idx in range(num_surveys):
        survey = survey_df.iloc[idx]
        survey_title = survey.get("title", f"survey_{idx}")
        sections = survey.get("section") or []

        for section_index, chapter_title in enumerate(sections):
            documents = _collect_section_documents(survey, section_index)
            query = f"{survey_title} - {chapter_title}"

            for baseline_name, baseline_fn in baseline_functions.items():
                try:
                    generated_text = baseline_fn(query, documents)
                except Exception as exc:
                    generated_text = ""
                    print(f"WARN: Baseline {baseline_name} failed for {survey_title} / {chapter_title}: {exc}")

                results.append({
                    "survey_title": survey_title,
                    "chapter_title": chapter_title,
                    "generated_text": generated_text,
                    "baseline_model": baseline_name
                })

    return results


def _load_existing_results(output_path: Path) -> List[Dict]:
    if not output_path.exists():
        return []
    try:
        with output_path.open("r") as handle:
            data = json.load(handle)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def main():
    project_root = _project_root()
    data_path = project_root / "data" / "raw" / "original_survey_df.pkl"
    output_dir = project_root / "experiments" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    num_surveys = _resolve_num_surveys()
    survey_df = _load_dataset(data_path)
    new_results = _prepare_baseline_outputs(survey_df, num_surveys)

    existing_results = _load_existing_results(output_path)
    combined_results = existing_results + new_results

    with output_path.open("w") as handle:
        json.dump(combined_results, handle, indent=2)

    print(f"Saved {len(new_results)} new records to {output_path}. Total records: {len(combined_results)}")


if __name__ == "__main__":
    main()
