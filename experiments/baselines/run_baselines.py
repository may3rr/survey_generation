import json
import os
import re
from pathlib import Path
from typing import Dict, List, Callable

import pandas as pd

# Optional import guard for LexRank baseline
try:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _LEXRANK_AVAILABLE = True
except ImportError:
    _LEXRANK_AVAILABLE = False

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
