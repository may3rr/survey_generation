import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import json
import pandas as pd
from rouge_score import rouge_scorer
from typing import Dict, List, Optional
import logging
import numpy as np
class SurveyEvaluator:
    """Survey evaluation class"""
    
    def __init__(self):
        self._setup_logger()
        # Use the same ROUGE configuration as in the paper
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.metrics = ['rouge1', 'rouge2', 'rougeL']
        self.score_types = ['precision', 'recall', 'fmeasure']

    def _setup_logger(self):
        """Setup logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_generated_survey(self, file_path: str) -> Optional[Dict]:
        """Load generated survey"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading survey file {file_path}: {str(e)}")
            return None

    def postprocess_text(self, text: str) -> str:
        """Post-process text, clean markdown syntax and extra line breaks"""
        # Clean text
        text = text.strip()
        
        # Clean markdown markers
        # Remove title markers starting with #
        text = ' '.join(line.lstrip('#').strip() for line in text.split('\n'))
        
        # Clean consecutive line breaks and spaces
        text = ' '.join(text.split())
        
        # Use simple rules for sentence splitting
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return "\n".join(sentences)

    def evaluate_survey(self, generated_survey: Dict, original_df: pd.DataFrame) -> Dict:
        """Evaluate generated survey"""
        try:
            # Find original survey
            original_survey = original_df[original_df['title'] == generated_survey['title']]
            if len(original_survey) == 0:
                self.logger.error(f"Original survey not found: {generated_survey['title']}")
                return {}
            original_survey = original_survey.iloc[0]
            
            # Merge all section texts into one complete text
            generated_full_text = ""
            original_full_text = ""
            
            # Collect texts from all sections
            for section_title, generated_content in generated_survey['sections'].items():
                if section_title in original_survey['section']:
                    section_index = original_survey['section'].index(section_title)
                    if section_index < len(original_survey['text']):
                        # Add separator to maintain section boundaries
                        generated_full_text += generated_content + "\n\n"
                        original_full_text += original_survey['text'][section_index] + "\n\n"
            
            # Text post-processing
            generated_full_text = self.postprocess_text(generated_full_text)
            original_full_text = self.postprocess_text(original_full_text)
            
            # Calculate overall ROUGE scores
            try:
                scores = self.scorer.score(original_full_text, generated_full_text)
                return {
                    'overall': {
                        metric: {
                            'precision': scores[metric].precision,
                            'recall': scores[metric].recall,
                            'fmeasure': scores[metric].fmeasure
                        }
                        for metric in self.metrics
                    }
                }
            except Exception as e:
                self.logger.error(f"Error calculating ROUGE scores: {str(e)}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error evaluating survey: {str(e)}")
            return {}

    def calculate_statistics(self, all_scores: List[Dict]) -> Dict:
        """Calculate mean and standard deviation for ROUGE f-measure scores."""
        try:
            stats: Dict[str, Dict[str, float]] = {}
            for metric in self.metrics:
                values = []
                for doc_scores in all_scores:
                    overall = doc_scores.get('overall', {})
                    metric_scores = overall.get(metric, {})
                    fmeasure = metric_scores.get('fmeasure')
                    if fmeasure is not None:
                        values.append(fmeasure)

                if values:
                    arr = np.array(values, dtype=float)
                    stats[metric] = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr))
                    }
                else:
                    stats[metric] = {'mean': 0.0, 'std': 0.0}

            if not all_scores:
                self.logger.warning("No scores available for statistics calculation")

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return {metric: {'mean': 0.0, 'std': 0.0} for metric in self.metrics}

    def print_overall_scores(self, statistics: Dict):
        """Print overall statistics in mean ± std format."""
        print("\nOverall ROUGE Scores:")
        for metric, metric_stats in statistics.items():
            mean = metric_stats.get('mean', 0.0)
            std = metric_stats.get('std', 0.0)
            print(f"{metric.upper()} (fmeasure): {mean:.4f} ± {std:.4f}")

def main():
    try:
        evaluator = SurveyEvaluator()
        project_root = Path(__file__).parent.parent.parent
        
        # Use paths from config file
        config_path = project_root / 'configs/config.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            raw_data_path = config.get('data', {}).get('raw_data_path', 'data/raw/original_survey_df.pkl')
            output_dir = config.get('output', {}).get('directory', 'qwen3_output')
        else:
            raw_data_path = 'data/raw/original_survey_df.pkl'
            output_dir = 'outputtest'
        
        # Load original data
        raw_data_full_path = project_root / raw_data_path
        if not raw_data_full_path.exists():
            print(f"Original data file does not exist: {raw_data_full_path}")
            print("Please run data preparation script first: python prepare_data.py")
            return
            
        original_df = pd.read_pickle(raw_data_full_path)
        
        # Check output directory
        output_full_path = project_root / output_dir
        generated_files = list(output_full_path.glob('*generated.json'))
        
        if not generated_files:
            print(f"No generated survey files found in: {output_full_path}")
            print("Please run experiment to generate surveys first")
            return
            
        print(f"Found {len(generated_files)} generated survey files")
        
        # Evaluate all generated surveys
        all_scores = []
        for input_path in generated_files:
            print(f"Evaluating: {input_path.name}")
            generated_survey = evaluator.load_generated_survey(str(input_path))
            if generated_survey:
                scores = evaluator.evaluate_survey(generated_survey, original_df)
                if scores:
                    all_scores.append(scores)
                    # Save individual evaluation results for each generated survey
                    individual_output_path = output_full_path / f"{input_path.stem}_evaluation.json"
                    with open(individual_output_path, 'w') as f:
                        json.dump({
                            'title': generated_survey['title'],
                            'scores': scores
                        }, f, indent=2)
        
        if all_scores:
            statistics = evaluator.calculate_statistics(all_scores)
            evaluator.print_overall_scores(statistics)
        else:
            print("No survey files were successfully evaluated")
            statistics = evaluator.calculate_statistics([])

        # Save evaluation results
        output_path = output_full_path / 'overall_evaluation.json'
        with open(output_path, 'w') as f:
            json.dump({
                'overall_statistics': statistics,
                'individual_scores': all_scores
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error in evaluation process: {str(e)}")

if __name__ == "__main__":
    main()
