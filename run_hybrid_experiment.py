#!/usr/bin/env python3
"""
Hybrid Search Experiment Runner
Run survey generation with the new hybrid search and re-ranking pipeline for first 30 surveys
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time
from rouge_score import rouge_scorer

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.data_processor import DataProcessor
from src.agents.paper_allocation import PaperAllocationAgent

def load_simple_config():
    """Load basic configuration"""
    return {
        'data': {
            'raw_data_path': './data/raw/original_survey_df.pkl',
            'processed_data_dir': './data/processed'
        }
    }

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hybrid_experiment.log')
        ]
    )
    return logging.getLogger(__name__)

def load_and_prepare_data(data_path: str, limit: int = 30) -> pd.DataFrame:
    """Load and prepare survey data"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading survey data from {data_path}")
    df = pd.read_pickle(data_path)
    
    # Take only first N surveys
    df_limited = df.head(limit)
    logger.info(f"Limited to first {len(df_limited)} surveys")
    
    return df_limited

def setup_data_processor(processed_data_dir: str) -> DataProcessor:
    """Setup and load data processor"""
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing DataProcessor with hybrid search capabilities...")
    processor = DataProcessor()
    
    if os.path.exists(processed_data_dir):
        logger.info(f"Loading processed data from {processed_data_dir}")
        processor.load_processed_data(processed_data_dir)
    else:
        logger.error(f"Processed data directory not found: {processed_data_dir}")
        logger.info("Please run the data processing pipeline first")
        raise FileNotFoundError(f"Processed data not found: {processed_data_dir}")
    
    logger.info("✅ DataProcessor loaded with hybrid search and re-ranking capabilities")
    return processor

def run_paper_allocation(agent: PaperAllocationAgent, 
                        survey_title: str, 
                        section_titles: List[str], 
                        survey_df: pd.DataFrame) -> Dict:
    """Run paper allocation for all sections of a survey"""
    logger = logging.getLogger(__name__)
    
    allocations = {}
    
    for section_title in section_titles:
        logger.info(f"Allocating papers for section: {section_title}")
        
        try:
            result = agent.allocate(
                section_title=section_title,
                survey_title=survey_title,
                survey_df=survey_df,
                k=5  # Allocate top 5 papers per section
            )
            
            allocations[section_title] = result
            logger.info(f"✅ Allocated {len(result['allocated_papers'])} papers for {section_title}")
            
        except Exception as e:
            logger.error(f"❌ Failed to allocate papers for {section_title}: {str(e)}")
            allocations[section_title] = {
                "section_title": section_title,
                "allocated_papers": []
            }
    
    return allocations

def calculate_rouge_scores(generated_text: str, reference_text: str) -> Dict[str, float]:
    """Calculate ROUGE scores between generated and reference text"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rouge2_p': scores['rouge2'].precision, 
        'rouge2_r': scores['rouge2'].recall,
        'rougeL_f': scores['rougeL'].fmeasure,
        'rougeL_p': scores['rougeL'].precision,
        'rougeL_r': scores['rougeL'].recall,
    }

def evaluate_allocations(allocations: Dict, reference_papers: List[Dict]) -> Dict:
    """Evaluate paper allocation quality"""
    logger = logging.getLogger(__name__)
    
    total_allocated = 0
    total_sections = len(allocations)
    
    for section_title, result in allocations.items():
        allocated_papers = result.get('allocated_papers', [])
        total_allocated += len(allocated_papers)
    
    metrics = {
        'total_sections': total_sections,
        'total_allocated_papers': total_allocated,
        'avg_papers_per_section': total_allocated / max(total_sections, 1),
        'allocation_success_rate': sum(1 for _, result in allocations.items() 
                                     if len(result.get('allocated_papers', [])) > 0) / max(total_sections, 1)
    }
    
    logger.info(f"Allocation metrics: {metrics}")
    return metrics

def run_hybrid_experiment(num_surveys: int = 30, output_dir: str = "outputs"):
    """Run the hybrid search experiment"""
    logger = setup_logging()
    logger.info(f"🚀 Starting Hybrid Search Experiment for {num_surveys} surveys")
    
    # Load configuration
    config = load_simple_config()
    
    # Setup paths
    data_path = config.get('data', {}).get('raw_data_path', './data/raw/original_survey_df.pkl')
    processed_data_dir = config.get('data', {}).get('processed_data_dir', './data/processed')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading survey data...")
    survey_df = load_and_prepare_data(data_path, limit=num_surveys)
    
    # Setup data processor with hybrid search
    logger.info("Setting up DataProcessor with hybrid search...")
    processor = setup_data_processor(processed_data_dir)
    
    # Initialize paper allocation agent
    logger.info("Initializing PaperAllocationAgent...")
    allocation_agent = PaperAllocationAgent(processor)
    
    # Results storage
    all_results = []
    overall_metrics = {
        'total_surveys': 0,
        'successful_surveys': 0,
        'total_allocations': 0,
        'allocation_metrics': [],
        'processing_times': []
    }
    
    # Process each survey
    for idx, (_, survey_row) in enumerate(survey_df.iterrows()):
        start_time = time.time()
        
        survey_title = survey_row['title']
        # Get section titles from the 'section' field
        section_titles = survey_row.get('section', [])
        
        # Limit to first 5 sections per survey for faster processing
        if len(section_titles) > 5:
            section_titles = section_titles[:5]
            logger.info(f"Limited to first 5 sections (originally {len(survey_row.get('section', []))} sections)")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Survey {idx+1}/{len(survey_df)}: {survey_title}")
        logger.info(f"Sections: {len(section_titles)}")
        
        # Log first few section titles for debugging
        if len(section_titles) > 0:
            logger.info(f"First few sections: {section_titles[:3]}")
        else:
            logger.warning("No sections found for this survey!")
        
        try:
            # Run paper allocation with hybrid search
            allocations = run_paper_allocation(
                agent=allocation_agent,
                survey_title=survey_title,
                section_titles=section_titles,
                survey_df=survey_df
            )
            
            # Evaluate allocations
            # Get reference papers from bib_titles and bib_abstracts
            reference_papers = []
            bib_titles_list = survey_row.get('bib_titles', [])
            bib_abstracts_list = survey_row.get('bib_abstracts', [])
            
            # Collect all reference papers from all sections
            all_ref_ids = set()
            for bib_dict in bib_titles_list:
                if isinstance(bib_dict, dict):
                    all_ref_ids.update(bib_dict.keys())
            
            reference_papers = list(all_ref_ids)
            allocation_metrics = evaluate_allocations(allocations, reference_papers)
            
            processing_time = time.time() - start_time
            
            # Store results
            survey_result = {
                'survey_id': idx + 1,
                'title': survey_title,
                'original_sections_count': len(survey_row.get('section', [])),
                'processed_sections_count': len(section_titles),
                'section_titles': section_titles,
                'allocations': allocations,
                'metrics': allocation_metrics,
                'reference_papers_count': len(reference_papers),
                'processing_time': processing_time
            }
            
            all_results.append(survey_result)
            
            # Update overall metrics
            overall_metrics['total_surveys'] += 1
            overall_metrics['successful_surveys'] += 1
            overall_metrics['total_allocations'] += allocation_metrics['total_allocated_papers']
            overall_metrics['allocation_metrics'].append(allocation_metrics)
            overall_metrics['processing_times'].append(processing_time)
            
            logger.info(f"✅ Survey {idx+1} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to process survey {idx+1}: {str(e)}")
            overall_metrics['total_surveys'] += 1
            continue
    
    # Calculate final metrics
    if overall_metrics['allocation_metrics']:
        avg_papers_per_section = sum(m['avg_papers_per_section'] for m in overall_metrics['allocation_metrics']) / len(overall_metrics['allocation_metrics'])
        avg_success_rate = sum(m['allocation_success_rate'] for m in overall_metrics['allocation_metrics']) / len(overall_metrics['allocation_metrics'])
        avg_processing_time = sum(overall_metrics['processing_times']) / len(overall_metrics['processing_times'])
        
        overall_metrics.update({
            'avg_papers_per_section': avg_papers_per_section,
            'avg_allocation_success_rate': avg_success_rate,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': sum(overall_metrics['processing_times'])
        })
    
    # Save results
    results_file = output_dir / f"hybrid_experiment_results_{num_surveys}surveys.json"
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_config': {
                'num_surveys': num_surveys,
                'data_path': str(data_path),
                'output_dir': str(output_dir),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'overall_metrics': overall_metrics,
            'survey_results': all_results
        }, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("🎉 HYBRID SEARCH EXPERIMENT COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Processed surveys: {overall_metrics['successful_surveys']}/{overall_metrics['total_surveys']}")
    logger.info(f"Total papers allocated: {overall_metrics['total_allocations']}")
    logger.info(f"Average papers per section: {overall_metrics.get('avg_papers_per_section', 0):.2f}")
    logger.info(f"Average allocation success rate: {overall_metrics.get('avg_allocation_success_rate', 0):.2f}")
    logger.info(f"Average processing time per survey: {overall_metrics.get('avg_processing_time', 0):.2f}s")
    logger.info(f"Total processing time: {overall_metrics.get('total_processing_time', 0):.2f}s")
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hybrid search experiment")
    parser.add_argument("--num-surveys", type=int, default=30, help="Number of surveys to process (default: 30)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_hybrid_experiment(num_surveys=args.num_surveys, output_dir=args.output_dir)