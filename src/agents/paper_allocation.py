# agents/paper_allocation.py

from typing import Dict, List, Optional
import json
import logging
from src.data_processor import DataProcessor
import json
import logging
from typing import Dict, List
import os
import sys
import pandas as pd
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient


class PaperAllocationAgent:
    """Paper Allocation Agent"""
    
    def __init__(self, data_processor: DataProcessor):
        self.processor = data_processor
        self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_paper_pool(self, df: pd.DataFrame, title: str) -> Dict:
        """
        Get reference paper pool for a specific survey
        Args:
            df: Original dataset
            title: Survey title
        Returns:
            Reference paper pool
        """
        # Find the corresponding survey
        survey = df[df['title'] == title].iloc[0]
        
        # Get all references for this survey
        paper_pool = {}
        
        # Extract information from bib_titles and bib_abstracts
        for bib_dict in survey['bib_titles']:
            for paper_id, title in bib_dict.items():
                if paper_id not in paper_pool:
                    paper_pool[paper_id] = {'title': title}
        
        for bib_dict in survey['bib_abstracts']:
            for paper_id, abstract in bib_dict.items():
                if paper_id in paper_pool:
                    paper_pool[paper_id]['abstract'] = abstract
        
        return paper_pool

    def allocate(self, section_title: str, survey_title: str, survey_df: pd.DataFrame, k: int = 5) -> Dict:
        """
        Allocate references for specified section
        Args:
            section_title: Section title
            survey_title: Survey title
            survey_df: Original dataset
            k: Number of related papers to return
        Returns:
            Dictionary of allocation results
        """
        try:
            # Get reference pool for this survey
            paper_pool = self.get_paper_pool(survey_df, survey_title)
            self.logger.info(f"Found {len(paper_pool)} papers in the reference pool")
            
            # Construct search query
            query = f"{survey_title} {section_title}"
            
            # Search in reference pool
            # Convert reference pool papers to processor-compatible format
            pool_papers = [
                {
                    'citation_id': pid,
                    'title': info['title'],
                    'abstract': info.get('abstract', '')
                }
                for pid, info in paper_pool.items()
            ]
            
            # Use processor's vector similarity search
            similar_papers = self.processor.search_in_papers(
                query=query,
                papers=pool_papers,
                k=k
            )
            
            result = {
                "section_title": section_title,
                "allocated_papers": similar_papers
            }
            
            self.logger.info(f"Successfully allocated {len(similar_papers)} papers for section: {section_title}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in paper allocation: {str(e)}")
            return {
                "section_title": section_title, 
                "allocated_papers": []
            }