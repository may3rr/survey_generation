# agents/abstract_writer.py

import json
import logging
from typing import Dict, List
from utils.api_client import GPTAPIClient
import json
import logging
from typing import Dict, List
import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient
class AbstractWriterAgent:
    """Abstract writing agent responsible for generating literature review abstracts"""
    
    def __init__(self, api_client: GPTAPIClient):
        """
        Initialize the agent
        Args:
            api_client: GPT API client instance
        """
        self.api_client = api_client
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_abstract_prompt(self, 
                              survey_title: str, 
                              allocation_results: Dict) -> str:
        """
        Create prompt for generating abstract
        Args:
            survey_title: Literature review title
            allocation_results: Paper allocation results
        Returns:
            Formatted prompt
        """
        prompt = f"""Please write an academic abstract for a literature review paper titled "{survey_title}".

The paper covers the following sections and key papers:
"""
        
        # Add information for each section
        for section_title, papers in allocation_results.items():
            prompt += f"\n{section_title}:\n"
            for paper in papers['allocated_papers'][:3]:  # Select top 3 most relevant papers for each section
                prompt += f"- {paper['title']}\n"

        prompt += """
Please write a comprehensive abstract that:
1. Introduces the research field and its significance
2. Outlines the main aspects covered in the review
3. Summarizes key findings or patterns from the literature
4. Follows academic writing conventions
5. Contains approximately 250 words

Ensure the abstract is:
- Concise yet informative
- Well-structured with clear flow
- Academic in tone
- Self-contained
"""
        return prompt

    def generate(self, 
                survey_title: str,
                allocation_results: Dict,
                max_tokens: int = 500) -> str:
        """
        Generate literature review abstract
        Args:
            survey_title: Literature review title
            allocation_results: Paper allocation results
            max_tokens: Maximum number of tokens to generate
        Returns:
            Generated abstract
        """
        try:
            self.logger.info("Generating abstract...")
            
            # Create prompt
            prompt = self._create_abstract_prompt(survey_title, allocation_results)
            
            # Call API to generate abstract
            abstract = self.api_client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            if abstract:
                self.logger.info("Successfully generated abstract")
                return abstract
            else:
                self.logger.error("Failed to generate abstract")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error generating abstract: {str(e)}")
            return ""