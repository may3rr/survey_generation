# agents/prompt_writer.py

import logging
from typing import Dict, List, Optional
import json
import logging
from typing import Dict, List
import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient

class PromptWriterAgent:
    """Prompt writing agent responsible for generating section content prompts"""
    
    def __init__(self):
        self._setup_logger()
        self._load_templates()
    
    def _setup_logger(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_templates(self):
        """Load base templates"""
        self.base_template = """As an academic writer, please write a comprehensive section for a literature review paper.

Title: "{title}"
Section: "{section}"

Based on the provided papers:
{papers_info}

Requirements:
1. Write in an academic style suitable for a literature review
2. Synthesize and analyze the main ideas from the provided papers
3. Compare and contrast different approaches
4. Identify trends, patterns, or gaps in the literature
5. Properly cite papers using [ID] format
6. Maintain logical flow and cohesion
7. Focus on critical analysis rather than just summarizing

The section should:
- Begin with a brief introduction of the topic
- Present key concepts and findings
- Discuss relationships between different works
- End with a brief summary or transition
- Be approximately 500-800 words

Output Format Example:
[Section content with proper citations, e.g., "Smith et al. [123] proposed..."]"""

    def _analyze_section_type(self, section_title: str) -> str:
        """
        Analyze section type and add specific prompts
        Args:
            section_title: Section title
        Returns:
            Section-specific prompts
        """
        section_lower = section_title.lower()
        
        if "introduction" in section_lower:
            return """Additional Requirements for Introduction:
- Provide context and background of the research field
- Present the significance and motivation of the topic
- Outline the scope and organization of the review
- Highlight key challenges or open problems"""
            
        elif "method" in section_lower or "approach" in section_lower:
            return """Additional Requirements for Methodology Section:
- Clearly explain technical concepts and principles
- Compare different methodologies systematically
- Discuss advantages and limitations of each approach
- Use precise technical language"""
            
        elif "future" in section_lower or "conclusion" in section_lower:
            return """Additional Requirements for Future Directions:
- Summarize key findings and patterns
- Identify research gaps and challenges
- Suggest promising research directions
- Discuss potential applications and implications"""
            
        return ""

    def generate(self, 
                survey_title: str, 
                section_title: str,
                papers_info: List[Dict]) -> str:
        """
        Generate prompts for the section
        Args:
            survey_title: Survey title
            section_title: Section title
            papers_info: List of related paper information
        Returns:
            Formatted prompt
        """
        try:
            # Organize paper information
            papers_text = ""
            for i, paper in enumerate(papers_info, 1):
                papers_text += f"""Paper {i} [ID: {paper['citation_id']}]:
Title: {paper['title']}
Abstract: {paper['abstract']}
---
"""
            
            # Get section-specific prompts
            section_specific = self._analyze_section_type(section_title)
            
            # Combine complete prompt
            prompt = self.base_template.format(
                title=survey_title,
                section=section_title,
                papers_info=papers_text
            )
            
            if section_specific:
                prompt += f"\n\n{section_specific}"
            
            self.logger.info(f"Generated prompt for section: {section_title}")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error generating prompt: {str(e)}")
            return self.base_template.format(
                title=survey_title,
                section=section_title,
                papers_info="Error occurred while processing papers."
            )