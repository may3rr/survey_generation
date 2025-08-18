# agents/section_writer.py

import logging
from typing import Dict, List, Optional
import json
from utils.api_client import GPTAPIClient
import concurrent.futures
import json
import logging
from typing import Dict, List
import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient

class SectionWriterAgent:
    """Section content agent responsible for generating specific content for each section"""
    
    def __init__(self, api_client: GPTAPIClient):
        """
        Initialize the agent
        Args:
            api_client: GPT API client instance
        """
        self.api_client = api_client
        self._setup_logger()
        self._setup_output_format()
    
    def _setup_logger(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('section_writer.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_output_format(self):
        """Set output format template"""
        self.section_template = """# {title}

{content}

References:
{references}
"""

    def _create_section_prompt(self, 
                             section_title: str, 
                             papers_info: List[Dict]) -> str:
        """
        Create prompt for section generation
        Args:
            section_title: Section title
            papers_info: Related paper information
        Returns:
            Formatted prompt
        """
        prompt = f"""Write a comprehensive section for a literature review paper.

Section Title: {section_title}

Based on the following papers:
"""
        # Add reference information
        for i, paper in enumerate(papers_info, 1):
            prompt += f"\n[{paper['citation_id']}] {paper['title']}\n"
            prompt += f"Abstract: {paper['abstract']}\n"

        # Add writing requirements
        prompt += """
Requirements:
1. Write in formal academic style suitable for a literature review
2. Synthesize and critically analyze the key ideas from the papers
3. Use proper citations in the format [ID]
4. Maintain logical flow and coherence
5. Include:
   - Brief introduction of the topic
   - Main body discussing key themes and findings
   - Synthesis of different viewpoints
   - Brief conclusion or transition
6. Length: approximately 300-400 words

The section should demonstrate:
- Clear organization
- Critical analysis
- Proper integration of citations
- Academic language and tone
"""
        return prompt

    def generate(self, 
                section_title: str,
                papers_info: List[Dict],
                prompt_template: str,
                max_tokens: int = 1500) -> Optional[str]:
        """
        Generate section content
        Args:
            section_title: Section title
            papers_info: Related paper information
            prompt_template: Prompt template
            max_tokens: Maximum number of tokens to generate
        Returns:
            Formatted section content
        """
        try:
            self.logger.info(f"Generating content for section: {section_title}")
            
            # Use provided template or create new prompt
            prompt = prompt_template if prompt_template else \
                    self._create_section_prompt(section_title, papers_info)
            
            # Generate content
            content = self.api_client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if not content:
                self.logger.error(f"Failed to generate content for {section_title}")
                return None
            
            # Extract cited paper IDs
            referenced_ids = set()
            for paper in papers_info:
                if paper['citation_id'] in content:
                    referenced_ids.add(paper['citation_id'])
            
            # Format references
            references = self._format_references(papers_info, referenced_ids)
            
            # Assemble complete section
            formatted_section = self.section_template.format(
                title=section_title,
                content=content,
                references=references
            )
            
            self.logger.info(f"Successfully generated content for {section_title}")
            return formatted_section
            
        except Exception as e:
            self.logger.error(f"Error generating section content: {str(e)}")
            return None

    def _format_references(self, 
                         papers_info: List[Dict],
                         referenced_ids: set) -> str:
        """
        Format reference list
        Args:
            papers_info: List of paper information
            referenced_ids: Set of referenced paper IDs
        Returns:
            Formatted reference string
        """
        references = []
        for paper in papers_info:
            if paper['citation_id'] in referenced_ids:
                ref = f"[{paper['citation_id']}] {paper['title']}"
                references.append(ref)
        
        return "\n".join(references)

    def generate_multiple_sections(self,
                                 sections_info: Dict[str, List[Dict]],
                                 prompt_templates: Dict[str, str] = None,
                                 max_workers: int = 3) -> Dict[str, str]:
        """
        Generate multiple sections in parallel
        Args:
            sections_info: Section information dictionary {section title: paper list}
            prompt_templates: Prompt template dictionary {section title: prompt}
            max_workers: Maximum number of parallel workers
        Returns:
            Section content dictionary {section title: content}
        """
        results = {}
        prompt_templates = prompt_templates or {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks
            future_to_section = {
                executor.submit(
                    self.generate,
                    section_title,
                    papers_info,
                    prompt_templates.get(section_title, "")
                ): section_title
                for section_title, papers_info in sections_info.items()
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_section):
                section_title = future_to_section[future]
                try:
                    content = future.result()
                    if content:
                        results[section_title] = content
                    else:
                        self.logger.error(f"Failed to generate content for {section_title}")
                except Exception as e:
                    self.logger.error(f"Error processing {section_title}: {str(e)}")
        
        return results

    def save_sections(self, 
                     sections_content: Dict[str, str],
                     output_path: str):
        """
        Save generated section content
        Args:
            sections_content: Section content dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(sections_content, f, indent=2)
            self.logger.info(f"Successfully saved sections to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving sections: {str(e)}")