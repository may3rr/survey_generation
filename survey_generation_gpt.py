# main.py

import os
import json
import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from src.data_processor import DataProcessor
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity

class GPTAPIClient:
    """GPT API Client"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = 'https://api.gpt.ge/v1',
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_text(self, 
                     prompt: str,
                     max_tokens: int = 1000,
                     temperature: float = 0.7) -> Optional[str]:
        """Generate text using GPT API"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json={
                        'model': 'gpt-4o-mini',
                        'messages': [
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': temperature,
                        'max_tokens': max_tokens
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                self.logger.info("Successfully generated text")
                return result['choices'][0]['message']['content']
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    self.logger.warning(f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    return None

class PaperAllocationAgent:
    """Paper allocation agent that combines direct matching and vector search"""
    
    def __init__(self, data_processor):
        """Initialize the agent
        Args:
            data_processor: The data processor instance for vector search
        """
        self.processor = data_processor
        self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_paper_pool(self, survey_df: pd.DataFrame, title: str) -> Dict:
        """Get the complete paper pool for vector search"""
        try:
            # Find the survey
            survey = survey_df[survey_df['title'] == title].iloc[0]
            
            paper_pool = {}
            
            # Collect all papers from all sections
            for bib_titles in survey['bib_titles']:
                for paper_id, title in bib_titles.items():
                    if paper_id not in paper_pool:
                        paper_pool[paper_id] = {'title': title}
            
            # Add abstracts
            for bib_abstracts in survey['bib_abstracts']:
                for paper_id, abstract in bib_abstracts.items():
                    if paper_id in paper_pool:
                        paper_pool[paper_id]['abstract'] = abstract
            
            return paper_pool
            
        except Exception as e:
            self.logger.error(f"Error getting paper pool: {str(e)}")
            return {}

    def allocate(self, section_title: str, survey_title: str, survey_df: pd.DataFrame, k: int = 5) -> Dict:
        """
        Allocate papers using direct matching or vector search as fallback
        
        Args:
            section_title: The title of the section
            survey_title: The title of the survey paper
            survey_df: DataFrame containing survey data
            k: Number of papers to return for vector search
            
        Returns:
            Dict containing section title and allocated papers
        """
        try:
            # Get the survey data
            survey_data = survey_df[survey_df['title'] == survey_title].iloc[0]
            
            # Find the section index
            try:
                section_idx = survey_data['section'].index(section_title)
            except ValueError:
                self.logger.error(f"Section {section_title} not found in survey")
                return {"section_title": section_title, "allocated_papers": []}
            
            # Check if we have citations for this section
            n_citations = survey_data['n_bibs'][section_idx] if section_idx < len(survey_data['n_bibs']) else 0
            
            if n_citations > 0:
                # Use direct matching if we have citations
                self.logger.info(f"Using direct matching for section {section_title} with {n_citations} citations")
                
                allocated_papers = []
                section_titles = survey_data['bib_titles'][section_idx]
                section_abstracts = survey_data['bib_abstracts'][section_idx] if section_idx < len(survey_data['bib_abstracts']) else {}
                
                for paper_id, title in section_titles.items():
                    paper_info = {
                        "citation_id": paper_id,
                        "title": title,
                        "abstract": section_abstracts.get(paper_id, "")
                    }
                    allocated_papers.append(paper_info)
            
            else:
                # Use vector search if no citations
                self.logger.info(f"Using vector search for section {section_title}")
                
                # Get complete paper pool for search
                paper_pool = self.get_paper_pool(survey_df, survey_title)
                
                # Convert paper pool to list format for search
                pool_papers = [
                    {
                        'citation_id': pid,
                        'title': info['title'],
                        'abstract': info.get('abstract', '')
                    }
                    for pid, info in paper_pool.items()
                ]
                
                # Prepare query from section info
                query = f"{survey_title} {section_title}"
                if section_idx < len(survey_data['text']):
                    query += f" {survey_data['text'][section_idx]}"
                
                # Search similar papers
                allocated_papers = self.processor.search_in_papers(query, pool_papers, k)
            
            result = {
                "section_title": section_title,
                "allocated_papers": allocated_papers
            }
            
            self.logger.info(f"Allocated {len(allocated_papers)} papers for section: {section_title}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in paper allocation: {str(e)}")
            return {"section_title": section_title, "allocated_papers": []}

class AbstractWriterAgent:
    """Abstract writer agent"""
    
    def __init__(self, api_client: GPTAPIClient):
        self.api_client = api_client
        self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate(self, survey_title: str, allocation_results: Dict) -> str:
        """Generate abstract"""
        try:
            prompt = f"""Write a comprehensive academic abstract for a survey paper titled "{survey_title}".

Key papers reviewed in each section:
"""
            # Add section information
            for section, papers in allocation_results.items():
                prompt += f"\n{section}:\n"
                for paper in papers['allocated_papers'][:3]:
                    prompt += f"- {paper['title']}\n"
            
            prompt += """
The abstract should:
1. Introduce the research field and its importance
2. Outline the main aspects covered
3. Summarize key findings and patterns
4. Follow academic writing style
5. Be approximately 250 words
"""
            
            self.logger.info("Generating abstract...")
            abstract = self.api_client.generate_text(prompt, max_tokens=500)
            return abstract if abstract else ""
            
        except Exception as e:
            self.logger.error(f"Error generating abstract: {str(e)}")
            return ""

class SectionWriterAgent:
    """Section content writer agent with outline awareness"""
    
    def __init__(self, api_client: GPTAPIClient):
        self.api_client = api_client
        self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate(self, 
                section_title: str, 
                papers_info: List[Dict], 
                survey_title: str,
                full_outline: List[str],
                max_tokens: int = 1500) -> str:
        """
        Generate section content with awareness of the full document structure
        
        Args:
            section_title: Title of the current section
            papers_info: List of papers allocated to this section
            survey_title: Title of the survey paper
            full_outline: List of all section titles in order
            max_tokens: Maximum tokens for generation
        """
        try:
            # Create outline context
            section_idx = full_outline.index(section_title)
            current_depth = len(section_title.split('.')[0].strip()) # Get section depth (I, II, A, etc.)
            
            prompt = f"""You are writing a section for an academic survey paper titled "{survey_title}".

Full paper outline:
"""
            # Add full outline with highlighting of current section
            for i, section in enumerate(full_outline):
                if i == section_idx:
                    prompt += f">>> {section} (Current section) <<<\n"
                else:
                    prompt += f"{section}\n"

            prompt += f'\nYour task is to write the section titled "{section_title}" based on these papers:\n'
            
            # Add paper information
            for paper in papers_info:
                prompt += f"\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n"

            prompt += """
Writing requirements:
1. Consider the section's position in the overall structure when writing
2. Ensure smooth transitions and connections with adjacent sections
3. Synthesize the main ideas from the papers
4. Compare and contrast different approaches
5. Identify key trends and patterns
6. Use proper academic writing style
7. Include citations in [ID] format
8. Write approximately 800 words
9. Structure with:
   - Brief introduction connecting to previous sections
   - Main discussion
   - Synthesis
   - Brief conclusion leading to next sections

Remember: Your section is part of a larger survey paper. Maintain coherence with the overall structure.
"""

            self.logger.info(f"Generating content for section: {section_title}")
            content = self.api_client.generate_text(prompt, max_tokens=max_tokens)
            return content if content else ""
            
        except Exception as e:
            self.logger.error(f"Error generating section: {str(e)}")
            return ""

class SurveyGenerator:
    """Survey generation system"""
    
    def __init__(self, api_key: str, processed_data_dir: str = "./processed_data"):
        """Initialize the survey generator
        
        Args:
            api_key: API key for GPT service
            processed_data_dir: Directory containing processed data
        """
        self._setup_logger()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.data_processor.load_processed_data(processed_data_dir)
        
        self.api_client = GPTAPIClient(api_key)
        self.paper_allocator = PaperAllocationAgent(self.data_processor)
        self.abstract_writer = AbstractWriterAgent(self.api_client)
        self.section_writer = SectionWriterAgent(self.api_client)
        
        # Create output directory
        self.output_dir = "./outputtest"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_single_survey(self, survey_info: Dict, survey_df: pd.DataFrame, output_prefix: str) -> Dict:
        """Generate a single survey with context-aware section writing"""
        try:
            self.logger.info(f"Starting generation for survey: {survey_info['title']}")
            
            # 1. Allocate papers for each section
            allocation_results = {}
            for section in survey_info['sections']:
                result = self.paper_allocator.allocate(
                    section_title=section,
                    survey_title=survey_info['title'],
                    survey_df=survey_df
                )
                allocation_results[section] = result
            
            # Save allocation results
            allocation_file = f"{self.output_dir}/{output_prefix}_allocations.json"
            with open(allocation_file, 'w') as f:
                json.dump(allocation_results, f, indent=2)
            
            # 2. Generate abstract
            abstract = self.abstract_writer.generate(
                survey_title=survey_info['title'],
                allocation_results=allocation_results
            )
            
            # 3. Generate sections with full context
            sections_content = {}
            full_outline = survey_info['sections']  # Complete outline of the paper
            
            for section in full_outline:
                content = self.section_writer.generate(
                    section_title=section,
                    papers_info=allocation_results[section]['allocated_papers'],
                    survey_title=survey_info['title'],
                    full_outline=full_outline
                )
                sections_content[section] = content
            
            # 4. Combine results
            survey_content = {
                'title': survey_info['title'],
                'abstract': abstract,
                'sections': sections_content,
                'paper_allocation': allocation_results
            }
            
            # 5. Save results
            output_file = f"{self.output_dir}/{output_prefix}_generated.json"
            with open(output_file, 'w') as f:
                json.dump(survey_content, f, indent=2)
            
            self.logger.info(f"Successfully generated survey: {survey_info['title']}")
            return survey_content
            
        except Exception as e:
            self.logger.error(f"Error generating survey: {str(e)}")
            return {}

    def generate_multiple_surveys(self, survey_df: pd.DataFrame, num_surveys: int = 3):
        """Generate multiple surveys"""
        try:
            self.logger.info(f"Starting generation of {num_surveys} surveys")

            available_surveys = len(survey_df)
            if num_surveys > available_surveys:
                raise ValueError(
                    f"Requested {num_surveys} surveys but only {available_surveys} are available in the dataset. "
                    "Please adjust the configuration."
                )

            for i in range(num_surveys):
                # Select a survey as template
                survey = survey_df.iloc[i]
                survey_info = {
                    'title': survey['title'],
                    'sections': survey['section'],
                    'original_text': survey['text']
                }
                
                # Generate survey
                self.generate_single_survey(
                    survey_info=survey_info,
                    survey_df=survey_df,
                    output_prefix=f"survey_{i+1}"
                )
            
            self.logger.info("Completed generation of all surveys")
            
        except Exception as e:
            self.logger.error(f"Error in survey generation process: {str(e)}")

def main():
    try:
        load_dotenv()

        # Read original data
        df = pd.read_pickle('./data/raw/original_survey_df.pkl')
        
        # Initialize generator
        api_key = os.getenv('GPT_API_KEY')
        if not api_key:
            raise ValueError("GPT_API_KEY environment variable not set. Please configure it in your .env file.")

        generator = SurveyGenerator(
            api_key=api_key
        )
        
        # Generate survey
        generator.generate_multiple_surveys(df, num_surveys=2)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
