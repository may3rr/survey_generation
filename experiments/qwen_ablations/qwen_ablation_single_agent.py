import os
import json
import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_processor import DataProcessor
from datetime import datetime

class GPTAPIClient:
    """GPT API Client with conversation history - keeping original parameters"""

    def __init__(self,
                 api_key: str,
                 base_url: str = 'https://api.vveai.com/v1',
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
        self.conversation_history = []
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def generate_text(self,
                     prompt: str,
                     max_tokens: int = 1000,  # Keep original default
                     temperature: float = 0.7) -> Optional[str]:
        """Generate text using GPT API - keeping original method signature"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Build request body - exactly same as original
                request_body = {
                    'model': 'qwen3-14b',
                    'messages': [
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    # Place enable_thinking directly at root level
                    'enable_thinking': False
                }

                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json=request_body, # Use constructed request body
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                self.logger.info("Successfully generated text")
                # Check response structure to ensure correct content extraction
                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    return result['choices'][0]['message']['content']
                else:
                    self.logger.error(f"Unexpected API response structure: {result}")
                    return None

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    self.logger.warning(f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}")
                    # Print complete error response text for debugging
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                         self.logger.warning(f"API Response Text: {e.response.text}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    # Print complete error response text for debugging
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                         self.logger.error(f"API Response Text: {e.response.text}")
                    return None
            except Exception as e:
                 retry_count += 1
                 self.logger.error(f"An unexpected error occurred during API call (attempt {retry_count}/{self.max_retries}): {str(e)}")
                 if retry_count < self.max_retries:
                     time.sleep(self.retry_delay)
                 else:
                     return None

    def generate_with_history(self,
                             new_message: str,
                             max_tokens: int = 1000,  # Keep original default
                             temperature: float = 0.7) -> Optional[str]:
        """Generate text with conversation history"""
        # Add new user message
        self.add_message('user', new_message)
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                request_body = {
                    'model': 'qwen3-14b',
                    'messages': self.conversation_history,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'enable_thinking': False
                }

                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json=request_body,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    assistant_response = result['choices'][0]['message']['content']
                    # Add assistant response to history
                    self.add_message('assistant', assistant_response)
                    self.logger.info("Successfully generated text with history")
                    return assistant_response
                else:
                    self.logger.error(f"Unexpected API response structure: {result}")
                    return None

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    self.logger.warning(f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                         self.logger.warning(f"API Response Text: {e.response.text}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                         self.logger.error(f"API Response Text: {e.response.text}")
                    return None
            except Exception as e:
                 retry_count += 1
                 self.logger.error(f"An unexpected error occurred during API call (attempt {retry_count}/{self.max_retries}): {str(e)}")
                 if retry_count < self.max_retries:
                     time.sleep(self.retry_delay)
                 else:
                     return None

class ConversationalSurveyAgent:
    """Single agent for conversational survey generation - keeping original logic"""
    
    def __init__(self, api_client: GPTAPIClient, data_processor: DataProcessor):
        self.api_client = api_client
        self.processor = data_processor  # Keep original name
        self._setup_logger()
        
        # Writing state
        self.current_survey = None
        self.current_step = "initialize"
        self.completed_content = {}
        self.paper_allocations = {}
        self.survey_outline = []
        self.current_section_index = 0
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_paper_pool(self, survey_df: pd.DataFrame, title: str) -> Dict:
        """Get the complete paper pool for vector search - exact copy from original"""
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
        Allocate papers using direct matching or vector search as fallback - exact copy from original
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

    def initialize_survey(self, survey_info: Dict, survey_df: pd.DataFrame) -> str:
        """Initialize a new survey writing session"""
        self.current_survey = survey_info
        self.survey_outline = survey_info['sections']
        self.current_step = "allocating_papers"
        self.completed_content = {}
        self.paper_allocations = {}
        self.current_section_index = 0
        
        # Clear conversation history for new survey
        self.api_client.clear_history()
        
        # Allocate papers for all sections using original logic
        self.logger.info("Allocating papers for all sections...")
        allocation_results = {}
        for section in self.survey_outline:
            result = self.allocate(
                section_title=section,
                survey_title=survey_info['title'],
                survey_df=survey_df
            )
            allocation_results[section] = result
            self.paper_allocations[section] = result['allocated_papers']
        
        self.current_step = "ready_for_abstract"
        
        # Initialize conversation
        init_message = f"""I am going to help you write a comprehensive survey paper titled "{survey_info['title']}".

The paper will have the following structure:
"""
        for i, section in enumerate(self.survey_outline, 1):
            init_message += f"{i}. {section}\n"

        init_message += f"""
I have already allocated relevant papers for each section based on the existing literature. 

We will write this survey step by step:
1. First, I'll write the abstract
2. Then we'll go through each section one by one
3. You can provide feedback and ask for revisions at any point

Let's start with the abstract. I'll write a comprehensive abstract that introduces the field, outlines the main aspects covered, and summarizes key findings.
"""
        
        return init_message

    def write_abstract(self) -> str:
        """Write the abstract using original AbstractWriterAgent logic"""
        if self.current_step != "ready_for_abstract":
            return "Error: Not ready to write abstract. Please initialize the survey first."
        
        # Use exact same prompt as original AbstractWriterAgent
        prompt = f"""Write a comprehensive academic abstract for a survey paper titled "{self.current_survey['title']}".

Key papers reviewed in each section:
"""
        # Add section information - showing ALL papers, not just top 3
        for section in self.survey_outline:
            papers = self.paper_allocations[section]
            prompt += f"\n{section}:\n"
            for paper in papers:  # Show ALL papers, not just [:3]
                prompt += f"- {paper['title']}\n"
        
        prompt += """
The abstract should:
1. Introduce the research field and its importance
2. Outline the main aspects covered
3. Summarize key findings and patterns
4. Follow academic writing style
5. Be approximately 250 words
"""
        
        # Use original parameters: max_tokens=500
        response = self.api_client.generate_text(prompt, max_tokens=500)
        
        if response:
            self.completed_content['abstract'] = response
            self.current_step = "abstract_written"
        
        return response or "Error generating abstract."

    def continue_to_sections(self) -> str:
        """Move to section writing phase"""
        if self.current_step != "abstract_written":
            return "Error: Abstract must be completed first."
        
        self.current_step = "writing_sections"
        self.current_section_index = 0
        
        return f"""Great! The abstract is complete. Now let's move on to writing the sections.

We'll write them in order:
{chr(10).join([f"{i+1}. {section}" for i, section in enumerate(self.survey_outline)])}

Let's start with the first section: "{self.survey_outline[0]}"

I'll write this section based on the allocated papers and ensure it flows well with the overall survey structure."""

    def write_current_section(self) -> str:
        """Write the current section using original SectionWriterAgent logic"""
        if self.current_step != "writing_sections":
            return "Error: Not in section writing phase."
        
        if self.current_section_index >= len(self.survey_outline):
            return "Error: All sections have been completed."
        
        current_section = self.survey_outline[self.current_section_index]
        papers_info = self.paper_allocations[current_section]
        
        # Use exact same prompt structure as original SectionWriterAgent
        # Create outline context
        section_idx = self.survey_outline.index(current_section)
        full_outline = self.survey_outline
        
        prompt = f"""You are writing a section for an academic survey paper titled "{self.current_survey['title']}".

Full paper outline:
"""
        # Add full outline with highlighting of current section
        for i, section in enumerate(full_outline):
            if i == section_idx:
                prompt += f">>> {section} (Current section) <<<\n"
            else:
                prompt += f"{section}\n"

        prompt += f'\nYour task is to write the section titled "{current_section}" based on these papers:\n'
        
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

        # Use original parameters: max_tokens=1500
        response = self.api_client.generate_with_history(prompt, max_tokens=1500)
        
        if response:
            self.completed_content[current_section] = response
            self.current_section_index += 1
        
        return response or "Error generating section."

    def continue_to_next_section(self) -> str:
        """Continue to the next section"""
        if self.current_section_index >= len(self.survey_outline):
            self.current_step = "survey_complete"
            return """🎉 Congratulations! All sections have been completed!

The survey paper is now complete with:
- Abstract ✓
- All sections written ✓

You can ask me to:
1. Review any specific section
2. Make revisions to any part
3. Generate the final compiled document
4. Provide a summary of what we've accomplished

What would you like to do next?"""
        
        next_section = self.survey_outline[self.current_section_index]
        return f"""Section completed! Let's continue to the next section.

Progress: {self.current_section_index}/{len(self.survey_outline)} sections completed

Next section: "{next_section}"

I'll now write this section, ensuring it connects well with the previous content and maintains the overall flow of the survey."""

    def get_progress_status(self) -> str:
        """Get current progress status"""
        status = f"""📊 Current Progress for Survey: "{self.current_survey['title'] if self.current_survey else 'None'}"

Current Step: {self.current_step}
"""
        
        if self.current_survey:
            status += f"Sections completed: {len(self.completed_content) - (1 if 'abstract' in self.completed_content else 0)}/{len(self.survey_outline)}\n"
            
            if 'abstract' in self.completed_content:
                status += "✓ Abstract completed\n"
            
            for i, section in enumerate(self.survey_outline):
                if section in self.completed_content:
                    status += f"✓ {section}\n"
                elif i == self.current_section_index and self.current_step == "writing_sections":
                    status += f"→ {section} (current)\n"
                else:
                    status += f"○ {section}\n"
        
        return status

    def save_progress(self, output_dir: str, survey_id: str):
        """Save current progress to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save conversation history
        conversation_file = os.path.join(output_dir, f"{survey_id}_conversation.json")
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'conversation_history': self.api_client.conversation_history,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # Save allocation results - same format as original
        allocation_results = {}
        for section in self.survey_outline:
            allocation_results[section] = {
                "section_title": section,
                "allocated_papers": self.paper_allocations[section]
            }
        
        allocation_file = os.path.join(output_dir, f"{survey_id}_allocations.json")
        with open(allocation_file, 'w', encoding='utf-8') as f:
            json.dump(allocation_results, f, indent=2, ensure_ascii=False)
        
        # Save current state
        state_file = os.path.join(output_dir, f"{survey_id}_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({
                'current_survey': self.current_survey,
                'current_step': self.current_step,
                'completed_content': self.completed_content,
                'paper_allocation': allocation_results,  # Use same key as original
                'survey_outline': self.survey_outline,
                'current_section_index': self.current_section_index,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # Save final survey if complete - same format as original
        if self.current_step == "survey_complete":
            final_file = os.path.join(output_dir, f"{survey_id}_generated.json")  # Same name as original
            final_survey = {
                'title': self.current_survey['title'],
                'abstract': self.completed_content.get('abstract', ''),
                'sections': {section: self.completed_content.get(section, '') 
                           for section in self.survey_outline},
                'paper_allocation': allocation_results  # Same key as original
            }
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(final_survey, f, indent=2, ensure_ascii=False)

class SurveyGenerator:
    """Survey generation system - keeping original name and structure"""
    
    def __init__(self, api_key: str, processed_data_dir: str = "./processed_data"):
        self._setup_logger()
        
        # Initialize components - same as original
        self.data_processor = DataProcessor()
        self.data_processor.load_processed_data(processed_data_dir)
        
        self.api_client = GPTAPIClient(api_key)
        self.agent = ConversationalSurveyAgent(self.api_client, self.data_processor)
        
        # Create output directory
        self.output_dir = "./ablationQwenSingleAgent"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_single_survey(self, survey_info: Dict, survey_df: pd.DataFrame, output_prefix: str) -> Dict:
        """Generate a single survey - keeping original method signature"""
        try:
            self.logger.info(f"Starting generation for survey: {survey_info['title']}")
            
            # Initialize
            init_response = self.agent.initialize_survey(survey_info, survey_df)
            
            # Generate abstract
            abstract = self.agent.write_abstract()
            
            # Move to sections
            self.agent.continue_to_sections()
            
            # Generate all sections
            sections_content = {}
            while self.agent.current_section_index < len(self.agent.survey_outline):
                current_section = self.agent.survey_outline[self.agent.current_section_index]
                content = self.agent.write_current_section()
                sections_content[current_section] = content
                
                if self.agent.current_section_index < len(self.agent.survey_outline):
                    self.agent.continue_to_next_section()
            
            # Mark as complete
            self.agent.current_step = "survey_complete"
            
            # Save results
            self.agent.save_progress(self.output_dir, output_prefix)
            
            # Return same format as original
            survey_content = {
                'title': survey_info['title'],
                'abstract': abstract,
                'sections': sections_content,
                'paper_allocation': {section: {"section_title": section, "allocated_papers": papers} 
                                   for section, papers in self.agent.paper_allocations.items()}
            }
            
            self.logger.info(f"Successfully generated survey: {survey_info['title']}")
            return survey_content
            
        except Exception as e:
            self.logger.error(f"Error generating survey: {str(e)}")
            return {}

    def generate_multiple_surveys(self, survey_df: pd.DataFrame, num_surveys: int = 3):
        """Generate multiple surveys - exact same as original"""
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
                
                # Reset agent for next survey
                self.agent = ConversationalSurveyAgent(self.api_client, self.data_processor)
            
            self.logger.info("Completed generation of all surveys")
            
        except Exception as e:
            self.logger.error(f"Error in survey generation process: {str(e)}")

def main():
    try:
        load_dotenv()

        # Read original data
        df = pd.read_pickle('./data/raw/original_survey_df.pkl')
        
        # Initialize generator
        api_key = os.getenv('QWEN_API_KEY')
        if not api_key:
            raise ValueError("QWEN_API_KEY environment variable not set. Please configure it in your .env file.")

        generator = SurveyGenerator(
            api_key=api_key
        )
        
        # Generate survey - same as original
        generator.generate_multiple_surveys(df, num_surveys=30)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
