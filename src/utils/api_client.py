# utils/api_client.py

import requests
import logging
import time
from typing import Dict, Optional, Tuple

class GPTAPIClient:
    """GPT API Client"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = 'https://api.gpt.ge/v1',
                 model: str = 'gpt-5-mini-minimal',
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize client
        Args:
            api_key: API key
            base_url: API base URL
            model: Model name to use
            max_retries: Maximum number of retries
            retry_delay: Retry delay time (seconds)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_text(self, 
                     prompt: str,
                     max_tokens: int = 1000,
                     temperature: float = 0.7) -> Tuple[Optional[str], Optional[Dict], float]:
        """
        Generate text
        Args:
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
        Returns:
            Generated text, returns None if failed
        """
        retry_count = 0
        while retry_count < self.max_retries:
            start_time = time.monotonic()
            try:
                # Fix URL construction to avoid duplicate /v1
                chat_url = f'{self.base_url}/chat/completions' if self.base_url.endswith('/v1') else f'{self.base_url}/v1/chat/completions'
                
                response = requests.post(
                    chat_url,
                    headers=self.headers,
                    json={
                        'model': self.model,
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
                duration = time.monotonic() - start_time
                usage = result.get('usage')
                
                self.logger.info("Successfully generated text")
                if result.get('choices'):
                    return result['choices'][0]['message']['content'], usage, duration
                return None, usage, duration
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                duration = time.monotonic() - start_time
                if retry_count < self.max_retries:
                    self.logger.warning(
                        f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}"
                    )
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    return None, None, duration
            
            except Exception as e:
                retry_count += 1
                duration = time.monotonic() - start_time
                self.logger.error(f"Unexpected error: {str(e)}")
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return None, None, duration
