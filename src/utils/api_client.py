# utils/api_client.py

import requests
import logging
import time
from typing import Optional

class GPTAPIClient:
    """GPT API 客户端"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = 'https://api.gpt.ge/v1',
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        初始化客户端
        Args:
            api_key: API密钥
            base_url: API基础URL
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间(秒)
        """
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
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_text(self, 
                     prompt: str,
                     max_tokens: int = 1000,
                     temperature: float = 0.7) -> Optional[str]:
        """
        生成文本
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 采样温度
        Returns:
            生成的文本，失败则返回None
        """
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
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    self.logger.warning(
                        f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    return None
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                return None