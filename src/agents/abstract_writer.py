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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient
class AbstractWriterAgent:
    """摘要编写智能体，负责生成综述论文的摘要"""
    
    def __init__(self, api_client: GPTAPIClient):
        """
        初始化智能体
        Args:
            api_client: GPT API客户端实例
        """
        self.api_client = api_client
        self._setup_logger()
    
    def _setup_logger(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_abstract_prompt(self, 
                              survey_title: str, 
                              allocation_results: Dict) -> str:
        """
        创建生成摘要的prompt
        Args:
            survey_title: 综述标题
            allocation_results: 论文分配结果
        Returns:
            格式化的prompt
        """
        prompt = f"""Please write an academic abstract for a literature review paper titled "{survey_title}".

The paper covers the following sections and key papers:
"""
        
        # 添加每个章节的信息
        for section_title, papers in allocation_results.items():
            prompt += f"\n{section_title}:\n"
            for paper in papers['allocated_papers'][:3]:  # 每个章节选择前3篇最相关的论文
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
        生成综述摘要
        Args:
            survey_title: 综述标题
            allocation_results: 论文分配结果
            max_tokens: 生成的最大token数
        Returns:
            生成的摘要
        """
        try:
            self.logger.info("Generating abstract...")
            
            # 创建prompt
            prompt = self._create_abstract_prompt(survey_title, allocation_results)
            
            # 调用API生成摘要
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