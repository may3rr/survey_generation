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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient

class SectionWriterAgent:
    """章节正文智能体，负责生成每个章节的具体内容"""
    
    def __init__(self, api_client: GPTAPIClient):
        """
        初始化智能体
        Args:
            api_client: GPT API客户端实例
        """
        self.api_client = api_client
        self._setup_logger()
        self._setup_output_format()
    
    def _setup_logger(self):
        """配置日志系统"""
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
        """设置输出格式模板"""
        self.section_template = """# {title}

{content}

References:
{references}
"""

    def _create_section_prompt(self, 
                             section_title: str, 
                             papers_info: List[Dict]) -> str:
        """
        创建章节生成的prompt
        Args:
            section_title: 章节标题
            papers_info: 相关论文信息
        Returns:
            格式化的prompt
        """
        prompt = f"""Write a comprehensive section for a literature review paper.

Section Title: {section_title}

Based on the following papers:
"""
        # 添加参考文献信息
        for i, paper in enumerate(papers_info, 1):
            prompt += f"\n[{paper['citation_id']}] {paper['title']}\n"
            prompt += f"Abstract: {paper['abstract']}\n"

        # 添加写作要求
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
        生成章节内容
        Args:
            section_title: 章节标题
            papers_info: 相关论文信息
            prompt_template: 提示词模板
            max_tokens: 最大生成token数
        Returns:
            格式化的章节内容
        """
        try:
            self.logger.info(f"Generating content for section: {section_title}")
            
            # 使用提供的模板或创建新的prompt
            prompt = prompt_template if prompt_template else \
                    self._create_section_prompt(section_title, papers_info)
            
            # 生成内容
            content = self.api_client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if not content:
                self.logger.error(f"Failed to generate content for {section_title}")
                return None
            
            # 提取引用的论文ID
            referenced_ids = set()
            for paper in papers_info:
                if paper['citation_id'] in content:
                    referenced_ids.add(paper['citation_id'])
            
            # 格式化参考文献
            references = self._format_references(papers_info, referenced_ids)
            
            # 组装完整的章节
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
        格式化参考文献列表
        Args:
            papers_info: 论文信息列表
            referenced_ids: 被引用的论文ID集合
        Returns:
            格式化的参考文献字符串
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
        并行生成多个章节
        Args:
            sections_info: 章节信息字典 {章节标题: 论文列表}
            prompt_templates: 提示词模板字典 {章节标题: 提示词}
            max_workers: 最大并行数
        Returns:
            章节内容字典 {章节标题: 内容}
        """
        results = {}
        prompt_templates = prompt_templates or {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务
            future_to_section = {
                executor.submit(
                    self.generate,
                    section_title,
                    papers_info,
                    prompt_templates.get(section_title, "")
                ): section_title
                for section_title, papers_info in sections_info.items()
            }
            
            # 收集结果
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
        保存生成的章节内容
        Args:
            sections_content: 章节内容字典
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(sections_content, f, indent=2)
            self.logger.info(f"Successfully saved sections to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving sections: {str(e)}")