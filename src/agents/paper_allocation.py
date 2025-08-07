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
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lhb.utils.api_client import GPTAPIClient


class PaperAllocationAgent:
    """文献分配智能体"""
    
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
        获取特定综述的参考文献池
        Args:
            df: 原始数据集
            title: 综述标题
        Returns:
            参考文献池
        """
        # 找到对应的综述
        survey = df[df['title'] == title].iloc[0]
        
        # 获取该综述的所有参考文献
        paper_pool = {}
        
        # 从bib_titles和bib_abstracts中提取信息
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
        为指定章节分配参考文献
        Args:
            section_title: 章节标题
            survey_title: 综述标题
            survey_df: 原始数据集
            k: 返回的相关论文数量
        Returns:
            分配结果的字典
        """
        try:
            # 获取该综述的参考文献池
            paper_pool = self.get_paper_pool(survey_df, survey_title)
            self.logger.info(f"Found {len(paper_pool)} papers in the reference pool")
            
            # 构造搜索查询
            query = f"{survey_title} {section_title}"
            
            # 在参考文献池中搜索
            # 将参考文献池的论文转换为processor可处理的格式
            pool_papers = [
                {
                    'citation_id': pid,
                    'title': info['title'],
                    'abstract': info.get('abstract', '')
                }
                for pid, info in paper_pool.items()
            ]
            
            # 使用processor的向量相似度搜索
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