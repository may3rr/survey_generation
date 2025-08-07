# src/data_processor.py

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import faiss
import json
import logging
from tqdm import tqdm
# 在文件顶部添加以下导入
from sklearn.metrics.pairwise import cosine_similarity

class DataProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化数据处理器
        Args:
            model_name: 用于文本嵌入的模型名称
        """
        # 基础数据存储
        self.raw_data = None
        self.bib_abstracts_dict = {}  # citation_id -> abstract
        self.bib_titles_dict = {}     # citation_id -> title
        self.citation_stats = {}      # citation_id -> 被引用次数
        
        # 向量检索相关
        self.model = SentenceTransformer(model_name)
        self.abstract_embeddings = None
        self.citation_ids = []  # 保持向量顺序
        self.index = None
        
        # 设置日志
        self._setup_logger()
    
    def _setup_logger(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_processor.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, pkl_path: str) -> None:
        """
        加载并处理原始数据
        Args:
            pkl_path: PKL文件路径
        """
        try:
            self.logger.info(f"Loading data from {pkl_path}")
            self.raw_data = pd.read_pickle(pkl_path)
            self._process_citations()
            self.logger.info(f"Successfully loaded {len(self.raw_data)} papers")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _process_citations(self) -> None:
        """处理所有引用信息"""
        self.logger.info("Processing citations...")
        
        # 重置数据结构
        self.bib_abstracts_dict.clear()
        self.bib_titles_dict.clear()
        self.citation_stats.clear()
        
        for _, row in tqdm(self.raw_data.iterrows(), total=len(self.raw_data)):
            # 处理摘要
            bib_abstracts_list = row['bib_abstracts']
            for bib_dict in bib_abstracts_list:
                self.bib_abstracts_dict.update(bib_dict)
            
            # 处理标题
            bib_titles_list = row['bib_titles']
            for bib_dict in bib_titles_list:
                self.bib_titles_dict.update(bib_dict)
            
            # 统计引用次数
            for bib_dict in bib_abstracts_list:
                for citation_id in bib_dict.keys():
                    self.citation_stats[citation_id] = self.citation_stats.get(citation_id, 0) + 1

        self.logger.info(f"Processed {len(self.bib_abstracts_dict)} unique citations")

    def create_vector_store(self) -> None:
        """创建向量存储"""
        self.logger.info("Creating vector store...")
        
        # 准备数据
        abstracts = []
        self.citation_ids = []
        
        for cid, abstract in self.bib_abstracts_dict.items():
            abstracts.append(abstract)
            self.citation_ids.append(cid)
        
        # 计算嵌入向量
        try:
            self.abstract_embeddings = self.model.encode(abstracts, show_progress_bar=True)
            
            # 创建FAISS索引
            dimension = self.abstract_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.abstract_embeddings.astype('float32'))
            
            self.logger.info(f"Successfully created vector store with {len(abstracts)} vectors")
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def save_processed_data(self, save_dir: str) -> None:
        """
        保存处理后的数据
        Args:
            save_dir: 保存目录
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存基础数据
            with open(f"{save_dir}/bib_abstracts.json", 'w') as f:
                json.dump(self.bib_abstracts_dict, f)
            
            with open(f"{save_dir}/bib_titles.json", 'w') as f:
                json.dump(self.bib_titles_dict, f)
            
            with open(f"{save_dir}/citation_stats.json", 'w') as f:
                json.dump(self.citation_stats, f)
            
            # 保存向量检索相关数据
            if self.index is not None:
                faiss.write_index(self.index, f"{save_dir}/abstract_index.faiss")
                np.save(f"{save_dir}/citation_ids.npy", np.array(self.citation_ids))
            
            self.logger.info(f"Successfully saved processed data to {save_dir}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def load_processed_data(self, save_dir: str) -> None:
        """
        加载处理后的数据
        Args:
            save_dir: 数据目录
        """
        try:
            # 加载基础数据
            with open(f"{save_dir}/bib_abstracts.json", 'r') as f:
                self.bib_abstracts_dict = json.load(f)
            
            with open(f"{save_dir}/bib_titles.json", 'r') as f:
                self.bib_titles_dict = json.load(f)
            
            with open(f"{save_dir}/citation_stats.json", 'r') as f:
                self.citation_stats = json.load(f)
            
            # 加载向量检索相关数据
            if os.path.exists(f"{save_dir}/abstract_index.faiss"):
                self.index = faiss.read_index(f"{save_dir}/abstract_index.faiss")
                self.citation_ids = np.load(f"{save_dir}/citation_ids.npy").tolist()
            
            self.logger.info(f"Successfully loaded processed data from {save_dir}")
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            raise

    def get_abstracts_by_ids(self, citation_ids: List[str]) -> Dict[str, str]:
        """
        获取指定引用ID的摘要
        Args:
            citation_ids: 引用ID列表
        Returns:
            ID到摘要的映射字典
        """
        return {cid: self.bib_abstracts_dict.get(cid, "") for cid in citation_ids}

    def get_titles_by_ids(self, citation_ids: List[str]) -> Dict[str, str]:
        """
        获取指定引用ID的标题
        Args:
            citation_ids: 引用ID列表
        Returns:
            ID到标题的映射字典
        """
        return {cid: self.bib_titles_dict.get(cid, "") for cid in citation_ids}

    def search_similar_abstracts(self, query: str, k: int = 5) -> List[Dict]:
        """
        搜索与查询最相似的摘要
        Args:
            query: 查询文本
            k: 返回结果数量
        Returns:
            相似文献列表，包含ID、标题、摘要和相似度分数
        """
        if self.index is None:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
        
        try:
            # 编码查询
            query_vector = self.model.encode([query])
            
            # 搜索最相似的向量
            distances, indices = self.index.search(query_vector.astype('float32'), k)
            
            # 整理结果
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                citation_id = self.citation_ids[idx]
                results.append({
                    'citation_id': citation_id,
                    'title': self.bib_titles_dict.get(citation_id, ""),
                    'abstract': self.bib_abstracts_dict.get(citation_id, ""),
                    'citation_count': self.citation_stats.get(citation_id, 0),
                    'similarity_score': float(1 / (1 + dist))
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise

    def get_citation_stats(self, citation_id: str) -> Optional[int]:
        """
        获取指定引用的被引用次数
        Args:
            citation_id: 引用ID
        Returns:
            被引用次数，如果不存在返回None
        """
        return self.citation_stats.get(citation_id)
    
    def search_in_papers(self, query: str, papers: List[Dict], k: int = 5) -> List[Dict]:
        """
        在指定的论文集合中搜索相似论文
        Args:
            query: 查询文本
            papers: 论文列表 [{citation_id, title, abstract}, ...]
            k: 返回结果数量
        Returns:
            相似论文列表
        """
        try:
            if not papers:
                self.logger.warning("Empty paper list provided for search")
                return []

            # 编码查询文本
            query_vector = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
            
            # 编码所有论文的标题和摘要
            paper_texts = [f"{p['title']}. {p['abstract']}" for p in papers]
            paper_vectors = self.model.encode(paper_texts, convert_to_numpy=True)
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, paper_vectors)[0]
            
            # 获取top-k的索引
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # 整理结果
            results = []
            for idx in top_k_indices:
                paper = papers[idx].copy()
                paper['similarity_score'] = float(similarities[idx])
                results.append(paper)
            
            self.logger.info(f"Found {len(results)} similar papers")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in paper search: {str(e)}")
            return []