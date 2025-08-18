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
# Add the following import at the top of the file
from sklearn.metrics.pairwise import cosine_similarity

class DataProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize data processor
        Args:
            model_name: Model name for text embedding
        """
        # Basic data storage
        self.raw_data = None
        self.bib_abstracts_dict = {}  # citation_id -> abstract
        self.bib_titles_dict = {}     # citation_id -> title
        self.citation_stats = {}      # citation_id -> citation count
        
        # Vector retrieval related
        self.model = SentenceTransformer(model_name)
        self.abstract_embeddings = None
        self.citation_ids = []  # maintain vector order
        self.index = None
        
        # Setup logging
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging system"""
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
        Load and process raw data
        Args:
            pkl_path: Path to PKL file
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
        """Process all citation information"""
        self.logger.info("Processing citations...")
        
        # Reset data structures
        self.bib_abstracts_dict.clear()
        self.bib_titles_dict.clear()
        self.citation_stats.clear()
        
        for _, row in tqdm(self.raw_data.iterrows(), total=len(self.raw_data)):
            # Process abstracts
            bib_abstracts_list = row['bib_abstracts']
            for bib_dict in bib_abstracts_list:
                self.bib_abstracts_dict.update(bib_dict)
            
            # Process titles
            bib_titles_list = row['bib_titles']
            for bib_dict in bib_titles_list:
                self.bib_titles_dict.update(bib_dict)
            
            # Count citations
            for bib_dict in bib_abstracts_list:
                for citation_id in bib_dict.keys():
                    self.citation_stats[citation_id] = self.citation_stats.get(citation_id, 0) + 1

        self.logger.info(f"Processed {len(self.bib_abstracts_dict)} unique citations")

    def create_vector_store(self) -> None:
        """Create vector store"""
        self.logger.info("Creating vector store...")
        
        # Prepare data
        abstracts = []
        self.citation_ids = []
        
        for cid, abstract in self.bib_abstracts_dict.items():
            abstracts.append(abstract)
            self.citation_ids.append(cid)
        
        # Calculate embedding vectors
        try:
            self.abstract_embeddings = self.model.encode(abstracts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = self.abstract_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.abstract_embeddings.astype('float32'))
            
            self.logger.info(f"Successfully created vector store with {len(abstracts)} vectors")
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def save_processed_data(self, save_dir: str) -> None:
        """
        Save processed data
        Args:
            save_dir: Save directory
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save basic data
            with open(f"{save_dir}/bib_abstracts.json", 'w') as f:
                json.dump(self.bib_abstracts_dict, f)
            
            with open(f"{save_dir}/bib_titles.json", 'w') as f:
                json.dump(self.bib_titles_dict, f)
            
            with open(f"{save_dir}/citation_stats.json", 'w') as f:
                json.dump(self.citation_stats, f)
            
            # Save vector retrieval related data
            if self.index is not None:
                faiss.write_index(self.index, f"{save_dir}/abstract_index.faiss")
                np.save(f"{save_dir}/citation_ids.npy", np.array(self.citation_ids))
            
            self.logger.info(f"Successfully saved processed data to {save_dir}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def load_processed_data(self, save_dir: str) -> None:
        """
        Load processed data
        Args:
            save_dir: Data directory
        """
        try:
            # Load basic data
            with open(f"{save_dir}/bib_abstracts.json", 'r') as f:
                self.bib_abstracts_dict = json.load(f)
            
            with open(f"{save_dir}/bib_titles.json", 'r') as f:
                self.bib_titles_dict = json.load(f)
            
            with open(f"{save_dir}/citation_stats.json", 'r') as f:
                self.citation_stats = json.load(f)
            
            # Load vector retrieval related data
            if os.path.exists(f"{save_dir}/abstract_index.faiss"):
                self.index = faiss.read_index(f"{save_dir}/abstract_index.faiss")
                self.citation_ids = np.load(f"{save_dir}/citation_ids.npy").tolist()
            
            self.logger.info(f"Successfully loaded processed data from {save_dir}")
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            raise

    def get_abstracts_by_ids(self, citation_ids: List[str]) -> Dict[str, str]:
        """
        Get abstracts for specified citation IDs
        Args:
            citation_ids: List of citation IDs
        Returns:
            Dictionary mapping IDs to abstracts
        """
        return {cid: self.bib_abstracts_dict.get(cid, "") for cid in citation_ids}

    def get_titles_by_ids(self, citation_ids: List[str]) -> Dict[str, str]:
        """
        Get titles for specified citation IDs
        Args:
            citation_ids: List of citation IDs
        Returns:
            Dictionary mapping IDs to titles
        """
        return {cid: self.bib_titles_dict.get(cid, "") for cid in citation_ids}

    def search_similar_abstracts(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for abstracts most similar to the query
        Args:
            query: Query text
            k: Number of results to return
        Returns:
            List of similar papers containing ID, title, abstract and similarity score
        """
        if self.index is None:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
        
        try:
            # Encode query
            query_vector = self.model.encode([query])
            
            # Search for most similar vectors
            distances, indices = self.index.search(query_vector.astype('float32'), k)
            
            # Organize results
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
        Get citation count for specified citation ID
        Args:
            citation_id: Citation ID
        Returns:
            Citation count, or None if not found
        """
        return self.citation_stats.get(citation_id)
    
    def search_in_papers(self, query: str, papers: List[Dict], k: int = 5) -> List[Dict]:
        """
        Search for similar papers within a specified paper collection
        Args:
            query: Query text
            papers: List of papers [{citation_id, title, abstract}, ...]
            k: Number of results to return
        Returns:
            List of similar papers
        """
        try:
            if not papers:
                self.logger.warning("Empty paper list provided for search")
                return []

            # Encode query text
            query_vector = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
            
            # Encode all paper titles and abstracts
            paper_texts = [f"{p['title']}. {p['abstract']}" for p in papers]
            paper_vectors = self.model.encode(paper_texts, convert_to_numpy=True)
            
            # Calculate similarity
            similarities = cosine_similarity(query_vector, paper_vectors)[0]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Organize results
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