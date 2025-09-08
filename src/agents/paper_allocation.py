# agents/paper_allocation.py

from typing import Dict, List, Optional
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
from src.data.data_processor import DataProcessor
from rank_bm25 import BM25Okapi
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class PaperAllocationAgent:
    """Paper Allocation Agent"""
    
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
        Get reference paper pool for a specific survey
        Args:
            df: Original dataset
            title: Survey title
        Returns:
            Reference paper pool
        """
        # Find the corresponding survey
        survey = df[df['title'] == title].iloc[0]
        
        # Get all references for this survey
        paper_pool = {}
        
        # Extract information from bib_titles and bib_abstracts
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
        Allocate references for specified section using hybrid search and re-ranking pipeline
        Args:
            section_title: Section title
            survey_title: Survey title
            survey_df: Original dataset
            k: Number of related papers to return
        Returns:
            Dictionary of allocation results
        """
        try:
            # Get reference pool for this survey
            paper_pool = self.get_paper_pool(survey_df, survey_title)
            self.logger.info(f"Found {len(paper_pool)} papers in the reference pool")
            
            # Convert reference pool papers to processor-compatible format
            pool_papers = [
                {
                    'citation_id': pid,
                    'title': info['title'],
                    'abstract': info.get('abstract', '')
                }
                for pid, info in paper_pool.items()
            ]
            
            # Step A: Candidate Retrieval - Hybrid Search Pipeline
            query = f"{survey_title} {section_title}"
            
            # Get semantic search results (top 20)
            vector_results = self.processor.search_in_papers(
                query=query,
                papers=pool_papers,
                k=min(20, len(pool_papers))
            )
            
            # Get keyword search results (top 20) using BM25
            # First need to build temporary BM25 index for this paper pool
            bm25_results = self._search_bm25_in_pool(query, pool_papers, k=min(20, len(pool_papers)))
            
            # Step B: Merge & Deduplicate
            candidate_papers = self._merge_and_deduplicate(vector_results, bm25_results)
            
            # Step C: Re-rank using CrossEncoder
            if candidate_papers:
                reranked_papers = self.processor.rerank_papers(query, candidate_papers)
            else:
                reranked_papers = []
            
            # Step D: Final Selection - take top k results
            final_papers = reranked_papers[:k]
            
            result = {
                "section_title": section_title,
                "allocated_papers": final_papers
            }
            
            self.logger.info(f"Successfully allocated {len(final_papers)} papers for section: {section_title}")
            self.logger.info(f"Pipeline: {len(vector_results)} semantic + {len(bm25_results)} keyword -> {len(candidate_papers)} candidates -> {len(final_papers)} final")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in paper allocation: {str(e)}")
            return {
                "section_title": section_title, 
                "allocated_papers": []
            }

    def _search_bm25_in_pool(self, query: str, papers: List[Dict], k: int = 20) -> List[Dict]:
        """
        Perform BM25 search within a specific paper pool
        Args:
            query: Search query
            papers: List of paper dictionaries
            k: Number of results to return
        Returns:
            List of papers with BM25 scores
        """
        try:
            if not papers:
                return []
            
            # Build temporary BM25 index for this paper pool
            corpus = []
            for paper in papers:
                text = f"{paper['title']} {paper['abstract']}".strip()
                tokens = text.lower().split()
                corpus.append(tokens)
            
            # Create BM25 index
            bm25_index = BM25Okapi(corpus)
            
            # Search
            query_tokens = query.lower().split()
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                if idx < len(papers):
                    paper = papers[idx].copy()
                    paper['bm25_score'] = float(scores[idx])
                    results.append(paper)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in BM25 pool search: {str(e)}")
            return []

    def _merge_and_deduplicate(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        Merge and deduplicate results from vector and BM25 search
        Args:
            vector_results: Results from semantic search
            bm25_results: Results from BM25 search
        Returns:
            Deduplicated list of candidate papers
        """
        try:
            # Use a dictionary to deduplicate by citation_id
            merged_papers = {}
            
            # Add vector results
            for paper in vector_results:
                citation_id = paper['citation_id']
                merged_papers[citation_id] = paper.copy()
            
            # Add BM25 results, merging scores if already present
            for paper in bm25_results:
                citation_id = paper['citation_id']
                if citation_id in merged_papers:
                    # Paper already exists from vector search, add BM25 score
                    merged_papers[citation_id]['bm25_score'] = paper.get('bm25_score', 0.0)
                else:
                    # New paper from BM25 search
                    merged_papers[citation_id] = paper.copy()
            
            # Convert back to list
            candidate_papers = list(merged_papers.values())
            
            self.logger.info(f"Merged {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(candidate_papers)} unique candidates")
            
            return candidate_papers
            
        except Exception as e:
            self.logger.error(f"Error merging results: {str(e)}")
            # Return combined list without deduplication as fallback
            return vector_results + bm25_results