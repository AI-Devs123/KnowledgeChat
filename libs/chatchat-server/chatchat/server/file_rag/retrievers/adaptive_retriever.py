"""
自适应检索器 - 根据查询类型自动选择检索策略
"""
from __future__ import annotations
from typing import List
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
import re

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService
from chatchat.server.file_rag.retrievers.vectorstore import VectorstoreRetrieverService
from chatchat.server.file_rag.retrievers.ensemble import EnsembleRetrieverService


class AdaptiveRetrieverService(BaseRetrieverService):
    """自适应检索器，根据查询特征选择最佳策略"""
    
    def do_init(self, top_k: int = 5):
        self.vs = None
        self.top_k = top_k
        self.vectorstore = None
        self.score_threshold = 0.5
    
    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: float,
    ):
        service = AdaptiveRetrieverService(top_k=top_k)
        service.vectorstore = vectorstore
        service.score_threshold = score_threshold
        return service
    
    def _analyze_query_type(self, query: str) -> str:
        """
        分析查询类型
        
        Returns:
            'keyword': 关键词查询（适合 BM25）
            'semantic': 语义查询（适合向量检索）
            'hybrid': 混合查询（适合 ensemble）
        """
        # 短查询且包含明确关键词 -> keyword
        if len(query) < 10 and any(kw in query for kw in ['如何', '什么', '怎么']):
            return 'keyword'
        
        # 长句子且语义丰富 -> semantic
        if len(query) > 20 and not re.search(r'[、，,]', query):
            return 'semantic'
        
        # 默认使用混合
        return 'hybrid'
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """根据查询类型选择检索策略"""
        query_type = self._analyze_query_type(query)
        
        if query_type == 'semantic':
            # 纯向量检索
            retriever = VectorstoreRetrieverService.from_vectorstore(
                self.vectorstore, self.top_k, self.score_threshold
            )
        else:
            # 混合检索（keyword 和 hybrid 都用这个）
            retriever = EnsembleRetrieverService.from_vectorstore(
                self.vectorstore, self.top_k, self.score_threshold
            )
        
        return retriever.get_relevant_documents(query)
