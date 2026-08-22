"""
混合融合检索器 - 使用 Reciprocal Rank Fusion (RRF) 算法
相比简单的加权平均，RRF 对排名更鲁棒
"""
from __future__ import annotations
from typing import List
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import jieba

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService


class HybridFusionRetrieverService(BaseRetrieverService):
    """混合融合检索器，使用 RRF 算法融合多个检索结果"""
    
    def do_init(self, retriever: BaseRetriever = None, top_k: int = 5, k: int = 60):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever
        self.k = k  # RRF 参数
    
    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: float,
        k: int = 60,
    ):
        """
        创建混合融合检索器
        
        Args:
            vectorstore: 向量存储
            top_k: 返回文档数量
            score_threshold: 分数阈值
            k: RRF 参数，默认 60
        """
        service = HybridFusionRetrieverService(top_k=top_k, k=k)
        service.vectorstore = vectorstore
        service.score_threshold = score_threshold
        return service
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """使用 RRF 融合向量检索和 BM25 检索结果"""
        # 向量检索
        vector_docs = self.vectorstore.similarity_search_with_score(
            query, k=self.top_k * 2
        )
        
        # BM25 检索
        docs = list(self.vectorstore.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=jieba.lcut_for_search,
        )
        bm25_retriever.k = self.top_k * 2
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        
        # RRF 融合
        fused_scores = {}
        
        # 处理向量检索结果
        for rank, (doc, score) in enumerate(vector_docs, 1):
            doc_id = doc.page_content
            if score >= self.score_threshold:
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (self.k + rank)
        
        # 处理 BM25 检索结果
        for rank, doc in enumerate(bm25_docs, 1):
            doc_id = doc.page_content
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (self.k + rank)
        
        # 排序并返回
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终文档列表
        doc_map = {doc.page_content: doc for doc, _ in vector_docs}
        doc_map.update({doc.page_content: doc for doc in bm25_docs})
        
        result = []
        for doc_id, score in sorted_docs[:self.top_k]:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                doc.metadata['relevance_score'] = score
                result.append(doc)
        
        return result
