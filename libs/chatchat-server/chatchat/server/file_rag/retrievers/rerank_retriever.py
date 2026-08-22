"""
重排序检索器 - 先用向量检索召回，再用 rerank 模型重排序
"""
from __future__ import annotations
from typing import List
from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService


class RerankRetrieverService(BaseRetrieverService):
    """重排序检索器"""
    
    def do_init(self, retriever: BaseRetriever = None, top_k: int = 5, 
                reranker=None, rerank_top_k: int = 20):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k  # 召回候选文档数
    
    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: float,
        reranker=None,
        rerank_top_k: int = 20,
    ):
        """
        创建重排序检索器
        
        Args:
            vectorstore: 向量存储
            top_k: 最终返回文档数量
            score_threshold: 分数阈值
            reranker: 重排序模型（如果为 None 则降级为普通向量检索）
            rerank_top_k: 召回候选文档数量，用于重排序
        """
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": rerank_top_k},
        )
        return RerankRetrieverService(
            retriever=retriever, 
            top_k=top_k, 
            reranker=reranker,
            rerank_top_k=rerank_top_k
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """先召回候选文档，再重排序"""
        # 第一阶段：向量检索召回
        docs = self.retriever.get_relevant_documents(query)
        
        # 第二阶段：重排序
        if self.reranker and len(docs) > 0:
            docs = self.reranker.compress_documents(documents=docs, query=query)
        
        return docs[:self.top_k]
