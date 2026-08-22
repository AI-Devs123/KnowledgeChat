from chatchat.server.file_rag.retrievers import (
    BaseRetrieverService,
    EnsembleRetrieverService,
    VectorstoreRetrieverService,
    MilvusVectorstoreRetrieverService,
)
from chatchat.server.file_rag.retrievers.rerank_retriever import RerankRetrieverService
from chatchat.server.file_rag.retrievers.adaptive_retriever import AdaptiveRetrieverService
from chatchat.server.file_rag.retrievers.hybrid_fusion import HybridFusionRetrieverService

Retrivals = {
    "milvusvectorstore": MilvusVectorstoreRetrieverService,
    "vectorstore": VectorstoreRetrieverService,
    "ensemble": EnsembleRetrieverService,
    "rerank": RerankRetrieverService,
    "adaptive": AdaptiveRetrieverService,
    "hybrid_fusion": HybridFusionRetrieverService,
}


def get_Retriever(type: str = "vectorstore") -> BaseRetrieverService:
    return Retrivals[type]
