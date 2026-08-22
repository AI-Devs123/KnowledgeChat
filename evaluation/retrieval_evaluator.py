"""
RAG 检索评估框架
支持评估驱动开发，对比不同检索策略的效果
"""
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.file_rag.utils import get_Retriever


class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self, kb_name: str, test_queries_path: str = "evaluation/test_queries.json"):
        self.kb_name = kb_name
        self.kb_service = KBServiceFactory.get_service_by_name(kb_name)
        
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            self.test_queries = json.load(f)
    
    def evaluate_retriever(self, retriever_type: str, top_k: int = 5, 
                          score_threshold: float = 0.5) -> Dict[str, Any]:
        """
        评估特定检索器
        
        Args:
            retriever_type: 检索器类型 ('vectorstore', 'ensemble', 'milvusvectorstore')
            top_k: 返回文档数
            score_threshold: 分数阈值
        """
        results = {
            'retriever_type': retriever_type,
            'top_k': top_k,
            'score_threshold': score_threshold,
            'queries': [],
            'metrics': {}
        }
        
        total_precision = 0
        total_recall = 0
        total_mrr = 0
        total_time = 0
        
        for test_case in self.test_queries:
            query = test_case['query']
            expected_docs = test_case['expected_docs']
            
            # 执行检索
            start_time = time.time()
            docs = self._search_with_retriever(query, retriever_type, top_k, score_threshold)
            elapsed_time = time.time() - start_time
            
            # 计算指标
            retrieved_files = [doc.metadata.get('source', '') for doc in docs]
            retrieved_files = [Path(f).name for f in retrieved_files]
            
            precision = self._calculate_precision(retrieved_files, expected_docs)
            recall = self._calculate_recall(retrieved_files, expected_docs)
            mrr = self._calculate_mrr(retrieved_files, expected_docs)
            
            total_precision += precision
            total_recall += recall
            total_mrr += mrr
            total_time += elapsed_time
            
            results['queries'].append({
                'query': query,
                'category': test_case['category'],
                'expected_docs': expected_docs,
                'retrieved_docs': retrieved_files[:5],
                'scores': [getattr(doc, 'relevance_score', getattr(doc.metadata, 'score', 0)) 
                          for doc in docs[:5]],
                'precision': precision,
                'recall': recall,
                'mrr': mrr,
                'time_ms': elapsed_time * 1000
            })
        
        # 计算平均指标
        num_queries = len(self.test_queries)
        results['metrics'] = {
            'avg_precision': total_precision / num_queries,
            'avg_recall': total_recall / num_queries,
            'avg_mrr': total_mrr / num_queries,
            'avg_time_ms': (total_time / num_queries) * 1000,
            'total_queries': num_queries
        }
        
        # 计算 F1
        p = results['metrics']['avg_precision']
        r = results['metrics']['avg_recall']
        results['metrics']['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        return results
    
    def _search_with_retriever(self, query: str, retriever_type: str, 
                              top_k: int, score_threshold: float):
        """使用指定检索器进行检索"""
        with self.kb_service.load_vector_store().acquire() as vs:
            retriever = get_Retriever(retriever_type).from_vectorstore(
                vs, top_k=top_k, score_threshold=score_threshold
            )
            return retriever.get_relevant_documents(query)
    
    def _calculate_precision(self, retrieved: List[str], expected: List[str]) -> float:
        """计算精确率：检索到的相关文档 / 检索到的文档总数"""
        if not retrieved:
            return 0.0
        relevant = sum(1 for doc in retrieved if any(exp in doc for exp in expected))
        return relevant / len(retrieved)
    
    def _calculate_recall(self, retrieved: List[str], expected: List[str]) -> float:
        """计算召回率：检索到的相关文档 / 期望的文档总数"""
        if not expected:
            return 0.0
        relevant = sum(1 for exp in expected if any(exp in doc for doc in retrieved))
        return relevant / len(expected)
    
    def _calculate_mrr(self, retrieved: List[str], expected: List[str]) -> float:
        """计算平均倒数排名 (MRR)"""
        for i, doc in enumerate(retrieved, 1):
            if any(exp in doc for exp in expected):
                return 1.0 / i
        return 0.0
    
    def compare_retrievers(self, retriever_types: List[str], 
                          top_k: int = 5, score_threshold: float = 0.5) -> Dict:
        """对比多个检索器的性能"""
        comparison = {
            'config': {
                'kb_name': self.kb_name,
                'top_k': top_k,
                'score_threshold': score_threshold
            },
            'results': []
        }
        
        for retriever_type in retriever_types:
            try:
                result = self.evaluate_retriever(retriever_type, top_k, score_threshold)
                comparison['results'].append(result)
            except Exception as e:
                print(f"Error evaluating {retriever_type}: {e}")
        
        return comparison
    
    def save_results(self, results: Dict, output_path: str):
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
    
    def print_summary(self, comparison: Dict):
        """打印对比摘要"""
        print("\n" + "="*80)
        print("检索器性能对比摘要")
        print("="*80)
        print(f"知识库: {comparison['config']['kb_name']}")
        print(f"配置: top_k={comparison['config']['top_k']}, "
              f"score_threshold={comparison['config']['score_threshold']}")
        print("-"*80)
        
        for result in comparison['results']:
            metrics = result['metrics']
            print(f"\n检索器: {result['retriever_type']}")
            print(f"  精确率 (Precision): {metrics['avg_precision']:.3f}")
            print(f"  召回率 (Recall):    {metrics['avg_recall']:.3f}")
            print(f"  F1 分数:           {metrics['f1_score']:.3f}")
            print(f"  MRR:              {metrics['avg_mrr']:.3f}")
            print(f"  平均响应时间:      {metrics['avg_time_ms']:.2f} ms")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # 示例用法
    evaluator = RetrievalEvaluator(kb_name="samples")
    
    # 对比不同检索器
    comparison = evaluator.compare_retrievers(
        retriever_types=['vectorstore', 'ensemble'],
        top_k=5,
        score_threshold=0.5
    )
    
    # 打印摘要
    evaluator.print_summary(comparison)
    
    # 保存详细结果
    evaluator.save_results(comparison, "evaluation/retrieval_comparison.json")
