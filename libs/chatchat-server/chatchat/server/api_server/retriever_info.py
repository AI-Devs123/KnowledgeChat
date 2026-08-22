"""
检索器信息 API
用于查看和管理当前使用的检索器配置
"""
from fastapi import APIRouter
from pathlib import Path
import yaml
from typing import Dict

router = APIRouter(prefix="/retriever", tags=["retriever"])


@router.get("/current", summary="获取当前检索器配置")
def get_current_retriever() -> Dict:
    """
    返回当前配置的检索器类型和参数
    """
    config_file = Path("kb_settings_retriever.yaml")
    
    if not config_file.exists():
        return {
            "retriever_type": "ensemble",
            "source": "default",
            "config_exists": False,
            "description": "BM25+向量混合检索 (默认)"
        }
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        retriever_type = config.get('DEFAULT_RETRIEVER_TYPE', 'ensemble')
        
        descriptions = {
            'vectorstore': '纯向量检索 - 语义相似度匹配',
            'ensemble': 'BM25+向量混合检索 - 关键词和语义结合',
            'adaptive': '自适应检索 - 根据查询类型自动选择',
            'hybrid_fusion': 'RRF融合检索 - 倒数排名融合',
            'rerank': '重排序检索 - 向量召回+精排'
        }
        
        return {
            "retriever_type": retriever_type,
            "source": "config_file",
            "config_exists": True,
            "description": descriptions.get(retriever_type, "未知检索器"),
            "config": {
                "top_k": config.get('VECTOR_SEARCH_TOP_K', 5),
                "score_threshold": config.get('SCORE_THRESHOLD', 0.5)
            }
        }
    except Exception as e:
        return {
            "retriever_type": "ensemble",
            "source": "default",
            "config_exists": True,
            "error": str(e),
            "description": "配置读取失败，使用默认 ensemble"
        }


@router.get("/available", summary="获取所有可用的检索器")
def get_available_retrievers() -> Dict:
    """
    返回所有可用的检索器类型及说明
    """
    return {
        "retrievers": [
            {
                "type": "vectorstore",
                "name": "纯向量检索",
                "description": "仅使用语义相似度匹配，适合语义丰富的长查询",
                "pros": "语义理解好",
                "cons": "对关键词匹配较弱"
            },
            {
                "type": "ensemble",
                "name": "混合检索",
                "description": "BM25关键词检索 + 向量语义检索，各占50%权重",
                "pros": "平衡关键词和语义",
                "cons": "权重固定，不够灵活",
                "default": True
            },
            {
                "type": "adaptive",
                "name": "自适应检索",
                "description": "根据查询类型（短/长、关键词/语义）自动选择最佳策略",
                "pros": "智能适配不同查询",
                "cons": "分类准确性依赖启发式规则"
            },
            {
                "type": "hybrid_fusion",
                "name": "RRF融合检索",
                "description": "使用倒数排名融合算法合并多路召回结果",
                "pros": "融合算法更科学",
                "cons": "计算量稍大"
            },
            {
                "type": "rerank",
                "name": "重排序检索",
                "description": "先用向量检索召回候选，再用Rerank模型精排",
                "pros": "精排效果好",
                "cons": "需要额外的Rerank模型，响应较慢"
            }
        ]
    }
