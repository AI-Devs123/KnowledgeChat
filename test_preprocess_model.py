"""
意图识别模板对比测试
对比旧版（单级0/1分类）vs 新版（二级路由）的效果
包含响应延迟和Token消耗统计
"""

import sys
import json
import time
from typing import Dict, Any, Tuple
sys.path.append("libs/chatchat-server")

from chatchat.server.utils import get_ChatOpenAI
from chatchat.settings import Settings


def extract_content(llm_response) -> str:
    """从LLM响应中提取文本内容"""
    if hasattr(llm_response, 'content'):
        # AIMessage对象
        return llm_response.content
    elif isinstance(llm_response, str):
        # 字符串
        return llm_response
    else:
        # 其他类型，转为字符串
        return str(llm_response)


def extract_token_usage(llm_response) -> Dict[str, int]:
    """从LLM响应中提取Token使用情况"""
    token_info = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    try:
        # 尝试从response_metadata中提取
        if hasattr(llm_response, 'response_metadata'):
            metadata = llm_response.response_metadata
            if 'token_usage' in metadata:
                usage = metadata['token_usage']
                token_info["prompt_tokens"] = usage.get('prompt_tokens', 0)
                token_info["completion_tokens"] = usage.get('completion_tokens', 0)
                token_info["total_tokens"] = usage.get('total_tokens', 0)
        # 或者从usage_metadata中提取
        elif hasattr(llm_response, 'usage_metadata'):
            usage = llm_response.usage_metadata
            token_info["prompt_tokens"] = usage.get('input_tokens', 0)
            token_info["completion_tokens"] = usage.get('output_tokens', 0)
            token_info["total_tokens"] = usage.get('total_tokens', 0)
        
        # 如果还是获取不到token信息（都是0），尝试其他属性
        if token_info["total_tokens"] == 0:
            # 尝试直接的 usage 属性
            if hasattr(llm_response, 'usage'):
                usage = llm_response.usage
                if isinstance(usage, dict):
                    token_info["prompt_tokens"] = usage.get('prompt_tokens', 0)
                    token_info["completion_tokens"] = usage.get('completion_tokens', 0)
                    token_info["total_tokens"] = usage.get('total_tokens', 0)
            
            # 尝试 additional_kwargs
            if token_info["total_tokens"] == 0 and hasattr(llm_response, 'additional_kwargs'):
                kwargs = llm_response.additional_kwargs
                if 'usage' in kwargs:
                    usage = kwargs['usage']
                    token_info["prompt_tokens"] = usage.get('prompt_tokens', 0)
                    token_info["completion_tokens"] = usage.get('completion_tokens', 0)
                    token_info["total_tokens"] = usage.get('total_tokens', 0)
    except Exception as e:
        pass  # 静默处理，使用估算
    
    # 如果还是获取不到，使用改进的估算方法
    if token_info["total_tokens"] == 0:
        content = extract_content(llm_response)
        
        # 改进的估算方法
        # 1. 统计中文字符数（中文约1.5-2 tokens/字符）
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        # 2. 统计英文单词数（英文约1.3 tokens/单词）
        english_words = len([w for w in content.split() if any(c.isalpha() for c in w)])
        # 3. 统计数字和符号（约1 token/字符）
        other_chars = len(content) - chinese_chars - sum(len(w) for w in content.split() if any(c.isalpha() for c in w))
        
        # 计算completion tokens
        estimated_completion = int(chinese_chars * 1.7 + english_words * 1.3 + other_chars * 0.5)
        
        # 估算prompt tokens（根据提示词长度）
        # 旧版default模板约100 tokens，新版router_main约300 tokens
        estimated_prompt = 150  # 保守估计
        
        token_info["completion_tokens"] = max(estimated_completion, 10)  # 至少10 tokens
        token_info["prompt_tokens"] = estimated_prompt
        token_info["total_tokens"] = token_info["completion_tokens"] + token_info["prompt_tokens"]
        token_info["estimated"] = True  # 标记为估算值
    
    return token_info

# 测试用例
TEST_CASES = [
    {
        "id": 1,
        "question": "查询今天北京地区的电子产品销售额",
        "expected_main": "data_query",
        "expected_sub": "sales_query",
        "description": "数据查询 - 带时间、地点、类别参数"
    },
    {
        "id": 2,
        "question": "什么是机器学习？它有哪些应用场景？",
        "expected_main": "knowledge_qa",
        "expected_sub": None,
        "description": "知识问答 - 概念解释"
    },
    {
        "id": 3,
        "question": "帮我搜索关于GPT-4的最新论文",
        "expected_main": "tool_execution",
        "expected_sub": "academic_search",
        "description": "工具执行 - 学术搜索"
    },
    {
        "id": 4,
        "question": "计算 (123 + 456) * 789 的结果",
        "expected_main": "tool_execution",
        "expected_sub": "math_calculate",
        "description": "工具执行 - 数学计算"
    },
    {
        "id": 5,
        "question": "生成一张日落海滩的图片",
        "expected_main": "tool_execution",
        "expected_sub": "text2image",
        "description": "工具执行 - 图像生成"
    },
    {
        "id": 6,
        "question": "北京现在天气如何",
        "expected_main": "data_query",
        "expected_sub": ["weather", "realtime_query"],  # 天气查询的不同表达
        "description": "数据查询 - 实时天气"
    },
    {
        "id": 7,
        "question": "统计本月销量前10名的产品",
        "expected_main": "data_query",
        "expected_sub": ["sales_query", "product_query"],  # 两种理解都合理
        "description": "数据查询 - 统计排名"
    },
    {
        "id": 8,
        "question": "Python如何读取CSV文件",
        "expected_main": "knowledge_qa",
        "expected_sub": None,
        "description": "知识问答 - 编程问题"
    },
    {
        "id": 9,
        "question": "附近有哪些咖啡店",
        "expected_main": "tool_execution",
        "expected_sub": "poi_search",
        "description": "工具执行 - 地点搜索"
    },
    {
        "id": 10,
        "question": "查看用户12345的购买历史",
        "expected_main": "data_query",
        "expected_sub": "user_profile",
        "description": "数据查询 - 用户信息"
    }
]


class IntentComparisonTester:
    """意图识别对比测试器"""
    
    def __init__(self):
        self.llm_old = get_ChatOpenAI(temperature=0.05, max_tokens=50)
        self.llm_new = get_ChatOpenAI(temperature=0.05, max_tokens=50)
        self.llm_sub = get_ChatOpenAI(temperature=0.1, max_tokens=256)
        
    def test_old_version(self, question: str) -> Dict[str, Any]:
        """测试旧版意图识别（0/1分类）"""
        start_time = time.time()
        
        try:
            prompt = Settings.prompt_settings.preprocess_model["default"]
            result = self.llm_old.invoke(prompt + question)
            
            latency = time.time() - start_time
            result_clean = extract_content(result).strip()
            token_usage = extract_token_usage(result)
            
            # 转换0/1为可读标签
            if "1" in result_clean:
                classification = "需要工具 (1)"
            elif "0" in result_clean:
                classification = "不需要工具 (0)"
            else:
                classification = f"未知 ({result_clean})"
            
            return {
                "success": True,
                "classification": classification,
                "raw_output": result_clean,
                "latency_ms": round(latency * 1000, 2),
                "token_usage": token_usage,
                "error": None
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "classification": "错误",
                "raw_output": None,
                "latency_ms": round(latency * 1000, 2),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "error": str(e)
            }
    
    def test_new_version(self, question: str) -> Dict[str, Any]:
        """测试新版意图识别（二级路由）"""
        start_time = time.time()
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        try:
            # 第一级：主路由
            main_start = time.time()
            main_prompt = Settings.prompt_settings.preprocess_model["router_main"]
            main_result = self.llm_new.invoke(main_prompt + question)
            main_latency = time.time() - main_start
            
            main_result_str = extract_content(main_result).strip()
            main_intent = main_result_str.lower()
            main_tokens = extract_token_usage(main_result)
            
            # 累加token
            for key in total_tokens:
                total_tokens[key] += main_tokens[key]
            
            # 验证主分类
            valid_intents = ["knowledge_qa", "data_query", "tool_execution"]
            if main_intent not in valid_intents:
                main_intent = "unknown"
            
            # 第二级：子路由（仅对data_query和tool_execution）
            sub_result = None
            parameters = {}
            sub_latency = 0
            
            if main_intent == "data_query":
                sub_start = time.time()
                sub_result = self._route_data_query(question)
                sub_latency = time.time() - sub_start
                
                if sub_result and "token_usage" in sub_result:
                    for key in total_tokens:
                        total_tokens[key] += sub_result["token_usage"][key]
                        
            elif main_intent == "tool_execution":
                sub_start = time.time()
                sub_result = self._route_tool_execution(question)
                sub_latency = time.time() - sub_start
                
                if sub_result and "token_usage" in sub_result:
                    for key in total_tokens:
                        total_tokens[key] += sub_result["token_usage"][key]
            
            if sub_result:
                parameters = sub_result.get("parameters", {})
            
            total_latency = time.time() - start_time
            
            return {
                "success": True,
                "main_intent": main_intent,
                "sub_intent": sub_result.get("intent") if sub_result else None,
                "parameters": parameters,
                "raw_main": main_result_str,
                "raw_sub": sub_result.get("raw") if sub_result else None,
                "latency_ms": round(total_latency * 1000, 2),
                "latency_breakdown": {
                    "main_route_ms": round(main_latency * 1000, 2),
                    "sub_route_ms": round(sub_latency * 1000, 2)
                },
                "token_usage": total_tokens,
                "error": None
            }
        except Exception as e:
            total_latency = time.time() - start_time
            return {
                "success": False,
                "main_intent": "error",
                "sub_intent": None,
                "parameters": {},
                "latency_ms": round(total_latency * 1000, 2),
                "token_usage": total_tokens,
                "error": str(e)
            }
    
    def _route_data_query(self, question: str) -> Dict[str, Any]:
        """数据查询子路由"""
        try:
            # 判断是数据库查询还是实时查询
            if any(kw in question for kw in ["天气", "股票", "新闻"]):
                template_key = "realtime_intent"
            else:
                template_key = "database_intent"
            
            prompt = Settings.prompt_settings.intent_data_query[template_key]
            result = self.llm_sub.invoke(prompt + question)
            result_str = extract_content(result)
            token_usage = extract_token_usage(result)
            
            # 清理JSON格式
            result_clean = result_str.strip()
            if "```json" in result_clean:
                result_clean = result_clean.split("```json")[1].split("```")[0]
            elif "```" in result_clean:
                result_clean = result_clean.split("```")[1].split("```")[0]
            
            # 解析JSON
            try:
                parsed = json.loads(result_clean)
                return {
                    "intent": parsed.get("intent"),
                    "parameters": parsed.get("parameters", {}),
                    "raw": result_str,
                    "token_usage": token_usage
                }
            except json.JSONDecodeError:
                # JSON解析失败，尝试修复
                try:
                    from json_repair import repair_json
                    parsed = repair_json(result_clean)
                    return {
                        "intent": parsed.get("intent", "parse_error"),
                        "parameters": parsed.get("parameters", {}),
                        "raw": result_str,
                        "token_usage": token_usage
                    }
                except:
                    return {
                        "intent": "parse_error",
                        "parameters": {},
                        "raw": result_str,
                        "token_usage": token_usage
                    }
        except Exception as e:
            return {
                "intent": "error",
                "parameters": {},
                "raw": str(e),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    def _route_tool_execution(self, question: str) -> Dict[str, Any]:
        """工具执行子路由"""
        try:
            # 根据关键词选择子模板
            if any(kw in question for kw in ["搜索", "查找", "检索", "论文"]):
                template_key = "search_intent"
            elif any(kw in question for kw in ["计算", "算", "公式"]):
                template_key = "compute_intent"
            elif any(kw in question for kw in ["生成", "画", "图片", "图像"]):
                template_key = "multimodal_intent"
            elif any(kw in question for kw in ["附近", "地图", "位置"]):
                template_key = "location_intent"
            else:
                template_key = "search_intent"
            
            prompt = Settings.prompt_settings.intent_tool_execution[template_key]
            result = self.llm_sub.invoke(prompt + question)
            result_str = extract_content(result)
            token_usage = extract_token_usage(result)
            
            # 清理和解析JSON
            result_clean = result_str.strip()
            if "```json" in result_clean:
                result_clean = result_clean.split("```json")[1].split("```")[0]
            elif "```" in result_clean:
                result_clean = result_clean.split("```")[1].split("```")[0]
            
            try:
                parsed = json.loads(result_clean)
                return {
                    "intent": parsed.get("intent"),
                    "parameters": parsed.get("parameters", {}),
                    "raw": result_str,
                    "token_usage": token_usage
                }
            except json.JSONDecodeError:
                from json_repair import repair_json
                parsed = repair_json(result_clean)
                return {
                    "intent": parsed.get("intent", "parse_error"),
                    "parameters": parsed.get("parameters", {}),
                    "raw": result_str,
                    "token_usage": token_usage
                }
        except Exception as e:
            return {
                "intent": "error",
                "parameters": {},
                "raw": str(e),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    def run_comparison(self, test_cases=None):
        """运行对比测试"""
        if test_cases is None:
            test_cases = TEST_CASES
        
        print("=" * 100)
        print("意图识别模板对比测试")
        print("=" * 100)
        print()
        
        old_correct = 0
        new_correct = 0
        old_latencies = []
        old_tokens = []
        new_latencies = []
        new_tokens = []
        
        # 新增：主分类和参数提取统计
        main_classification_correct = 0  # 主分类正确数
        parameter_extraction_stats = {
            "total_cases_with_params": 0,  # 应该提取参数的用例总数
            "successful_extractions": 0,    # 成功提取参数的次数
            "extracted_params": {
                "time_range": 0,
                "location": 0,
                "category": 0,
                "metric": 0,
                "query": 0,
                "expression": 0,
                "prompt": 0,
                "keyword": 0
            }
        }
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*100}")
            print(f"测试 {case['id']}: {case['description']}")
            print(f"{'='*100}")
            print(f"问题: {case['question']}")
            print(f"期望分类: {case['expected_main']}")
            if case.get('expected_sub'):
                print(f"期望子类: {case['expected_sub']}")
            print()
            
            # 旧版测试
            print("【旧版结果】单级0/1分类")
            print("-" * 100)
            old_result = self.test_old_version(case['question'])
            if old_result['success']:
                print(f"分类结果: {old_result['classification']}")
                print(f"原始输出: {old_result['raw_output']}")
                print(f"⏱️  响应延迟: {old_result['latency_ms']} ms")
                token_usage = old_result['token_usage']
                is_estimated = token_usage.get('estimated', False)
                est_mark = " (估算)" if is_estimated else ""
                print(f"🎫 Token消耗: {token_usage['total_tokens']} tokens{est_mark} "
                      f"(prompt: {token_usage['prompt_tokens']}, "
                      f"completion: {token_usage['completion_tokens']})")
                
                # 收集统计数据
                old_latencies.append(old_result['latency_ms'])
                old_tokens.append(old_result['token_usage']['total_tokens'])
                
                # 简单的正确性判断
                expected_tool = case['expected_main'] in ['data_query', 'tool_execution']
                result_tool = "1" in old_result['raw_output']
                if expected_tool == result_tool:
                    old_correct += 1
                    print("✅ 分类正确")
                else:
                    print("❌ 分类错误")
            else:
                print(f"❌ 错误: {old_result['error']}")
            
            print()
            
            # 新版测试
            print("【新版结果】二级路由")
            print("-" * 100)
            new_result = self.test_new_version(case['question'])
            if new_result['success']:
                print(f"主分类: {new_result['main_intent']}")
                if new_result['sub_intent']:
                    print(f"子分类: {new_result['sub_intent']}")
                if new_result['parameters']:
                    print(f"提取参数:")
                    for k, v in new_result['parameters'].items():
                        print(f"  - {k}: {v}")
                
                print(f"⏱️  响应延迟: {new_result['latency_ms']} ms "
                      f"(一级: {new_result['latency_breakdown']['main_route_ms']} ms, "
                      f"二级: {new_result['latency_breakdown']['sub_route_ms']} ms)")
                token_usage = new_result['token_usage']
                is_estimated = token_usage.get('estimated', False)
                est_mark = " (估算)" if is_estimated else ""
                print(f"🎫 Token消耗: {token_usage['total_tokens']} tokens{est_mark} "
                      f"(prompt: {token_usage['prompt_tokens']}, "
                      f"completion: {token_usage['completion_tokens']})")
                
                # 收集统计数据
                new_latencies.append(new_result['latency_ms'])
                new_tokens.append(new_result['token_usage']['total_tokens'])
                
                # 正确性判断
                main_correct = new_result['main_intent'] == case['expected_main']
                sub_correct = True
                
                # 统计主分类准确率
                if main_correct:
                    main_classification_correct += 1
                
                # 统计参数提取能力
                if case['expected_main'] in ['data_query', 'tool_execution']:
                    parameter_extraction_stats["total_cases_with_params"] += 1
                    
                    # 检查是否成功提取了参数
                    if new_result['parameters'] and len(new_result['parameters']) > 0:
                        has_valid_param = False
                        for key, value in new_result['parameters'].items():
                            if value and value.strip():  # 非空参数
                                has_valid_param = True
                                # 统计各类参数的提取次数
                                if key in parameter_extraction_stats["extracted_params"]:
                                    parameter_extraction_stats["extracted_params"][key] += 1
                        
                        if has_valid_param:
                            parameter_extraction_stats["successful_extractions"] += 1
                
                # 支持多个期望子分类
                if case.get('expected_sub') and new_result['sub_intent']:
                    expected_subs = case['expected_sub']
                    if isinstance(expected_subs, str):
                        expected_subs = [expected_subs]
                    
                    # 只要匹配任意一个期望子分类就算正确
                    sub_correct = any(exp_sub in new_result['sub_intent'] for exp_sub in expected_subs)
                
                if main_correct and sub_correct:
                    new_correct += 1
                    print("✅ 分类正确")
                else:
                    if not main_correct:
                        print(f"❌ 主分类错误（期望: {case['expected_main']}, 实际: {new_result['main_intent']}）")
                    if not sub_correct:
                        print(f"⚠️  子分类不匹配（期望: {case['expected_sub']}, 实际: {new_result['sub_intent']}）")
            else:
                print(f"❌ 错误: {new_result['error']}")
            
            print()
        
        # 统计结果
        print("\n" + "=" * 100)
        print("测试结果统计")
        print("=" * 100)
        total = len(test_cases)
        print(f"总测试数: {total}")
        
        print(f"\n📊 整体准确率对比:")
        print(f"  旧版正确: {old_correct}/{total} ({old_correct/total*100:.1f}%)")
        print(f"  新版正确: {new_correct}/{total} ({new_correct/total*100:.1f}%)")
        
        # 新增：主分类准确率
        print(f"\n🎯 主分类准确率 (最重要指标):")
        main_accuracy = main_classification_correct / total * 100
        print(f"  新版主分类: {main_classification_correct}/{total} ({main_accuracy:.1f}%)")
        if main_accuracy >= 90:
            print(f"  评价: ✅ 优秀 (≥90%)")
        elif main_accuracy >= 80:
            print(f"  评价: ✅ 良好 (≥80%)")
        elif main_accuracy >= 70:
            print(f"  评价: ⚠️ 及格 (≥70%)")
        else:
            print(f"  评价: ❌ 需要优化 (<70%)")
        
        # 新增：参数提取能力
        print(f"\n📦 参数提取能力 (核心价值):")
        if parameter_extraction_stats["total_cases_with_params"] > 0:
            param_success_rate = (parameter_extraction_stats["successful_extractions"] / 
                                 parameter_extraction_stats["total_cases_with_params"] * 100)
            print(f"  需要提取参数的用例: {parameter_extraction_stats['total_cases_with_params']} 个")
            print(f"  成功提取参数: {parameter_extraction_stats['successful_extractions']} 个 ({param_success_rate:.1f}%)")
            
            # 显示各类参数的提取情况
            print(f"\n  各类参数提取统计:")
            for param_name, count in parameter_extraction_stats["extracted_params"].items():
                if count > 0:
                    print(f"    - {param_name}: {count} 次")
            
            if param_success_rate >= 90:
                print(f"\n  评价: ✅ 优秀 - 参数提取能力强")
            elif param_success_rate >= 70:
                print(f"\n  评价: ✅ 良好 - 大部分参数能正确提取")
            elif param_success_rate >= 50:
                print(f"\n  评价: ⚠️ 及格 - 参数提取需要改进")
            else:
                print(f"\n  评价: ❌ 较弱 - 参数提取能力不足")
        else:
            print(f"  无需要提取参数的测试用例")
        
        # 计算平均延迟和Token消耗
        avg_old_latency = 0
        avg_old_tokens = 0
        avg_new_latency = 0
        avg_new_tokens = 0
        
        if old_latencies:
            avg_old_latency = sum(old_latencies) / len(old_latencies)
            avg_old_tokens = sum(old_tokens) / len(old_tokens)
            print(f"\n⏱️  平均响应延迟:")
            print(f"  旧版: {avg_old_latency:.2f} ms")
            
            print(f"\n🎫 平均Token消耗 (估算值):")
            print(f"  旧版: {avg_old_tokens:.1f} tokens")
            print(f"  ⚠️  注意: 由于模型不返回token信息,使用估算值")
        else:
            print(f"\n⏱️  平均响应延迟:")
            print(f"  旧版: 无数据")
            print(f"\n🎫 平均Token消耗:")
            print(f"  旧版: 无数据")
        
        if new_latencies:
            avg_new_latency = sum(new_latencies) / len(new_latencies)
            avg_new_tokens = sum(new_tokens) / len(new_tokens)
            
            if old_latencies and avg_old_latency > 0:
                latency_increase = avg_new_latency - avg_old_latency
                latency_percent = (avg_new_latency / avg_old_latency - 1) * 100
                print(f"  新版: {avg_new_latency:.2f} ms (增加 {latency_increase:.2f} ms, +{latency_percent:.1f}%)")
            else:
                print(f"  新版: {avg_new_latency:.2f} ms")
            
            if old_tokens and avg_old_tokens > 0:
                token_increase = avg_new_tokens - avg_old_tokens
                token_percent = (avg_new_tokens / avg_old_tokens - 1) * 100
                print(f"  新版: {avg_new_tokens:.1f} tokens (增加 {token_increase:.1f} tokens, +{token_percent:.1f}%)")
            else:
                print(f"  新版: {avg_new_tokens:.1f} tokens")
        else:
            print(f"  新版: 无数据")
        
        print()
        
        if new_correct > old_correct:
            improvement = (new_correct - old_correct) / total * 100
            print(f"✨ 准确率提升: +{improvement:.1f}%")
        elif new_correct == old_correct:
            print("⚖️  新旧版准确率相同")
        else:
            decline = (old_correct - new_correct) / total * 100
            print(f"⚠️  准确率下降: -{decline:.1f}%")
        
        print("=" * 100)
        
        return {
            "total": total,
            "old_correct": old_correct,
            "new_correct": new_correct,
            "old_accuracy": old_correct / total if total > 0 else 0,
            "new_accuracy": new_correct / total if total > 0 else 0,
            "main_classification_correct": main_classification_correct,
            "main_classification_accuracy": main_classification_correct / total if total > 0 else 0,
            "parameter_extraction": parameter_extraction_stats,
            "avg_old_latency_ms": avg_old_latency,
            "avg_new_latency_ms": avg_new_latency,
            "avg_old_tokens": avg_old_tokens,
            "avg_new_tokens": avg_new_tokens
        }


def main():
    """主函数"""
    print("\n正在初始化测试环境...")
    print("加载配置和模型...\n")
    
    tester = IntentComparisonTester()
    
    # 运行完整对比测试
    results = tester.run_comparison()
    
    # 保存结果到文件
    try:
        with open("intent_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "test_cases": TEST_CASES,
                "summary": results,
                "timestamp": "2024-08"
            }, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: intent_comparison_results.json")
    except Exception as e:
        print(f"\n保存结果失败: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
