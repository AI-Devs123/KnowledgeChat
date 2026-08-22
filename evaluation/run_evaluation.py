"""
运行检索器评估实验
评估驱动开发 - 对比不同检索策略

标准流程:
1. 建立 baseline (ensemble 混合检索)
2. 逐个测试优化方案
3. 与 baseline 对比
4. 输出改进报告
"""
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.retrieval_evaluator import RetrievalEvaluator


def run_baseline_evaluation(evaluator, config):
    """
    步骤 1: 运行 baseline 评估
    使用原始的 ensemble 混合检索作为基准
    """
    print("\n" + "="*80)
    print("步骤 1: 建立 BASELINE - 原始混合检索 (ensemble)")
    print("="*80)
    
    baseline_result = evaluator.evaluate_retriever(
        retriever_type='ensemble',
        **config
    )
    
    # 保存 baseline
    baseline_file = "evaluation/baseline.json"
    evaluator.save_results(baseline_result, baseline_file)
    
    print(f"\n✓ Baseline 已建立并保存到: {baseline_file}")
    print(f"\n--- Baseline 性能指标 ---")
    print(f"  Precision: {baseline_result['metrics']['avg_precision']:.3f}")
    print(f"  Recall:    {baseline_result['metrics']['avg_recall']:.3f}")
    print(f"  F1 Score:  {baseline_result['metrics']['f1_score']:.3f}")
    print(f"  MRR:       {baseline_result['metrics']['avg_mrr']:.3f}")
    print(f"  响应时间:  {baseline_result['metrics']['avg_time_ms']:.2f} ms")
    
    return baseline_result


def compare_with_baseline(baseline, new_result):
    """
    计算相对于 baseline 的改进百分比
    """
    baseline_metrics = baseline['metrics']
    new_metrics = new_result['metrics']
    
    improvements = {}
    for metric in ['avg_precision', 'avg_recall', 'f1_score', 'avg_mrr']:
        baseline_val = baseline_metrics[metric]
        new_val = new_metrics[metric]
        if baseline_val > 0:
            improvement = ((new_val - baseline_val) / baseline_val) * 100
            improvements[metric] = improvement
        else:
            improvements[metric] = 0.0
    
    # 响应时间的改进（越小越好，所以符号相反）
    baseline_time = baseline_metrics['avg_time_ms']
    new_time = new_metrics['avg_time_ms']
    if baseline_time > 0:
        improvements['avg_time_ms'] = ((baseline_time - new_time) / baseline_time) * 100
    else:
        improvements['avg_time_ms'] = 0.0
    
    return improvements


def run_optimization_experiments(evaluator, baseline, config):
    """
    步骤 2: 运行优化方案评估
    逐个测试新的检索策略并与 baseline 对比
    """
    print("\n" + "="*80)
    print("步骤 2: 测试优化方案并与 Baseline 对比")
    print("="*80)
    
    # 定义优化方案
    optimization_strategies = [
        {
            'name': 'vectorstore',
            'description': '纯向量检索 - 仅使用语义相似度',
        },
        {
            'name': 'adaptive',
            'description': '自适应检索 - 根据查询类型动态选择策略',
        },
        {
            'name': 'hybrid_fusion',
            'description': 'RRF 融合检索 - 使用倒数排名融合算法',
        },
        # {
        #     'name': 'rerank',
        #     'description': '重排序检索 - 向量召回 + Rerank 模型',
        # },
    ]
    
    comparison_report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'baseline': baseline,
        'optimizations': []
    }
    
    for i, strategy in enumerate(optimization_strategies, 1):
        print(f"\n--- 测试方案 {i}: {strategy['name']} ---")
        print(f"说明: {strategy['description']}")
        
        try:
            result = evaluator.evaluate_retriever(
                retriever_type=strategy['name'],
                **config
            )
            
            # 计算改进
            improvements = compare_with_baseline(baseline, result)
            
            # 保存结果
            result_file = f"evaluation/optimization_{strategy['name']}.json"
            evaluator.save_results(result, result_file)
            
            # 打印对比
            print(f"\n性能指标:")
            print(f"  Precision: {result['metrics']['avg_precision']:.3f} "
                  f"({improvements['avg_precision']:+.1f}%)")
            print(f"  Recall:    {result['metrics']['avg_recall']:.3f} "
                  f"({improvements['avg_recall']:+.1f}%)")
            print(f"  F1 Score:  {result['metrics']['f1_score']:.3f} "
                  f"({improvements['f1_score']:+.1f}%)")
            print(f"  MRR:       {result['metrics']['avg_mrr']:.3f} "
                  f"({improvements['avg_mrr']:+.1f}%)")
            print(f"  响应时间:  {result['metrics']['avg_time_ms']:.2f} ms "
                  f"({improvements['avg_time_ms']:+.1f}%)")
            
            # 判断是否优于 baseline
            if result['metrics']['f1_score'] > baseline['metrics']['f1_score']:
                print(f"✓ 优于 Baseline!")
            else:
                print(f"✗ 未超过 Baseline")
            
            # 记录到报告
            comparison_report['optimizations'].append({
                'strategy': strategy['name'],
                'description': strategy['description'],
                'result': result,
                'improvements': improvements
            })
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    return comparison_report


def generate_final_report(comparison_report):
    """
    步骤 3: 生成最终对比报告
    """
    print("\n" + "="*80)
    print("步骤 3: 生成对比分析报告")
    print("="*80)
    
    baseline = comparison_report['baseline']
    optimizations = comparison_report['optimizations']
    
    # 找出最佳方案
    best_strategy = None
    best_f1 = baseline['metrics']['f1_score']
    
    print(f"\n📊 完整性能对比表:")
    print("-"*80)
    print(f"{'策略':<20} {'Precision':<12} {'Recall':<12} {'F1':<10} {'MRR':<10} {'时间(ms)':<10}")
    print("-"*80)
    
    # Baseline
    b_metrics = baseline['metrics']
    print(f"{'Baseline (ensemble)':<20} "
          f"{b_metrics['avg_precision']:<12.3f} "
          f"{b_metrics['avg_recall']:<12.3f} "
          f"{b_metrics['f1_score']:<10.3f} "
          f"{b_metrics['avg_mrr']:<10.3f} "
          f"{b_metrics['avg_time_ms']:<10.2f}")
    
    # 优化方案
    for opt in optimizations:
        metrics = opt['result']['metrics']
        name = opt['strategy']
        
        # 标记最佳
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_strategy = opt
            marker = " ⭐"
        else:
            marker = ""
        
        print(f"{name:<20} "
              f"{metrics['avg_precision']:<12.3f} "
              f"{metrics['avg_recall']:<12.3f} "
              f"{metrics['f1_score']:<10.3f} "
              f"{metrics['avg_mrr']:<10.3f} "
              f"{metrics['avg_time_ms']:<10.2f}"
              f"{marker}")
    
    print("-"*80)
    
    # 推荐方案
    print("\n" + "="*80)
    if best_strategy:
        print("🏆 推荐方案 (相比 Baseline 的最佳改进)")
        print("="*80)
        print(f"\n策略: {best_strategy['strategy']}")
        print(f"说明: {best_strategy['description']}")
        print(f"\n改进幅度:")
        improvements = best_strategy['improvements']
        print(f"  Precision: {improvements['avg_precision']:+.1f}%")
        print(f"  Recall:    {improvements['avg_recall']:+.1f}%")
        print(f"  F1 Score:  {improvements['f1_score']:+.1f}%")
        print(f"  MRR:       {improvements['avg_mrr']:+.1f}%")
        print(f"  响应时间:  {improvements['avg_time_ms']:+.1f}%")
        
        print(f"\n📝 应用建议:")
        print(f"  在 kb_settings_retriever.yaml 中修改:")
        print(f"  DEFAULT_RETRIEVER_TYPE: {best_strategy['strategy']}")
    else:
        print("📌 结论: 当前 Baseline (ensemble) 仍是最佳方案")
        print("="*80)
        print("建议:")
        print("  1. 扩充测试集，增加更多真实查询样本")
        print("  2. 尝试调整参数 (top_k, score_threshold)")
        print("  3. 考虑启用 rerank 重排序模型")
    
    # 保存完整报告
    report_file = "evaluation/comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 完整报告已保存到: {report_file}")
    
    return best_strategy


def main():
    """主评估流程"""
    print("\n" + "🚀"*40)
    print("RAG 检索优化 - 评估驱动开发")
    print("🚀"*40)
    
    # 配置
    config = {
        'top_k': 5,
        'score_threshold': 0.5
    }
    
    print(f"\n配置参数: top_k={config['top_k']}, score_threshold={config['score_threshold']}")
    
    # 初始化评估器
    evaluator = RetrievalEvaluator(
        kb_name="samples",
        test_queries_path="evaluation/test_queries.json"
    )
    
    # 步骤 1: 建立 baseline
    baseline = run_baseline_evaluation(evaluator, config)
    
    # 步骤 2: 测试优化方案
    comparison_report = run_optimization_experiments(evaluator, baseline, config)
    
    # 步骤 3: 生成报告
    best_strategy = generate_final_report(comparison_report)
    
    print("\n" + "="*80)
    print("✅ 评估完成!")
    print("="*80)


if __name__ == "__main__":
    main()
