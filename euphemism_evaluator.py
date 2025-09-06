#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tab分隔委婉语评估器
专门处理Tab分隔的委婉语学术数据集
"""

import json
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

class TabEuphemismEvaluator:
    def __init__(self, tsv_data_path: str, openrouter_api_key: str):
        """
        初始化Tab分隔委婉语评估器
        
        Args:
            tsv_data_path: TSV数据文件路径
            openrouter_api_key: OpenRouter API密钥
        """
        self.tsv_data_path = tsv_data_path
        self.api_key = _api_key
        self.base_url = "..." ### your API url
        
        self.raw_data = None
        self.test_data = None
        self.results = {}
        
        # 评估模型配置（复用成功配置）
        self.models = {
            'qwen-72b': {
                'name': 'Qwen 72B',
                'model_id': 'qwen/qwen-2.5-72b-instruct',
                'role': 'primary_validator',
                'type': '中文模型'
            },
            'gemini-2.5-flash': {
                'name': 'Gemini 2.5 Flash',
                'model_id': 'google/gemini-2.5-flash',
                'role': 'primary_validator', 
                'type': '国际模型'
            },
            'deepseek-chat': {
                'name': 'DeepSeek Chat',
                'model_id': 'deepseek/deepseek-chat',
                'role': 'arbitrator',
                'type': '中文模型'
            }
        }
    
    def load_euphemism_data(self):
        """加载委婉语数据"""
        try:
            # 读取Tab分隔的文件
            self.raw_data = pd.read_csv(
                self.tsv_data_path, 
                sep='\t', 
                encoding='utf-8',
                quoting=3  # QUOTE_NONE，避免引号问题
            )
            
            print(f"✅ 成功加载委婉语数据: {len(self.raw_data)} 条")
            
            # 显示数据结构
            print(f"📋 数据列: {list(self.raw_data.columns)}")
            
            # 检查关键列是否存在
            required_columns = ['EUPHEMISM', 'EXAMPLE OF USAGE', 'MEANING IN CHINESE']
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            
            if missing_columns:
                print(f"⚠️ 缺失关键列: {missing_columns}")
                return False
            
            # 统计目标领域分布
            if 'TARGET DOMAIN' in self.raw_data.columns:
                domain_counts = self.raw_data['TARGET DOMAIN'].value_counts()
                print(f"\n📊 目标领域分布:")
                for domain, count in domain_counts.head(10).items():
                    print(f"   {domain}: {count} 条")
            
            # 显示数据样本
            print(f"\n📋 数据样本:")
            for i in range(min(3, len(self.raw_data))):
                euphemism = self.raw_data.iloc[i]['EUPHEMISM']
                meaning = self.raw_data.iloc[i]['MEANING IN CHINESE']
                example = self.raw_data.iloc[i]['EXAMPLE OF USAGE'][:50] + "..."
                print(f"   {i+1}. {euphemism} → {meaning}")
                print(f"      例句: {example}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def create_evaluation_tasks(self, sample_size: int = 100):
        """创建评估任务"""
        if self.raw_data is None:
            print("❌ 请先加载数据")
            return False
        
        print(f"🎯 创建委婉语理解评估任务")
        print("="*50)
        
        # 准备测试数据
        self.test_data = self._prepare_test_samples(sample_size)
        
        # 任务1: 委婉语识别任务
        self.evaluation_tasks = {
            'euphemism_identification': {
                'name': '委婉语识别',
                'description': '判断句子中是否使用了委婉语',
                'prompt_template': self._create_identification_prompt(),
                'scoring_method': 'binary_classification'
            },
            
            'euphemism_explanation': {
                'name': '委婉语含义解释',
                'description': '解释委婉语的真实含义',
                'prompt_template': self._create_explanation_prompt(),
                'scoring_method': 'semantic_matching'
            }
        }
        
        print(f"✅ 创建了 {len(self.evaluation_tasks)} 个评估任务")
        print(f"📊 准备了 {len(self.test_data)} 个测试样本")
        
        return True
    
    def _prepare_test_samples(self, sample_size: int):
        """准备测试样本"""
        print(f"📋 准备测试样本...")
        
        # 过滤有效数据
        valid_data = self.raw_data.dropna(subset=['EUPHEMISM', 'EXAMPLE OF USAGE', 'MEANING IN CHINESE'])
        
        # 随机抽样
        if len(valid_data) > sample_size:
            test_samples = valid_data.sample(sample_size, random_state=42)
        else:
            test_samples = valid_data
        
        # 转换为评估格式
        test_data = []
        for idx, row in test_samples.iterrows():
            # 创建正例（包含委婉语的句子）
            positive_sample = {
                'id': f"{row['ID']}_positive",
                'text': row['EXAMPLE OF USAGE'],
                'euphemism_word': row['EUPHEMISM'],
                'true_meaning': row['MEANING IN CHINESE'],
                'target_domain': row.get('TARGET DOMAIN', 'Unknown'),
                'has_euphemism': True,
                'expected_answer': 'A'  # A表示有委婉语
            }
            test_data.append(positive_sample)
            
            # 创建负例（将委婉语替换为直接表达）
            direct_text = self._create_direct_expression(row['EXAMPLE OF USAGE'], row['EUPHEMISM'], row['MEANING IN CHINESE'])
            if direct_text:
                negative_sample = {
                    'id': f"{row['ID']}_negative",
                    'text': direct_text,
                    'euphemism_word': '',
                    'true_meaning': row['MEANING IN CHINESE'],
                    'target_domain': row.get('TARGET DOMAIN', 'Unknown'),
                    'has_euphemism': False,
                    'expected_answer': 'B'  # B表示没有委婉语
                }
                test_data.append(negative_sample)
        
        print(f"✅ 准备了 {len(test_data)} 个测试样本 (正例: {len([s for s in test_data if s['has_euphemism']])}, 负例: {len([s for s in test_data if not s['has_euphemism']])})")
        
        return test_data
    
    def _create_direct_expression(self, original_text, euphemism, meaning):
        """创建直接表达的对照句子"""
        # 简单替换策略
        if euphemism in original_text:
            # 提取meaning中的核心词汇
            meaning_clean = meaning.replace('[', '').replace(']', '').replace('多用于死者的悼慰', '')
            direct_words = meaning_clean.split('，')[0].split('、')[0].strip()
            
            # 替换委婉语为直接表达
            direct_text = original_text.replace(euphemism, direct_words)
            return direct_text
        
        return None
    
    def _create_identification_prompt(self):
        """创建委婉语识别prompt"""
        return '''作为中文语言专家，请判断以下句子中是否使用了委婉语表达：

句子："{text}"

## 委婉语定义：
委婉语是用间接、温和的方式表达敏感、不愉快或禁忌内容的语言现象。

## 判断要点：
• 委婉语：避免直接表达，使用隐喻、转喻等修辞手法
• 直接表达：直白地表达意思，不做修饰或回避

## 委婉语参考例子：

### 委婉语例子（选择A）：
1. "他已经安息了" → 委婉表达死亡（隐喻：死亡如睡眠）
2. "她终于静静地安眠" → 委婉表达死亡
3. "走完了人生历程" → 委婉表达死亡（隐喻：生命如旅程）
4. "百年之后" → 委婉表达死亡（用长寿反说死亡）
5. "背世" → 委婉表达死亡（隐喻：死亡如离开）
6. "闭眼" → 委婉表达死亡（转喻：用死亡特征代替死亡）
7. "毕命" → 委婉表达死亡（隐喻：生命如任务完成）

### 直接表达例子（选择B）：
1. "他昨天去世了" → 直接表达死亡
2. "她死了" → 直接表达死亡
3. "他结束了生命" → 直接表达死亡
4. "很多年以后" → 直接表达时间
5. "离开了人世" → 相对直接的表达
6. "他闭上了眼睛" → 如果指睡觉，则是直接表达
7. "完成了任务" → 如果指工作，则是直接表达

请选择：
A. 句子中使用了委婉语
B. 句子中没有使用委婉语

只需回答A或B：'''
    
    def _create_explanation_prompt(self):
        """创建委婉语解释prompt"""
        return '''作为中文语言专家，请解释以下句子中委婉语的真实含义：

句子："{text}"
委婉语："{euphemism_word}"

请用简洁明确的语言解释这个委婉语的真实含义。

## 解释要求：
• 用直白的语言说明真实含义
• 不超过10个字
• 避免再次使用委婉语

## 解释示例：
委婉语："安息" → 真实含义："死亡"
委婉语："安眠" → 真实含义："死亡"
委婉语："百年" → 真实含义："死亡"
委婉语："背世" → 真实含义："死亡"
委婉语："闭眼" → 真实含义："死亡"

请直接给出真实含义（不超过10个字）：'''
    
    def call_model(self, model_key: str, prompt: str, max_retries: int = 3) -> Optional[str]:
        """调用模型API"""
        model_config = self.models[model_key]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tab-euphemism-eval",
            "X-Title": "Tab Chinese Euphemism Evaluation"
        }
        
        data = {
            "model": model_config['model_id'],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"      ⏳ API限流，等待{wait_time}秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt == max_retries - 1:
                        print(f"      ❌ API错误 {response.status_code}")
                        return None
                    time.sleep(1)
            
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"      ⚠️ 请求异常: {e}")
                    return None
                time.sleep(2)
        
        return None
    
    def parse_response(self, response: str, task_type: str) -> str:
        """解析模型响应"""
        if not response:
            return "ERROR"
        
        response = response.strip()
        
        if task_type == 'euphemism_identification':
            # 查找A或B选项
            if response.upper().startswith('A') or '答案：A' in response:
                return 'A'
            elif response.upper().startswith('B') or '答案：B' in response:
                return 'B'
            
            # 查找关键词
            if '委婉语' in response or '使用了' in response:
                return 'A'
            elif '没有' in response or '不是' in response:
                return 'B'
        
        elif task_type == 'euphemism_explanation':
            # 对于解释任务，返回清理后的响应
            # 移除常见的前缀
            cleaned = response.replace('真实含义：', '').replace('含义：', '').replace('意思：', '')
            return cleaned.strip()
        
        return "UNCLEAR"
    
    def run_evaluation(self, task_name: str, max_samples: Optional[int] = None):
        """运行评估任务（使用双重验证+仲裁）"""
        if not self.test_data or task_name not in self.evaluation_tasks:
            print(f"❌ 任务 {task_name} 不存在或数据未准备")
            return False
        
        task_info = self.evaluation_tasks[task_name]
        test_samples = self.test_data[:max_samples] if max_samples else self.test_data
        
        print(f"\n🚀 开始评估: {task_info['name']}")
        print(f"📊 测试样本: {len(test_samples)} 条")
        print("="*60)
        
        results = []
        agreement_count = 0
        arbitration_count = 0
        
        for i, sample in enumerate(test_samples):
            print(f"\n📝 样本 {i+1}/{len(test_samples)}")
            print(f"   类型: {'委婉语句子' if sample['has_euphemism'] else '直接表达'}")
            print(f"   文本: {sample['text'][:60]}...")
            
            # 构建prompt
            if task_name == 'euphemism_identification':
                prompt = task_info['prompt_template'].format(text=sample['text'])
            else:
                prompt = task_info['prompt_template'].format(
                    text=sample['text'],
                    euphemism_word=sample['euphemism_word']
                )
            
            # 双重验证
            print(f"   🇨🇳 Qwen 72B: ", end="")
            qwen_response = self.call_model('qwen-72b', prompt)
            qwen_prediction = self.parse_response(qwen_response, task_name) if qwen_response else "ERROR"
            print(f"{qwen_prediction}")
            
            print(f"   🌍 Gemini 2.5: ", end="")
            gemini_response = self.call_model('gemini-2.5-flash', prompt)
            gemini_prediction = self.parse_response(gemini_response, task_name) if gemini_response else "ERROR"
            print(f"{gemini_prediction}")
            
            # 一致性检查和仲裁
            final_prediction = None
            arbitration_used = False
            
            if qwen_prediction == gemini_prediction and qwen_prediction not in ["ERROR", "UNCLEAR"]:
                final_prediction = qwen_prediction
                agreement_count += 1
                print(f"   ✅ 一致预测: {final_prediction}")
            else:
                print(f"   ⚖️  需要仲裁...")
                deepseek_response = self.call_model('deepseek-chat', prompt)
                final_prediction = self.parse_response(deepseek_response, task_name) if deepseek_response else "ERROR"
                arbitration_used = True
                arbitration_count += 1
                print(f"   🔍 仲裁结果: {final_prediction}")
            
            # 评估准确性（仅对识别任务）
            is_correct = None
            if task_name == 'euphemism_identification' and final_prediction in ['A', 'B']:
                is_correct = final_prediction == sample['expected_answer']
                status = "✅" if is_correct else "❌"
                print(f"   {status} 预测: {final_prediction}, 期望: {sample['expected_answer']}")
            
            # 记录结果
            result = {
                'sample_id': sample['id'],
                'text': sample['text'],
                'has_euphemism': sample['has_euphemism'],
                'euphemism_word': sample['euphemism_word'],
                'true_meaning': sample['true_meaning'],
                'qwen_prediction': qwen_prediction,
                'gemini_prediction': gemini_prediction,
                'final_prediction': final_prediction,
                'expected_answer': sample.get('expected_answer'),
                'correct': is_correct,
                'arbitration_used': arbitration_used,
                'target_domain': sample['target_domain']
            }
            
            results.append(result)
            time.sleep(1)  # API调用间隔
        
        # 保存结果
        self.results[task_name] = {
            'task_info': task_info,
            'results': results,
            'statistics': {
                'total_samples': len(results),
                'agreement_rate': agreement_count / len(results),
                'arbitration_rate': arbitration_count / len(results)
            }
        }
        
        # 显示统计
        self._print_task_statistics(task_name)
        
        return True
    
    def _print_task_statistics(self, task_name: str):
        """打印任务统计"""
        if task_name not in self.results:
            return
        
        task_results = self.results[task_name]
        results = task_results['results']
        stats = task_results['statistics']
        
        print(f"\n📊 {task_results['task_info']['name']} 统计结果")
        print("="*50)
        
        print(f"📋 基础统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   模型一致率: {stats['agreement_rate']:.2%}")
        print(f"   仲裁使用率: {stats['arbitration_rate']:.2%}")
        
        # 计算准确率（仅对识别任务）
        if task_name == 'euphemism_identification':
            valid_results = [r for r in results if r['correct'] is not None]
            if valid_results:
                correct_count = sum(1 for r in valid_results if r['correct'])
                accuracy = correct_count / len(valid_results)
                print(f"   整体准确率: {accuracy:.2%} ({correct_count}/{len(valid_results)})")
                
                # 按类型分析
                positive_results = [r for r in valid_results if r['has_euphemism']]
                negative_results = [r for r in valid_results if not r['has_euphemism']]
                
                if positive_results:
                    pos_correct = sum(1 for r in positive_results if r['correct'])
                    pos_accuracy = pos_correct / len(positive_results)
                    print(f"   委婉语识别准确率: {pos_accuracy:.2%} ({pos_correct}/{len(positive_results)})")
                
                if negative_results:
                    neg_correct = sum(1 for r in negative_results if r['correct'])
                    neg_accuracy = neg_correct / len(negative_results)
                    print(f"   直接表达识别准确率: {neg_accuracy:.2%} ({neg_correct}/{len(negative_results)})")
    
    def export_results(self, output_dir: str = 'euphemism_evaluation_results'):
        """导出评估结果"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出每个任务的详细结果
        for task_name, task_data in self.results.items():
            task_file = os.path.join(output_dir, f'{task_name}_detailed_results.json')
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
        
        # 导出汇总报告
        summary_report = {
            'evaluation_date': datetime.now().isoformat(),
            'data_source': self.tsv_data_path,
            'total_original_samples': len(self.raw_data) if self.raw_data is not None else 0,
            'test_samples_created': len(self.test_data) if self.test_data else 0,
            'tasks_completed': len(self.results),
            'summary_statistics': {}
        }
        
        for task_name, task_data in self.results.items():
            summary_report['summary_statistics'][task_name] = task_data['statistics']
        
        summary_file = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 评估结果已导出到: {output_dir}/")

def main():
    print("🎓 Tab分隔中文委婉语理解评估系统")
    print("="*60)
    
    # 获取输入
    tsv_path = input("请输入Tab分隔数据文件路径 (默认: database.csv): ").strip()
    if not tsv_path:
        tsv_path = "zh_eupm_dataset.csv"
    
    api_key = input("请输入 API密钥 (回车使用之前的): ").strip()
    if not api_key:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("openrouter_api_key")
                if api_key:
                    print("🔑 使用保存的API密钥")
        except:
            pass
    
    if not api_key:
        print("❌ 未找到API密钥")
        return
    
    # 创建评估器
    evaluator = TabEuphemismEvaluator(tsv_path, api_key)
    
    # 加载数据
    if not evaluator.load_euphemism_data():
        return
    
    # 创建评估任务
    sample_size = input("测试样本数量 (默认: 50): ").strip()
    sample_size = int(sample_size) if sample_size.isdigit() else 50
    
    if not evaluator.create_evaluation_tasks(sample_size):
        return
    
    # 选择任务
    print(f"\n🎯 可用评估任务:")
    tasks = list(evaluator.evaluation_tasks.keys())
    for i, task in enumerate(tasks, 1):
        task_info = evaluator.evaluation_tasks[task]
        print(f"   {i}. {task_info['name']} - {task_info['description']}")
    
    task_choice = input(f"选择任务 (1-{len(tasks)}, 默认: 1): ").strip()
    try:
        selected_task = tasks[int(task_choice) - 1]
    except (ValueError, IndexError):
        selected_task = tasks[0]  # 默认选择第一个
    
    print(f"✅ 选择了任务: {evaluator.evaluation_tasks[selected_task]['name']}")
    
    # 开始评估
    print(f"\n💰 预估成本: ~${sample_size * 3 * 0.001:.2f}")
    confirm = input("确认开始评估吗? (y/N): ").strip().lower()
    
    if confirm == 'y':
        if evaluator.run_evaluation(selected_task):
            evaluator.export_results()
            
            print(f"\n🎉 委婉语评估完成!")
            print(f"🏆 这是基于学术数据集的权威中文委婉语理解评估!")
            print(f"📊 使用了双重验证+仲裁的严谨方法!")
    else:
        print("评估已取消")

if __name__ == "__main__":
    main()
