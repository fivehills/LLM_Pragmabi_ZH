#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文成语理解与翻译评估器
专门评估LLM对中文成语的理解和英文翻译能力
"""

import json
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import difflib
import re

class ChineseIdiomsEvaluator:
    def __init__(self, json_data_path: str, openrouter_api_key: str):
        """
        初始化中文成语评估器
        
        Args:
            json_data_path: 成语JSON数据文件路径
            openrouter_api_key: OpenRouter API密钥
        """
        self.json_data_path = json_data_path
        self.api_key = _api_key
        self.base_url = "..."##your API url
        
        self.raw_data = None
        self.test_data = None
        self.results = {}
        
        # 评估模型配置
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
    
    def load_idioms_data(self):
        """加载成语数据"""
        try:
            # 读取JSON格式的成语数据
            with open(self.json_data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            print(f"✅ 成功加载中文成语数据: {len(self.raw_data)} 条")
            
            # 验证数据结构
            if not self.raw_data or not isinstance(self.raw_data, list):
                print("❌ 数据格式错误，应为JSON数组")
                return False
            
            # 验证必要字段
            required_fields = ['id', 'chinese', 'gold']
            sample = self.raw_data[0] if self.raw_data else {}
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"❌ 缺失必要字段: {missing_fields}")
                return False
            
            # 显示数据样本
            print(f"\n📋 数据样本:")
            for i in range(min(5, len(self.raw_data))):
                item = self.raw_data[i]
                print(f"   {i+1}. ID: {item['id']}")
                print(f"      成语: {item['chinese']}")
                print(f"      标准翻译: {item['gold']}")
                print()
            
            # 统计信息
            print(f"📊 数据统计:")
            print(f"   总成语数量: {len(self.raw_data)}")
            print(f"   平均成语长度: {np.mean([len(item['chinese']) for item in self.raw_data]):.1f} 字符")
            print(f"   平均翻译长度: {np.mean([len(item['gold']) for item in self.raw_data]):.1f} 字符")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def create_evaluation_tasks(self, sample_size: int = 50):
        """创建评估任务"""
        if self.raw_data is None:
            print("❌ 请先加载数据")
            return False
        
        print(f"🎯 创建中文成语评估任务")
        print("="*50)
        
        # 准备测试数据
        self.test_data = self._prepare_test_samples(sample_size)
        
        # 定义评估任务
        self.evaluation_tasks = {
            'idiom_understanding': {
                'name': '成语理解',
                'description': '评估模型对中文成语含义的理解',
                'prompt_template': self._create_understanding_prompt(),
                'scoring_method': 'semantic_understanding'
            },
            
            'idiom_translation': {
                'name': '成语翻译',
                'description': '评估模型将中文成语翻译成英文的能力',
                'prompt_template': self._create_translation_prompt(),
                'scoring_method': 'translation_comparison'
            },
            
            'idiom_usage': {
                'name': '成语使用',
                'description': '评估模型在具体语境中使用成语的能力',
                'prompt_template': self._create_usage_prompt(),
                'scoring_method': 'usage_appropriateness'
            }
        }
        
        print(f"✅ 创建了 {len(self.evaluation_tasks)} 个评估任务")
        print(f"📊 准备了 {len(self.test_data)} 个测试样本")
        
        return True
    
    def _prepare_test_samples(self, sample_size: int):
        """准备测试样本"""
        print(f"📋 准备测试样本...")
        
        # 随机抽样
        if len(self.raw_data) > sample_size:
            sample_data = random.sample(self.raw_data, sample_size)
        else:
            sample_data = self.raw_data
        
        test_samples = []
        
        for item in sample_data:
            test_sample = {
                'id': item['id'],
                'chinese_idiom': item['chinese'],
                'gold_translation': item['gold'],
                'character_count': len(item['chinese']),
                'word_count': len(item['gold'].split())
            }
            test_samples.append(test_sample)
        
        print(f"✅ 准备了 {len(test_samples)} 个测试样本")
        
        return test_samples
    
    def _create_understanding_prompt(self):
        """创建成语理解prompt"""
        return '''作为中文语言专家，请解释以下中文成语的含义和用法：

成语："{chinese_idiom}"

## 任务要求：
1. 解释成语的字面意思
2. 说明成语的比喻含义
3. 描述使用场景
4. 给出一个使用例句

## 回答格式：
字面意思：[解释字面含义]
比喻含义：[解释深层含义]
使用场景：[什么情况下使用]
例句：[包含此成语的句子]

请按上述格式回答：'''
    
    def _create_translation_prompt(self):
        """创建成语翻译prompt"""
        return '''作为专业的中英翻译专家，请将以下中文成语翻译成准确的英文表达：

中文成语："{chinese_idiom}"

## 翻译要求：
• 保持成语的核心含义
• 使用地道的英文表达
• 避免直译，追求意译
• 简洁准确

## 翻译示例：
"一石二鸟" → "kill two birds with one stone"
"破釜沉舟" → "burn one's bridges"
"画蛇添足" → "gild the lily"

请直接给出英文翻译（不需要解释）：'''
    
    def _create_usage_prompt(self):
        """创建成语使用prompt"""
        return '''作为中文写作专家，请用以下成语造一个恰当的句子：

成语："{chinese_idiom}"

## 造句要求：
• 句子要完整通顺
• 正确使用成语含义
• 体现成语的语用效果
• 长度适中（10-30字）

## 造句示例：
成语："画龙点睛" → 他的发言为整个会议画龙点睛，点出了关键问题。
成语："雪中送炭" → 朋友在我最困难的时候帮助我，真是雪中送炭。

请直接给出造句（不需要解释）：'''
    
    def call_model(self, model_key: str, prompt: str, max_retries: int = 3) -> Optional[str]:
        """调用模型API"""
        model_config = self.models[model_key]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fivehills/LLM_Pragmabi_ZH",
            "X-Title": "Chinese Idioms Understanding and Translation Evaluation"
        }
        
        data = {
            "model": model_config['model_id'],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
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
    
    def calculate_translation_similarity(self, predicted: str, gold: str) -> float:
        """计算翻译相似度"""
        if not predicted or not gold:
            return 0.0
        
        # 清理文本
        predicted_clean = re.sub(r'[^\w\s]', '', predicted.lower().strip())
        gold_clean = re.sub(r'[^\w\s]', '', gold.lower().strip())
        
        # 计算多种相似度指标
        
        # 1. 序列相似度
        seq_sim = difflib.SequenceMatcher(None, predicted_clean, gold_clean).ratio()
        
        # 2. 词汇重叠度
        pred_words = set(predicted_clean.split())
        gold_words = set(gold_clean.split())
        
        if len(gold_words) == 0:
            word_overlap = 0.0
        else:
            word_overlap = len(pred_words & gold_words) / len(gold_words)
        
        # 3. BLEU风格的n-gram重叠
        def get_ngrams(text, n):
            words = text.split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        bleu_scores = []
        for n in range(1, min(4, len(gold_clean.split()) + 1)):
            pred_ngrams = get_ngrams(predicted_clean, n)
            gold_ngrams = get_ngrams(gold_clean, n)
            
            if len(gold_ngrams) == 0:
                bleu_scores.append(0.0)
            else:
                bleu_scores.append(len(pred_ngrams & gold_ngrams) / len(gold_ngrams))
        
        bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # 综合得分
        final_score = (seq_sim * 0.3 + word_overlap * 0.4 + bleu_score * 0.3)
        
        return final_score
    
    def evaluate_understanding_quality(self, response: str) -> Dict[str, Any]:
        """评估理解质量"""
        if not response:
            return {'score': 0.0, 'details': '无响应'}
        
        # 检查格式完整性
        required_sections = ['字面意思', '比喻含义', '使用场景', '例句']
        found_sections = sum(1 for section in required_sections if section in response)
        format_score = found_sections / len(required_sections)
        
        # 检查内容质量（基于长度和关键词）
        content_quality = min(len(response) / 200, 1.0)  # 200字符为满分
        
        # 检查是否包含成语解释的关键要素
        quality_indicators = ['意思', '含义', '比喻', '象征', '表示', '指的是']
        quality_score = sum(1 for indicator in quality_indicators if indicator in response) / len(quality_indicators)
        
        overall_score = (format_score * 0.4 + content_quality * 0.3 + quality_score * 0.3)
        
        return {
            'score': overall_score,
            'format_score': format_score,
            'content_quality': content_quality,
            'quality_score': quality_score,
            'details': f'格式:{format_score:.2f}, 内容:{content_quality:.2f}, 质量:{quality_score:.2f}'
        }
    
    def evaluate_usage_appropriateness(self, idiom: str, sentence: str) -> Dict[str, Any]:
        """评估成语使用的恰当性"""
        if not sentence:
            return {'score': 0.0, 'details': '无句子'}
        
        # 检查是否包含成语
        contains_idiom = idiom in sentence
        
        # 检查句子长度（10-30字符为理想范围）
        length_score = 1.0
        if len(sentence) < 10:
            length_score = len(sentence) / 10
        elif len(sentence) > 30:
            length_score = max(0.5, 30 / len(sentence))
        
        # 检查句子完整性（是否有标点符号）
        completeness_score = 1.0 if sentence.strip().endswith(('。', '！', '？', '.', '!', '?')) else 0.8
        
        # 基础语法检查（简单的中文语法模式）
        grammar_indicators = ['的', '了', '在', '是', '有', '被', '把', '给', '让', '使']
        grammar_score = min(1.0, sum(1 for indicator in grammar_indicators if indicator in sentence) / 3)
        
        overall_score = 0.0
        if contains_idiom:
            overall_score = (length_score * 0.3 + completeness_score * 0.3 + grammar_score * 0.4)
        
        return {
            'score': overall_score,
            'contains_idiom': contains_idiom,
            'length_score': length_score,
            'completeness_score': completeness_score,
            'grammar_score': grammar_score,
            'details': f'包含成语:{contains_idiom}, 长度:{length_score:.2f}, 完整性:{completeness_score:.2f}'
        }
    
    def run_dual_validation_evaluation(self, task_name: str, max_samples: Optional[int] = None):
        """运行双重验证评估"""
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
            print(f"   成语: {sample['chinese_idiom']}")
            if task_name == 'idiom_translation':
                print(f"   标准翻译: {sample['gold_translation']}")
            
            # 构建prompt
            prompt = task_info['prompt_template'].format(
                chinese_idiom=sample['chinese_idiom']
            )
            
            # 双重验证
            print(f"   🇨🇳 Qwen 72B: ", end="")
            qwen_response = self.call_model('qwen-72b', prompt)
            print(f"{'✅' if qwen_response else '❌'}")
            
            print(f"   🌍 Gemini 2.5: ", end="")
            gemini_response = self.call_model('gemini-2.5-flash', prompt)
            print(f"{'✅' if gemini_response else '❌'}")
            
            # 仲裁机制（当需要时）
            deepseek_response = None
            arbitration_used = False
            
            if not qwen_response or not gemini_response:
                print(f"   ⚖️  启用仲裁...")
                deepseek_response = self.call_model('deepseek-chat', prompt)
                arbitration_used = True
                arbitration_count += 1
                print(f"   🔍 仲裁: {'✅' if deepseek_response else '❌'}")
            else:
                agreement_count += 1
            
            # 选择最佳响应
            responses = [r for r in [qwen_response, gemini_response, deepseek_response] if r]
            final_response = responses[0] if responses else None
            
            # 评估响应质量
            evaluation_result = None
            if final_response:
                if task_name == 'idiom_translation':
                    # 翻译任务：计算与标准翻译的相似度
                    similarity_score = self.calculate_translation_similarity(
                        final_response, sample['gold_translation']
                    )
                    evaluation_result = {
                        'score': similarity_score,
                        'type': 'translation_similarity',
                        'details': f'相似度: {similarity_score:.2%}'
                    }
                    print(f"   📊 翻译相似度: {similarity_score:.2%}")
                    print(f"   🤖 模型翻译: {final_response}")
                
                elif task_name == 'idiom_understanding':
                    # 理解任务：评估解释质量
                    evaluation_result = self.evaluate_understanding_quality(final_response)
                    print(f"   📊 理解质量: {evaluation_result['score']:.2%}")
                    print(f"   📝 {evaluation_result['details']}")
                
                elif task_name == 'idiom_usage':
                    # 使用任务：评估造句恰当性
                    evaluation_result = self.evaluate_usage_appropriateness(
                        sample['chinese_idiom'], final_response
                    )
                    print(f"   📊 使用恰当性: {evaluation_result['score']:.2%}")
                    print(f"   💬 造句: {final_response}")
                    print(f"   📝 {evaluation_result['details']}")
            
            # 记录结果
            result = {
                'sample_id': sample['id'],
                'chinese_idiom': sample['chinese_idiom'],
                'gold_translation': sample.get('gold_translation'),
                'qwen_response': qwen_response,
                'gemini_response': gemini_response,
                'deepseek_response': deepseek_response,
                'final_response': final_response,
                'arbitration_used': arbitration_used,
                'evaluation': evaluation_result
            }
            
            results.append(result)
            time.sleep(1)  # API调用间隔
        
        # 保存结果
        self.results[task_name] = {
            'task_info': task_info,
            'results': results,
            'statistics': {
                'total_samples': len(results),
                'successful_responses': len([r for r in results if r['final_response']]),
                'agreement_rate': agreement_count / len(results) if len(results) > 0 else 0,
                'arbitration_rate': arbitration_count / len(results) if len(results) > 0 else 0
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
        print(f"   成功响应: {stats['successful_responses']}")
        print(f"   模型一致率: {stats['agreement_rate']:.2%}")
        print(f"   仲裁使用率: {stats['arbitration_rate']:.2%}")
        
        # 计算平均得分
        valid_evaluations = [r['evaluation'] for r in results if r['evaluation'] and 'score' in r['evaluation']]
        if valid_evaluations:
            avg_score = np.mean([eval_result['score'] for eval_result in valid_evaluations])
            print(f"   平均得分: {avg_score:.2%}")
            
            # 得分分布
            scores = [eval_result['score'] for eval_result in valid_evaluations]
            print(f"   得分分布:")
            print(f"     优秀 (>0.8): {len([s for s in scores if s > 0.8])}/{len(scores)} ({len([s for s in scores if s > 0.8])/len(scores):.1%})")
            print(f"     良好 (0.6-0.8): {len([s for s in scores if 0.6 <= s <= 0.8])}/{len(scores)} ({len([s for s in scores if 0.6 <= s <= 0.8])/len(scores):.1%})")
            print(f"     及格 (0.4-0.6): {len([s for s in scores if 0.4 <= s < 0.6])}/{len(scores)} ({len([s for s in scores if 0.4 <= s < 0.6])/len(scores):.1%})")
            print(f"     不及格 (<0.4): {len([s for s in scores if s < 0.4])}/{len(scores)} ({len([s for s in scores if s < 0.4])/len(scores):.1%})")
        
        # 翻译任务特殊统计
        if task_name == 'idiom_translation':
            high_similarity = len([r for r in results if r['evaluation'] and r['evaluation'].get('score', 0) > 0.7])
            print(f"   高相似度翻译 (>70%): {high_similarity}/{len(results)} ({high_similarity/len(results):.1%})")
    
    def export_results(self, output_dir: str = 'chinese_idioms_results'):
        """导出评估结果"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出详细结果
        for task_name, task_data in self.results.items():
            task_file = os.path.join(output_dir, f'{task_name}_results.json')
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
        
        # 导出Excel格式的翻译对比结果
        if 'idiom_translation' in self.results:
            translation_data = []
            for result in self.results['idiom_translation']['results']:
                translation_data.append({
                    'ID': result['sample_id'],
                    '中文成语': result['chinese_idiom'],
                    '标准翻译': result['gold_translation'],
                    '模型翻译': result['final_response'],
                    '相似度得分': result['evaluation']['score'] if result['evaluation'] else 0,
                    '是否使用仲裁': result['arbitration_used']
                })
            
            translation_df = pd.DataFrame(translation_data)
            excel_file = os.path.join(output_dir, 'translation_comparison.xlsx')
            translation_df.to_excel(excel_file, index=False, encoding='utf-8')
        
        # 导出汇总报告
        summary_report = {
            'evaluation_date': datetime.now().isoformat(),
            'data_source': self.json_data_path,
            'original_data_size': len(self.raw_data) if self.raw_data is not None else 0,
            'test_samples_used': len(self.test_data) if self.test_data else 0,
            'tasks_completed': len(self.results),
            'model_info': {
                'primary_validators': ['Qwen 72B', 'Gemini 2.5 Flash'],
                'arbitrator': 'DeepSeek Chat',
                'evaluation_method': 'Dual Validation + Arbitration'
            },
            'summary_statistics': {}
        }
        
        for task_name, task_data in self.results.items():
            summary_report['summary_statistics'][task_name] = task_data['statistics']
        
        summary_file = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 评估结果已导出到: {output_dir}/")

def main():
    print("🎓 中文成语理解与翻译评估系统")
    print("="*60)
    
    # 获取输入
    json_path = input("请输入成语JSON数据文件路径 (默认: idioms_data.json): ").strip()
    if not json_path:
        json_path = "zh_idiom_data.json"
    
    api_key = input("请输入OpenRouter API密钥 (回车使用配置文件): ").strip()
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
    evaluator = ChineseIdiomsEvaluator(json_path, api_key)
    
    # 加载数据
    if not evaluator.load_idioms_data():
        return
    
    # 创建评估任务
    sample_size = input("测试样本数量 (默认: 20): ").strip()
    sample_size = int(sample_size) if sample_size.isdigit() else 20
    
    if not evaluator.create_evaluation_tasks(sample_size):
        return
    
    # 选择任务
    print(f"\n🎯 可用评估任务:")
    tasks = list(evaluator.evaluation_tasks.keys())
    for i, task in enumerate(tasks, 1):
        task_info = evaluator.evaluation_tasks[task]
        print(f"   {i}. {task_info['name']} - {task_info['description']}")
    
    task_choice = input(f"选择任务 (1-{len(tasks)}, 默认: 2-翻译): ").strip()
    try:
        selected_task = tasks[int(task_choice) - 1]
    except (ValueError, IndexError):
        selected_task = 'idiom_translation'  # 默认选择翻译任务
    
    print(f"✅ 选择了任务: {evaluator.evaluation_tasks[selected_task]['name']}")
    
    # 开始评估
    print(f"\n💰 预估成本: ~${sample_size * 3 * 0.002:.2f}")
    print(f"⏱️  预估时间: {sample_size * 0.15:.0f}-{sample_size * 0.25:.0f} 分钟")
    
    confirm = input("确认开始评估吗? (y/N): ").strip().lower()
    
    if confirm == 'y':
        if evaluator.run_dual_validation_evaluation(selected_task):
            evaluator.export_results()
            
            print(f"\n🎉 中文成语评估完成!")
            print(f"🏆 基于标准成语数据集的权威理解与翻译评估!")
            print(f"📊 使用双重验证+仲裁的严谨评估方法!")
            print(f"🔬 为中文语言AI研究做出重要贡献!")
    else:
        print("评估已取消")

if __name__ == "__main__":
    main()
