#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版中文讽刺识别测试器
针对初次测试结果进行优化
"""

import json
import requests
import time
from datetime import datetime

class ImprovedIronyTester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "..."##your API url
        
        # 同样的测试样本
        self.test_samples = [
            {"text": "这是最受气包的一届总统特朗普:我被陷害,我被诬告,我冤枉!特朗普反对者:我们只是在行使民主自由的权利!!", "label": 0, "type": "政治评论"},
            {"text": "那些相信经济全球化会走向命运共同体的人现在天天被打脸。", "label": 0, "type": "经济观点"},
            {"text": "嗯,这很西方,有强盗思维逻辑", "label": 0, "type": "文化评价"},
            {"text": "呵呵呵呵,站着说话不腰疼,棚户区改造你们怎么不说,穷人不是人?", "label": 0, "type": "社会批评"},
            {"text": "波音:世界末日到了空客:武林盟主是我的了:这是百年难遇的机会", "label": 0, "type": "商业调侃"},
            {"text": "我最喜欢阿柳一本正经地胡说八道啦。", "label": 1, "type": "人物调侃"},
            {"text": "特朗普这个宣传员做的不错。打脸美国所谓的自由民主。", "label": 1, "type": "政治讽刺"},
            {"text": "令人震惊的美国特色......", "label": 1, "type": "国际讽刺"},
            {"text": "黄鼠狼给鸡拜年,没安好心。", "label": 1, "type": "成语讽刺"},
            {"text": "哈哈,骚操作啊", "label": 0, "type": "网络用语"}
        ]
        
        # 测试不同的prompt模板
        self.prompt_templates = {
            'original': '''请判断以下中文句子是否为讽刺/反语表达：

句子："{text}"

请仔细分析句子的语境、语气和潜在含义，然后做出判断。

选项：
A. 是讽刺/反语
B. 不是讽刺/反语

请只回答A或B：''',
            
            'improved_v1': '''请分析以下中文句子是否包含讽刺或反语的表达方式：

句子："{text}"

判断标准：
- 讽刺/反语：说话者表面说的和真实想法相反，带有批评、嘲讽或调侃意味
- 非讽刺：直白表达观点，即使是批评也是直接的，没有反话或讽刺语气

请选择：
A. 包含讽刺/反语
B. 不包含讽刺/反语

只回答A或B：''',
            
            'improved_v2': '''作为一个中文语言专家，请判断以下句子是否使用了讽刺手法：

句子："{text}"

讽刺的特征：
1. 表面意思和真实意图相反
2. 带有嘲讽、调侃的语气
3. 通过对比或夸张表达不满

请仔细考虑后选择：
A. 这是讽刺表达
B. 这不是讽刺表达

答案：''',
            
            'few_shot': '''以下是一些讽刺识别的例子：

例子1：
句子："你可真厉害啊，连这么简单的题都做错了"
判断：A（讽刺 - 表面夸奖实际批评）

例子2：  
句子："今天天气真不错，下了一整天雨"
判断：B（不是讽刺 - 可能是无奈或客观描述）

例子3：
句子："他工作态度很认真，经常加班到很晚"
判断：B（不是讽刺 - 直接夸奖）

现在请判断：
句子："{text}"

A. 是讽刺
B. 不是讽刺

答案：'''
        }
        
        self.test_models = {
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini',
                'model_id': 'openai/gpt-4o-mini'
            },
            'claude-3-haiku': {
                'name': 'Claude 3 Haiku', 
                'model_id': 'anthropic/claude-3-haiku'
            }
        }
    
    def call_model(self, model_config: dict, prompt: str) -> str:
        """调用模型API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fivehills/LLM_Pragmabi_ZH",
            "X-Title": "Chinese Sarcasm  Evaluation Improved"
        }
        
        data = {
            "model": model_config['model_id'],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"ERROR_{response.status_code}"
                
        except Exception as e:
            return f"ERROR_{str(e)[:50]}"
    
    def parse_response(self, response: str) -> str:
        """改进的响应解析"""
        if not response or response.startswith('ERROR'):
            return "ERROR"
        
        response = response.strip().upper()
        
        # 查找明确的A或B答案
        if response.startswith('A') or '答案：A' in response or '选择：A' in response:
            return 'A'
        elif response.startswith('B') or '答案：B' in response or '选择：B' in response:
            return 'B'
        
        # 查找关键词
        positive_keywords = ['讽刺', '反语', '嘲讽', '调侃', '是的']
        negative_keywords = ['不是', '不包含', '没有', '直接', '客观']
        
        has_positive = any(kw in response for kw in positive_keywords)
        has_negative = any(kw in response for kw in negative_keywords)
        
        if has_positive and not has_negative:
            return 'A'
        elif has_negative and not has_positive:
            return 'B'
        
        return "UNCLEAR"
    
    def test_prompt_templates(self):
        """测试不同的prompt模板"""
        print("🧪 测试不同的prompt模板")
        print("="*50)
        
        results = {}
        
        for template_name, template in self.prompt_templates.items():
            print(f"\n📝 测试模板: {template_name}")
            
            # 只用GPT-4o Mini测试（节省成本）
            model_config = self.test_models['gpt-4o-mini']
            
            correct = 0
            total = 0
            template_results = []
            
            for i, sample in enumerate(self.test_samples[:5]):  # 只测试前5个样本
                prompt = template.format(text=sample['text'])
                response = self.call_model(model_config, prompt)
                prediction = self.parse_response(response)
                
                expected = 'A' if sample['label'] == 1 else 'B'
                is_correct = prediction == expected
                
                if prediction not in ['ERROR', 'UNCLEAR']:
                    total += 1
                    if is_correct:
                        correct += 1
                
                status = "✅" if is_correct else "❌" if prediction not in ['ERROR', 'UNCLEAR'] else "⚠️"
                print(f"   样本{i+1}: {status} {prediction} (期望:{expected}) - {sample['type']}")
                
                template_results.append({
                    'sample': sample,
                    'prediction': prediction,
                    'expected': expected,
                    'correct': is_correct,
                    'response': response[:100] + "..." if len(response) > 100 else response
                })
                
                time.sleep(1)  # API调用间隔
            
            accuracy = correct / total if total > 0 else 0
            print(f"   📊 准确率: {accuracy:.2%} ({correct}/{total})")
            
            results[template_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results': template_results
            }
        
        return results
    
    def find_best_prompt(self):
        """找出最佳prompt模板"""
        print("🎯 寻找最佳prompt模板")
        
        template_results = self.test_prompt_templates()
        
        # 找出表现最好的模板
        best_template = max(template_results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n🏆 最佳模板: {best_template[0]}")
        print(f"📊 准确率: {best_template[1]['accuracy']:.2%}")
        
        if best_template[1]['accuracy'] >= 0.6:
            print("✅ 找到了较好的prompt模板！")
            return best_template[0]
        else:
            print("⚠️  所有模板表现都不够理想，需要进一步优化")
            return None
    
    def run_full_test_with_best_prompt(self, best_template_name: str):
        """使用最佳prompt运行完整测试"""
        if best_template_name not in self.prompt_templates:
            print("❌ 无效的模板名称")
            return
        
        print(f"\n🚀 使用最佳模板 '{best_template_name}' 运行完整测试")
        
        template = self.prompt_templates[best_template_name]
        results = {}
        
        for model_key, model_config in self.test_models.items():
            print(f"\n🤖 测试模型: {model_config['name']}")
            
            correct = 0
            total = 0
            model_results = []
            
            for i, sample in enumerate(self.test_samples):
                prompt = template.format(text=sample['text'])
                response = self.call_model(model_config, prompt)
                prediction = self.parse_response(response)
                
                expected = 'A' if sample['label'] == 1 else 'B'
                is_correct = prediction == expected
                
                if prediction not in ['ERROR', 'UNCLEAR']:
                    total += 1
                    if is_correct:
                        correct += 1
                
                status = "✅" if is_correct else "❌" if prediction not in ['ERROR', 'UNCLEAR'] else "⚠️"
                print(f"   样本{i+1}: {status} {prediction} (期望:{expected}) - {sample['type']}")
                
                model_results.append({
                    'sample': sample,
                    'prediction': prediction,
                    'expected': expected,
                    'correct': is_correct,
                    'response': response
                })
                
                time.sleep(1)
            
            accuracy = correct / total if total > 0 else 0
            print(f"   📊 {model_config['name']} 准确率: {accuracy:.2%} ({correct}/{total})")
            
            results[model_key] = {
                'model_name': model_config['name'],
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results': model_results
            }
        
        return results
    
    def analyze_errors(self, results: dict):
        """分析错误模式"""
        print(f"\n🔍 错误模式分析")
        print("="*40)
        
        # 收集所有错误样本
        all_errors = []
        for model_data in results.values():
            for result in model_data['results']:
                if not result['correct'] and result['prediction'] not in ['ERROR', 'UNCLEAR']:
                    all_errors.append(result)
        
        if not all_errors:
            print("✅ 没有错误样本")
            return
        
        # 按错误类型分类
        false_positives = []  # 预测为讽刺，实际不是
        false_negatives = []  # 预测为非讽刺，实际是
        
        for error in all_errors:
            if error['prediction'] == 'A' and error['expected'] == 'B':
                false_positives.append(error)
            elif error['prediction'] == 'B' and error['expected'] == 'A':
                false_negatives.append(error)
        
        print(f"📊 错误统计:")
        print(f"   误判为讽刺: {len(false_positives)} 个")
        print(f"   误判为非讽刺: {len(false_negatives)} 个")
        
        if false_positives:
            print(f"\n❌ 容易被误判为讽刺的样本:")
            for error in false_positives[:3]:
                text = error['sample']['text'][:60] + "..." if len(error['sample']['text']) > 60 else error['sample']['text']
                print(f"   • {text}")
                print(f"     类型: {error['sample']['type']}")
        
        if false_negatives:
            print(f"\n❌ 容易被误判为非讽刺的样本:")
            for error in false_negatives[:3]:
                text = error['sample']['text'][:60] + "..." if len(error['sample']['text']) > 60 else error['sample']['text']
                print(f"   • {text}")
                print(f"     类型: {error['sample']['type']}")

def main():
    print("🧪 中文讽刺识别测试")
    print("="*40)
    
    # 获取API密钥
    api_key = input("请输入API密钥 (回车使用之前的): ").strip()
    if not api_key:
        # 尝试从config.json读取
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("..api_key")
                if api_key:
                    print("🔑 使用保存的API密钥")
        except:
            pass
    
    if not api_key:
        print("❌ 未找到API密钥")
        return
    
    tester = ImprovedIronyTester(api_key)
    

    
    confirm = input("\n确认开始改进版测试吗? (y/N): ").strip().lower()
    if confirm != 'y':
        print("测试已取消")
        return
    
    # 寻找最佳prompt
    best_template = tester.find_best_prompt()
    
    if best_template:
        # 使用最佳prompt运行完整测试
        results = tester.run_full_test_with_best_prompt(best_template)
        
        if results:
            # 分析错误
            tester.analyze_errors(results)
            
            # 保存结果
            output = {
                'test_info': {
                    'test_date': datetime.now().isoformat(),
                    'best_template': best_template,
                    'samples_tested': len(tester.test_samples)
                },
                'results': results
            }
            
            with open('improved_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 改进版测试结果已保存到: improved_test_results.json")
            
            # 给出建议
            avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
            print(f"\n💡 改进后评估:")
            if avg_accuracy >= 0.7:
                print(f"   ✅ 表现大幅改善！可以进行大规模评估")
                print(f"   📋 建议使用 '{best_template}' 模板进行完整评估")
            elif avg_accuracy >= 0.5:
                print(f"   ⚠️  有所改善，但仍需进一步优化")
                print(f"   💡 考虑增加更多few-shot例子或调整判断标准")
            else:
                print(f"   ❌ 改善有限，可能需要重新设计评估策略")
    
    print("\n🎉 改进版测试完成！")

if __name__ == "__main__":
    main()
