# Chinese LLM Pragmatic Understanding Evaluation

A comprehensive evaluation framework for testing Large Language Models' understanding of Chinese pragmatic meaning across three key dimensions: euphemisms, sarcasm/irony, and idiom translation.

## Overview

This repository provides standardized benchmarks and evaluation tools for assessing how well LLMs understand nuanced Chinese language expressions that require pragmatic inference beyond literal comprehension.

## Evaluation Dimensions

### 1. Euphemism Understanding (委婉语理解)
- **Dataset**: Tab-separated Chinese euphemism corpus
- **Task**: Identify euphemistic expressions and explain their real meanings
- **Skills Tested**: Indirect meaning interpretation, cultural sensitivity

### 2. Sarcasm/Irony Detection (讽刺识别)
- **Dataset**: Curated Chinese text samples with sarcasm labels
- **Task**: Binary classification of sarcastic vs. direct statements
- **Skills Tested**: Context understanding, tone recognition, pragmatic inference

### 3. Idiom Translation (成语翻译)
- **Dataset**: Chinese idioms with gold-standard English translations
- **Task**: Translate idioms preserving meaning and cultural nuance
- **Skills Tested**: Cultural knowledge, semantic equivalence, cross-lingual pragmatics

## Repository Structure

```
chinese-llm-pragmatic-eval/
├── README.md
├── requirements.txt
├── config.json.template
├── datasets/
│   ├── euphemisms/
│   │   ├── zh_eupm_dataset.csv
│   │   └── metadata.json
│   ├── sarcasm/
│   │   ├── sarcasm_samples.json
│   │   └── metadata.json
│   └── idioms/
│       ├── idiom_translation_pairs.json
│       └── metadata.json
├── evaluators/
│   ├── base_evaluator.py
│   ├── euphemism_evaluator.py
│   ├── sarcasm_evaluator.py
│   └── idiom_evaluator.py
├── prompts/
│   ├── euphemism_prompts.py
│   ├── sarcasm_prompts.py
│   └── idiom_prompts.py
├── models/
│   ├── model_configs.py
│   └── api_clients.py
├── results/
│   └── analysis/
├── scripts/
│   ├── run_full_evaluation.py
│   ├── run_single_task.py
│   └── analyze_results.py
└── docs/
    ├── evaluation_methodology.md
    ├── dataset_descriptions.md
    └── model_comparison.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/chinese-llm-pragmatic-eval.git
cd chinese-llm-pragmatic-eval
pip install -r requirements.txt
```

### Configuration

1. Copy the template configuration:
```bash
cp config.json.template config.json
```

2. Add your API keys:
```json
{
  "openrouter_api_key": "your-openrouter-key",
  "openai_api_key": "your-openai-key",
  "anthropic_api_key": "your-anthropic-key"
}
```

### Running Evaluations

#### Full Evaluation Suite
```bash
python scripts/run_full_evaluation.py --models gpt-4o,claude-3 --sample-size 100
```

#### Single Task Evaluation
```bash
# Euphemism understanding
python scripts/run_single_task.py --task euphemism --model gpt-4o --samples 50

# Sarcasm detection
python scripts/run_single_task.py --task sarcasm --model claude-3 --samples 50

# Idiom translation
python scripts/run_single_task.py --task idiom --model gemini-pro --samples 50
```

#### Results Analysis
```bash
python scripts/analyze_results.py --results-dir results/ --generate-report
```

## Evaluation Methodology

### Scoring Metrics

#### Euphemism Understanding
- **Identification Accuracy**: Binary classification performance (euphemistic vs. direct)
- **Explanation Quality**: Semantic similarity to gold standard meanings
- **Cultural Sensitivity**: Appropriateness of indirect language recognition

#### Sarcasm Detection
- **Classification Accuracy**: Precision, recall, F1-score for sarcasm detection
- **Context Sensitivity**: Performance across different sarcasm types
- **Robustness**: Consistency across multiple prompt formulations

#### Idiom Translation
- **Semantic Accuracy**: BLEU score and semantic similarity metrics
- **Cultural Preservation**: Retention of metaphorical/cultural meaning
- **Fluency**: Natural English expression quality

### Multi-Annotator Validation

The framework employs a dual-validation approach:
- **Primary Models**: Two high-performance models provide initial judgments
- **Arbitration**: Third model resolves disagreements
- **Agreement Rate**: Tracks inter-model consistency

## Supported Models

### OpenRouter Models
- GPT-4o, GPT-4o Mini
- Claude 3 (Haiku, Sonnet, Opus)
- Gemini Pro/Flash
- Qwen 2.5 (various sizes)
- DeepSeek Chat

### Local Models
- Extend `models/api_clients.py` for local model inference

## Results and Benchmarks

### Performance Baselines

| Model | Euphemism Acc. | Sarcasm F1 | Idiom BLEU | Overall Score |
|-------|---------------|------------|------------|---------------|
| GPT-4o | 0.85 | 0.78 | 0.72 | 0.78 |
| Claude-3 | 0.82 | 0.81 | 0.69 | 0.77 |
| Gemini-Pro | 0.79 | 0.74 | 0.65 | 0.73 |

*Scores based on 100-sample evaluations*

### Error Analysis

Common failure patterns:
- **Euphemisms**: Literal interpretation of indirect expressions
- **Sarcasm**: Missing contextual tone indicators
- **Idioms**: Over-literal character-by-character translation

## Contributing

### Adding New Datasets
1. Follow the JSON schema in `datasets/metadata.json`
2. Include validation scripts
3. Provide baseline human performance scores

### Adding New Models
1. Implement API client in `models/api_clients.py`
2. Add model configuration in `models/model_configs.py`
3. Test with sample evaluations

### Improving Prompts
1. Follow prompt templates in `prompts/`
2. Include few-shot examples
3. Test across multiple models

## Citation

If you use this evaluation framework, please cite:

```bibtex
@misc{chinese_llm_pragmatic_eval_2024,
  title={Chinese LLM Pragmatic Understanding Evaluation Framework},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-org/chinese-llm-pragmatic-eval}
}
```
## Contact

email: sharpksun@hotmail.com

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Euphemism dataset based on academic linguistic research
- Sarcasm samples curated from social media and literature
- Idiom translations validated by bilingual experts
