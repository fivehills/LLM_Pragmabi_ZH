# Installation and Usage Guide

## Quick Setup

### 1. Project Setup

```bash
# Clone or create the project directory
mkdir LLM_Pragmabi_ZH
cd LLM_Pragmabi_ZH

# Run the setup script to create directory structure
python scripts/run_full_evaluation.py --models your-model --sample-size 10
```

### Adding New Tasks

1. Create new evaluator in `evaluators/`:
```python
from .base_evaluator import BaseEvaluator

class YourTaskEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> bool:
        # Implementation
        pass
    
    def get_prompt_for_sample(self, sample, task_name):
        # Implementation
        pass
    
    # Implement other abstract methods
```

2. Register in main runner:
```python
'your_task': {
    'evaluator_class': YourTaskEvaluator,
    'dataset_path': 'datasets/your_task/data.json (or .csv)',
    'description': 'Your task description'
}
```

### Adding New Datasets

1. Follow existing JSON schema:
```json
{
  "metadata": {
    "description": "Dataset description",
    "total_samples": 100,
    "source": "Data source"
  },
  "samples": [
    {
      "text": "Sample text",
      "label": 1,
      "type": "category"
    }
  ]
}
```

2. Update dataset path in task configuration
3. Test loading:
```bash
python -c "
from evaluators.your_evaluator import YourEvaluator
evaluator = YourEvaluator()
success = evaluator.load_dataset('path/to/dataset.json')
print('Dataset loaded:', success)
"
```

## Cost Estimation

### API Usage Calculator

```python
def estimate_cost(sample_size, num_tasks=3, num_models=2):
    """Estimate evaluation costs"""
    
    # Base calculations
    total_samples = sample_size * num_tasks
    primary_calls = total_samples * num_models
    arbitration_calls = total_samples * 0.3  # ~30% disagreement rate
    total_calls = primary_calls + arbitration_calls
    
    # Cost estimates (per 1K tokens)
    model_costs = {
        'gpt-4o-mini': 0.00015,  # $0.15/1M tokens
        'claude-3-haiku': 0.00025,  # $0.25/1M tokens
        'gpt-4o': 0.005,  # $5/1M tokens
        'claude-3-sonnet': 0.003  # $3/1M tokens
    }
    
    # Assume ~100 tokens per call
    tokens_per_call = 100
    avg_cost_per_1k = 0.0005  # Average cost
    
    total_cost = (total_calls * tokens_per_call / 1000) * avg_cost_per_1k
    
    return {
        'total_api_calls': int(total_calls),
        'estimated_cost_usd': round(total_cost, 2),
        'estimated_time_minutes': round(total_calls * 2 / 60, 1)  # 2 sec/call
    }

# Example usage
print("Cost for 50 samples:", estimate_cost(50))
print("Cost for 200 samples:", estimate_cost(200))
```

### Typical Costs
- **Small test (20 samples)**: ~$0.05, 5 minutes
- **Medium evaluation (50 samples)**: ~$0.15, 10 minutes  
- **Large evaluation (200 samples)**: ~$0.60, 40 minutes
- **Full benchmark (500 samples)**: ~$1.50, 100 minutes

## Research Applications

### Academic Use Cases

1. **Cross-lingual Pragmatics Research**
   ```bash
   # Compare models across languages
   python scripts/run_full_evaluation.py --models multilingual-models --sample-size 100
   ```

2. **Cultural Understanding Assessment**
   ```bash
   # Focus on cultural nuances
   python scripts/run_full_evaluation.py --task idiom --sample-size 200 --models all-models
   ```

3. **Model Comparison Studies**
   ```bash
   # Systematic comparison
   python scripts/run_full_evaluation.py --models gpt-4o,claude-3,gemini-pro,qwen-72b --sample-size 150
   ```

### Commercial Applications

1. **Model Selection for Chinese NLP**
   - Evaluate models for specific use cases
   - Compare cost vs. performance trade-offs
   - Identify optimal models for different pragmatic tasks

2. **Quality Assurance for Translation Services**
   - Benchmark translation quality for idioms
   - Assess cultural sensitivity in AI translations
   - Monitor performance across updates

3. **Chatbot Evaluation**
   - Test understanding of indirect communication
   - Evaluate sarcasm detection capabilities
   - Assess cultural appropriateness

### Dataset Expansion

1. **Euphemism Dataset**
   - Add domain-specific euphemisms (medical, political, social)
   - Include regional variations
   - Expand to cover more sensitive topics

2. **Sarcasm Dataset**
   - Add context-dependent examples
   - Include social media data
   - Cover different sarcasm types (verbal irony, situational irony)

3. **Idiom Dataset**
   - Add modern Chinese idioms
   - Include regional idioms
   - Expand cultural context annotations

## Citation

If you use this framework in academic research, please cite:

```bibtex
@misc{chinese_llm_pragmatic_eval_2024,
  title={Chinese LLM Pragmatic Understanding Evaluation Framework},
  author={Anonymous},
  year={2024},
  url={https://github.com/fivehills/LLM_Pragmabi_ZH},
  note={A comprehensive evaluation framework for Chinese pragmatic understanding in large language models}
}
```

## Contributing

### Code Contributions

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-evaluator`
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

### Dataset Contributions

1. Follow existing data schemas
2. Provide validation scripts
3. Include metadata and source information
4. Add baseline human performance scores

### Issue Reporting

When reporting issues, include:
- Python version and dependencies
- Configuration file (without API keys)
- Error messages and stack traces
- Sample data that reproduces the issue

## License

This project is released under the MIT License. See LICENSE file for details.

The datasets may have different licenses - please check individual dataset files for specific terms./setup_project.py
```

### 2. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or with virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy configuration template
cp config.json.template config.json

# Edit config.json and add your API keys
```

Example config.json:
```json
{
  "openrouter_api_key": "sk-or-v1-your-key-here",
  "default_models": ["gpt-4o-mini", "claude-3-haiku"],
  "max_retries": 3,
  "request_timeout": 60,
  "rate_limit_delay": 1.0
}
```

### 4. Dataset Preparation

```bash
# Add your euphemism dataset (TSV format)
cp your_euphemism_data.csv datasets/euphemisms/zh_eupm_dataset.csv

# Sample datasets are already provided for sarcasm and idioms
```

## Usage Examples

### Basic Evaluation

```bash
# Run single task evaluation
python scripts/run_full_evaluation.py --task sarcasm --sample-size 20

# Run full evaluation with all tasks
python scripts/run_full_evaluation.py --sample-size 50

# Specify models to test
python scripts/run_full_evaluation.py --models gpt-4o claude-3-sonnet --sample-size 30
```

### Advanced Usage

```bash
# Large-scale evaluation
python scripts/run_full_evaluation.py --sample-size 200 --models gpt-4o,claude-3-sonnet,gemini-pro

# Quick test with default settings
python scripts/run_full_evaluation.py --sample-size 10 --dry-run

# Generate analysis after evaluation
python scripts/analyze_results.py --generate-report --generate-plots
```

### Programmatic Usage

```python
from evaluators.euphemism_evaluator import EuphemismEvaluator
from evaluators.sarcasm_evaluator import SarcasmEvaluator
from evaluators.idiom_evaluator import IdiomEvaluator

# Initialize evaluator
evaluator = EuphemismEvaluator("config.json")
evaluator.load_dataset("datasets/euphemisms/zh_eupm_dataset.csv")

# Create test samples
test_samples = evaluator.create_test_samples(50)

# Run evaluation
results = evaluator.run_evaluation("euphemism", test_samples, ["gpt-4o-mini"])

# Export results
evaluator.export_results("results/")
```

## Expected Output Structure

```
results/
├── euphemism_results.json          # Detailed euphemism evaluation results
├── euphemism_results.csv           # CSV format for analysis
├── sarcasm_results.json            # Detailed sarcasm evaluation results
├── sarcasm_results.csv             # CSV format for analysis
├── idiom_results.json              # Detailed idiom evaluation results
├── idiom_results.csv               # CSV format for analysis
├── evaluation_summary.json         # Overall summary
├── comprehensive_report.json       # Cross-task analysis
├── detailed_report.html           # HTML report (if generated)
└── performance_analysis.png        # Performance plots (if generated)
```

## Performance Metrics

### Euphemism Understanding
- **Identification Accuracy**: Binary classification of euphemistic vs. direct expressions
- **Explanation Quality**: Semantic similarity between predicted and gold meanings
- **Cultural Sensitivity**: Appropriateness of indirect language recognition

### Sarcasm Detection
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Standard classification metrics for sarcasm detection
- **Agreement Rate**: Inter-model consistency

### Idiom Translation
- **BLEU Score**: N-gram based translation quality
- **Semantic Similarity**: Meaning preservation assessment
- **Cultural Preservation**: Retention of metaphorical/cultural elements

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Verify API key is set correctly
   cat config.json | grep api_key
   
   # Test API connection
   python -c "
   import json
   import requests
   with open('config.json') as f:
       config = json.load(f)
   print('API key loaded:', bool(config.get('openrouter_api_key')))
   "
   ```

2. **Dataset Loading Issues**
   ```bash
   # Check dataset format
   head -5 datasets/euphemisms/zh_eupm_dataset.csv
   
   # Validate JSON datasets
   python -c "
   import json
   with open('datasets/sarcasm/sarcasm_samples.json') as f:
       data = json.load(f)
   print('Sarcasm samples:', len(data['samples']))
   "
   ```

3. **Memory Issues**
   ```bash
   # Reduce sample size for testing
   python scripts/run_full_evaluation.py --sample-size 10
   
   # Run tasks individually
   python scripts/run_full_evaluation.py --task sarcasm --sample-size 20
   ```

### Performance Optimization

1. **Reduce API Costs**
   - Use smaller sample sizes for testing
   - Start with cheaper models (gpt-4o-mini, claude-3-haiku)
   - Implement local caching for repeated evaluations

2. **Speed Up Evaluation**
   - Reduce `rate_limit_delay` in config if your API plan allows
   - Use fewer arbitration models
   - Process tasks in parallel (requires code modification)

### Validation

```bash
# Test configuration
python scripts/run_full_evaluation.py --dry-run --sample-size 5

# Validate datasets
python -c "
from evaluators.sarcasm_evaluator import SarcasmEvaluator
evaluator = SarcasmEvaluator()
evaluator.load_dataset('datasets/sarcasm/sarcasm_samples.json')
print('Dataset loaded successfully')
"

# Quick functionality test
python scripts/run_full_evaluation.py --sample-size 3 --models gpt-4o-mini
```

## Extending the Framework

### Adding New Models

1. Add model configuration to `base_evaluator.py`:
```python
'your-model': {
    'name': 'Your Model Name',
    'provider': 'your-provider',
    'model_id': 'provider/model-id',
    'max_tokens': 150,
    'temperature': 0.1
}
```

2. Test with existing evaluations:
```bash
python scripts


## Contact

 email: sharpksun@hotmail.com
