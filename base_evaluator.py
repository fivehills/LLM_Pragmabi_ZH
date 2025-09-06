#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Evaluator Framework for Chinese LLM Pragmatic Understanding
Provides common functionality for all evaluation tasks
"""

import json
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    """Base class for all Chinese pragmatic understanding evaluators"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the base evaluator with configuration"""
        self.config = self._load_config(config_path)
        self.results = {}
        self.test_data = None
        self.evaluation_start_time = None
        
        # Model configurations
        self.models = {
            'gpt-4o': {
                'name': 'GPT-4o',
                'provider': 'openai',
                'model_id': 'openai/gpt-4o',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini',
                'provider': 'openai',
                'model_id': 'openai/gpt-4o-mini',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'claude-3-sonnet': {
                'name': 'Claude 3 Sonnet',
                'provider': 'anthropic',
                'model_id': 'anthropic/claude-3-sonnet',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'claude-3-haiku': {
                'name': 'Claude 3 Haiku',
                'provider': 'anthropic',
                'model_id': 'anthropic/claude-3-haiku',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'gemini-pro': {
                'name': 'Gemini Pro',
                'provider': 'google',
                'model_id': 'google/gemini-pro',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'qwen-72b': {
                'name': 'Qwen 2.5 72B',
                'provider': 'qwen',
                'model_id': 'qwen/qwen-2.5-72b-instruct',
                'max_tokens': 150,
                'temperature': 0.1
            },
            'deepseek-chat': {
                'name': 'DeepSeek Chat',
                'provider': 'deepseek',
                'model_id': 'deepseek/deepseek-chat',
                'max_tokens': 150,
                'temperature': 0.1
            }
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default settings")
            return {
                "openrouter_api_key": None,
                "default_models": ["gpt-4o-mini", "claude-3-haiku"],
                "max_retries": 3,
                "request_timeout": 60,
                "rate_limit_delay": 1.0
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def call_model(self, model_key: str, prompt: str, max_retries: Optional[int] = None) -> Optional[str]:
        """Call a model via OpenRouter API with retry logic"""
        if model_key not in self.models:
            logger.error(f"Unknown model: {model_key}")
            return None
            
        model_config = self.models[model_key]
        max_retries = max_retries or self.config.get("max_retries", 3)
        
        headers = {
            "Authorization": f"Bearer {self.config['openrouter_api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fivehills/LLM_Pragmabi_ZH",
            "X-Title": "Chinese LLM Pragmatic Understanding Evaluation"
        }
        
        data = {
            "model": model_config['model_id'],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": model_config['max_tokens'],
            "temperature": model_config['temperature']
        }
        
        url = "..." ###please use your API or other API agents
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.config.get("request_timeout", 60)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * self.config.get("rate_limit_delay", 1.0)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"API error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(1)
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
        
        return None
    
    def dual_validation_evaluation(self, sample: Dict[str, Any], task_name: str, 
                                 primary_models: List[str], arbitrator_model: str) -> Dict[str, Any]:
        """
        Perform dual validation with arbitration for a single sample
        
        Args:
            sample: Test sample data
            task_name: Name of the evaluation task
            primary_models: List of primary validation models
            arbitrator_model: Arbitrator model for disagreements
            
        Returns:
            Evaluation result with predictions and consensus
        """
        prompt = self.get_prompt_for_sample(sample, task_name)
        predictions = {}
        responses = {}
        
        # Get predictions from primary models
        for model_key in primary_models:
            response = self.call_model(model_key, prompt)
            prediction = self.parse_response(response, task_name)
            predictions[model_key] = prediction
            responses[model_key] = response
            
            logger.debug(f"{model_key}: {prediction}")
            time.sleep(self.config.get("rate_limit_delay", 1.0))
        
        # Check for agreement
        unique_predictions = set(pred for pred in predictions.values() 
                               if pred not in ["ERROR", "UNCLEAR"])
        
        if len(unique_predictions) == 1:
            # Agreement found
            final_prediction = list(unique_predictions)[0]
            arbitration_used = False
            logger.debug(f"Agreement: {final_prediction}")
        else:
            # Disagreement, use arbitrator
            arbitrator_response = self.call_model(arbitrator_model, prompt)
            final_prediction = self.parse_response(arbitrator_response, task_name)
            arbitration_used = True
            responses[arbitrator_model] = arbitrator_response
            logger.debug(f"Arbitration: {final_prediction}")
            time.sleep(self.config.get("rate_limit_delay", 1.0))
        
        # Evaluate correctness if ground truth available
        evaluation = self.evaluate_prediction(sample, final_prediction, task_name)
        
        return {
            'sample_id': sample.get('id', 'unknown'),
            'predictions': predictions,
            'responses': responses,
            'final_prediction': final_prediction,
            'arbitration_used': arbitration_used,
            'evaluation': evaluation,
            'sample_data': sample
        }
    
    def run_evaluation(self, task_name: str, test_samples: List[Dict[str, Any]], 
                      models: Optional[List[str]] = None, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete evaluation for a task
        
        Args:
            task_name: Name of the evaluation task
            test_samples: List of test samples
            models: Models to use (defaults to config)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Complete evaluation results
        """
        self.evaluation_start_time = datetime.now()
        
        if models is None:
            models = self.config.get("default_models", ["gpt-4o-mini", "claude-3-haiku"])
        
        # Ensure we have at least 2 models for dual validation
        if len(models) < 2:
            models.append("gpt-4o-mini")  # fallback
        
        primary_models = models[:2]
        arbitrator_model = models[2] if len(models) > 2 else "deepseek-chat"
        
        # Limit samples if specified
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        logger.info(f"Starting {task_name} evaluation")
        logger.info(f"Primary models: {primary_models}")
        logger.info(f"Arbitrator: {arbitrator_model}")
        logger.info(f"Test samples: {len(test_samples)}")
        
        results = []
        agreement_count = 0
        
        for i, sample in enumerate(test_samples):
            logger.info(f"Processing sample {i+1}/{len(test_samples)}")
            
            result = self.dual_validation_evaluation(
                sample, task_name, primary_models, arbitrator_model
            )
            
            if not result['arbitration_used']:
                agreement_count += 1
            
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(test_samples)
                logger.info(f"Progress: {progress:.1%} - Agreement rate: {agreement_count/(i+1):.1%}")
        
        # Calculate statistics
        statistics = self.calculate_statistics(results, task_name)
        
        evaluation_result = {
            'task_name': task_name,
            'evaluation_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.evaluation_start_time).total_seconds() / 60,
            'models_used': {
                'primary': primary_models,
                'arbitrator': arbitrator_model
            },
            'statistics': statistics,
            'results': results
        }
        
        self.results[task_name] = evaluation_result
        logger.info(f"Evaluation completed - Overall performance: {statistics.get('accuracy', 'N/A')}")
        
        return evaluation_result
    
    def calculate_statistics(self, results: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for evaluation results"""
        total_samples = len(results)
        agreement_count = sum(1 for r in results if not r['arbitration_used'])
        valid_results = [r for r in results if r['evaluation'] is not None]
        
        stats = {
            'total_samples': total_samples,
            'valid_evaluations': len(valid_results),
            'agreement_rate': agreement_count / total_samples if total_samples > 0 else 0,
            'arbitration_rate': (total_samples - agreement_count) / total_samples if total_samples > 0 else 0
        }
        
        # Task-specific statistics
        if valid_results:
            task_stats = self.calculate_task_specific_statistics(valid_results, task_name)
            stats.update(task_stats)
        
        return stats
    
    def export_results(self, output_dir: str = "results") -> None:
        """Export evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detailed results
        for task_name, task_data in self.results.items():
            # JSON export
            json_file = output_path / f"{task_name}_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            
            # CSV export for analysis
            csv_data = []
            for result in task_data['results']:
                row = {
                    'sample_id': result['sample_id'],
                    'final_prediction': result['final_prediction'],
                    'arbitration_used': result['arbitration_used'],
                }
                
                # Add model predictions
                for model, pred in result['predictions'].items():
                    row[f'{model}_prediction'] = pred
                
                # Add evaluation metrics
                if result['evaluation']:
                    if isinstance(result['evaluation'], dict):
                        for key, value in result['evaluation'].items():
                            row[f'eval_{key}'] = value
                    else:
                        row['eval_score'] = result['evaluation']
                
                csv_data.append(row)
            
            csv_file = output_path / f"{task_name}_results.csv"
            pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8')
        
        # Export summary report
        summary = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'tasks_evaluated': list(self.results.keys()),
                'total_duration_minutes': sum(
                    task_data['duration_minutes'] for task_data in self.results.values()
                )
            },
            'task_statistics': {
                task_name: task_data['statistics'] 
                for task_name, task_data in self.results.items()
            }
        }
        
        summary_file = output_path / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results exported to {output_path}")
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def load_dataset(self, dataset_path: str) -> bool:
        """Load dataset specific to the evaluation task"""
        pass
    
    @abstractmethod
    def get_prompt_for_sample(self, sample: Dict[str, Any], task_name: str) -> str:
        """Generate prompt for a specific sample and task"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str, task_name: str) -> str:
        """Parse model response into standardized format"""
        pass
    
    @abstractmethod
    def evaluate_prediction(self, sample: Dict[str, Any], prediction: str, task_name: str) -> Optional[Dict[str, Any]]:
        """Evaluate prediction against ground truth"""
        pass
    
    @abstractmethod
    def calculate_task_specific_statistics(self, results: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
        """Calculate task-specific evaluation statistics"""
        pass
