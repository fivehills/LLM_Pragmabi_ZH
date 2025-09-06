#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main evaluation runner for Chinese LLM Pragmatic Understanding
Coordinates all three evaluation tasks: euphemisms, sarcasm, and idioms
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Import task-specific evaluators
from evaluators.euphemism_evaluator import EuphemismEvaluator
from evaluators.sarcasm_evaluator import SarcasmEvaluator
from evaluators.idiom_evaluator import IdiomEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChinesePragmaticEvaluator:
    """Main coordinator for Chinese pragmatic understanding evaluation"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.evaluators = {}
        self.results = {}
        
        # Initialize task evaluators
        self.task_configs = {
            'euphemism': {
                'evaluator_class': EuphemismEvaluator,
                'dataset_path': 'datasets/euphemisms/zh_eupm_dataset.csv',
                'description': 'Chinese euphemism understanding and explanation'
            },
            'sarcasm': {
                'evaluator_class': SarcasmEvaluator,
                'dataset_path': 'datasets/sarcasm/sarcasm_samples.json',
                'description': 'Chinese sarcasm and irony detection'
            },
            'idiom': {
                'evaluator_class': IdiomEvaluator,
                'dataset_path': 'datasets/idioms/idiom_translation_pairs.json',
                'description': 'Chinese idiom translation and cultural understanding'
            }
        }
    
    def initialize_evaluators(self, tasks: list = None):
        """Initialize evaluators for specified tasks"""
        if tasks is None:
            tasks = list(self.task_configs.keys())
        
        for task in tasks:
            if task not in self.task_configs:
                logger.warning(f"Unknown task: {task}")
                continue
            
            try:
                evaluator_class = self.task_configs[task]['evaluator_class']
                evaluator = evaluator_class(self.config_path)
                
                # Load dataset
                dataset_path = self.task_configs[task]['dataset_path']
                if not evaluator.load_dataset(dataset_path):
                    logger.error(f"Failed to load dataset for {task}")
                    continue
                
                self.evaluators[task] = evaluator
                logger.info(f"Initialized {task} evaluator")
                
            except Exception as e:
                logger.error(f"Failed to initialize {task} evaluator: {e}")
    
    def run_single_task(self, task: str, models: list = None, sample_size: int = 50):
        """Run evaluation for a single task"""
        if task not in self.evaluators:
            logger.error(f"Task {task} not initialized")
            return None
        
        logger.info(f"Running {task} evaluation")
        logger.info(f"Description: {self.task_configs[task]['description']}")
        
        evaluator = self.evaluators[task]
        
        # Create test samples
        test_samples = evaluator.create_test_samples(sample_size)
        if not test_samples:
            logger.error(f"Failed to create test samples for {task}")
            return None
        
        # Run evaluation
        result = evaluator.run_evaluation(task, test_samples, models)
        
        if result:
            self.results[task] = result
            logger.info(f"Completed {task} evaluation")
            return result
        else:
            logger.error(f"Failed to complete {task} evaluation")
            return None
    
    def run_full_evaluation(self, models: list = None, sample_size: int = 50, 
                          tasks: list = None):
        """Run complete evaluation across all tasks"""
        if tasks is None:
            tasks = list(self.evaluators.keys())
        
        logger.info("Starting full Chinese pragmatic understanding evaluation")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Models: {models}")
        logger.info(f"Sample size per task: {sample_size}")
        
        # Estimate costs and time
        total_samples = len(tasks) * sample_size
        num_models = len(models) if models else 2
        estimated_calls = total_samples * (num_models + 0.3)  # +30% for arbitration
        estimated_cost = estimated_calls * 0.002  # rough estimate
        estimated_time = estimated_calls * 2 / 60  # 2 seconds per call
        
        logger.info(f"Estimated API calls: {estimated_calls:.0f}")
        logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        logger.info(f"Estimated time: {estimated_time:.1f} minutes")
        
        # Run each task
        for task in tasks:
            try:
                self.run_single_task(task, models, sample_size)
            except Exception as e:
                logger.error(f"Error in {task} evaluation: {e}")
                continue
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        logger.info("Full evaluation completed")
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report across all tasks"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'tasks_completed': list(self.results.keys()),
                'total_duration': sum(
                    result['duration_minutes'] for result in self.results.values()
                )
            },
            'overall_statistics': {},
            'task_summaries': {},
            'cross_task_analysis': {}
        }
        
        # Calculate overall statistics
        total_samples = sum(result['statistics']['total_samples'] for result in self.results.values())
        avg_agreement = sum(result['statistics']['agreement_rate'] for result in self.results.values()) / len(self.results)
        
        report['overall_statistics'] = {
            'total_samples_evaluated': total_samples,
            'average_agreement_rate': avg_agreement,
            'tasks_completed': len(self.results),
            'overall_success_rate': self._calculate_overall_success_rate()
        }
        
        # Task summaries
        for task, result in self.results.items():
            summary = {
                'description': self.task_configs[task]['description'],
                'samples_evaluated': result['statistics']['total_samples'],
                'agreement_rate': result['statistics']['agreement_rate'],
                'key_metrics': self._extract_key_metrics(task, result)
            }
            report['task_summaries'][task] = summary
        
        # Cross-task analysis
        report['cross_task_analysis'] = self._perform_cross_task_analysis()
        
        # Save comprehensive report
        output_path = Path("results/comprehensive_report.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comprehensive report saved to {output_path}")
        return report
    
    def _calculate_overall_success_rate(self):
        """Calculate overall success rate across all tasks"""
        success_rates = []
        for task, result in self.results.items():
            if 'accuracy' in result['statistics']:
                success_rates.append(result['statistics']['accuracy'])
            elif 'f1_score' in result['statistics']:
                success_rates.append(result['statistics']['f1_score'])
            elif 'bleu_score' in result['statistics']:
                success_rates.append(result['statistics']['bleu_score'])
        
        return sum(success_rates) / len(success_rates) if success_rates else 0.0
    
    def _extract_key_metrics(self, task, result):
        """Extract key metrics for each task type"""
        stats = result['statistics']
        
        if task == 'euphemism':
            return {
                'identification_accuracy': stats.get('identification_accuracy', 0),
                'explanation_quality': stats.get('explanation_similarity', 0),
                'overall_score': stats.get('accuracy', 0)
            }
        elif task == 'sarcasm':
            return {
                'precision': stats.get('precision', 0),
                'recall': stats.get('recall', 0),
                'f1_score': stats.get('f1_score', 0),
                'accuracy': stats.get('accuracy', 0)
            }
        elif task == 'idiom':
            return {
                'bleu_score': stats.get('bleu_score', 0),
                'semantic_similarity': stats.get('semantic_similarity', 0),
                'cultural_preservation': stats.get('cultural_score', 0)
            }
        
        return stats
    
    def _perform_cross_task_analysis(self):
        """Perform cross-task analysis to identify patterns"""
        analysis = {
            'model_consistency': {},
            'difficulty_patterns': {},
            'strengths_weaknesses': {}
        }
        
        # Analyze model consistency across tasks
        all_models = set()
        for result in self.results.values():
            all_models.update(result['models_used']['primary'])
            all_models.add(result['models_used']['arbitrator'])
        
        for model in all_models:
            model_performance = {}
            for task, result in self.results.items():
                # Extract model-specific performance if available
                task_stats = result['statistics']
                if 'accuracy' in task_stats:
                    model_performance[task] = task_stats['accuracy']
                elif 'f1_score' in task_stats:
                    model_performance[task] = task_stats['f1_score']
            
            if model_performance:
                analysis['model_consistency'][model] = {
                    'performance_by_task': model_performance,
                    'average_performance': sum(model_performance.values()) / len(model_performance),
                    'consistency_score': 1.0 - (max(model_performance.values()) - min(model_performance.values()))
                }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Chinese LLM Pragmatic Understanding Evaluation')
    
    parser.add_argument('--task', choices=['euphemism', 'sarcasm', 'idiom', 'all'], 
                       default='all', help='Task to evaluate')
    parser.add_argument('--models', nargs='+', 
                       default=['gpt-4o-mini', 'claude-3-haiku'],
                       help='Models to evaluate')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples per task')
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running evaluation')
    
    args = parser.parse_args()
    
    # Validate configuration
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create config.json with your API keys")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = ChinesePragmaticEvaluator(args.config)
    
    # Determine tasks to run
    if args.task == 'all':
        tasks = ['euphemism', 'sarcasm', 'idiom']
    else:
        tasks = [args.task]
    
    # Initialize task evaluators
    evaluator.initialize_evaluators(tasks)
    
    if not evaluator.evaluators:
        logger.error("No evaluators successfully initialized")
        sys.exit(1)
    
    # Show configuration
    logger.info(f"Configuration:")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Sample size: {args.sample_size}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    if args.dry_run:
        logger.info("Dry run completed - no evaluation performed")
        return
    
    # Confirm execution
    total_samples = len(tasks) * args.sample_size
    estimated_cost = total_samples * len(args.models) * 0.002
    
    print(f"\nEvaluation Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  Estimated time: {total_samples * 2 / 60:.1f} minutes")
    
    confirm = input("\nProceed with evaluation? (y/N): ").strip().lower()
    if confirm != 'y':
        logger.info("Evaluation cancelled")
        return
    
    # Run evaluation
    try:
        if len(tasks) == 1:
            results = evaluator.run_single_task(tasks[0], args.models, args.sample_size)
        else:
            results = evaluator.run_full_evaluation(args.models, args.sample_size, tasks)
        
        if results:
            # Export results
            for task_evaluator in evaluator.evaluators.values():
                task_evaluator.export_results(args.output_dir)
            
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results saved to {args.output_dir}/")
        else:
            logger.error("Evaluation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()