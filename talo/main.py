#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Add VLMEvalKit path
VLMEVALKIT_DIR = "dir2vlmevalkit" 
sys.path.append(VLMEVALKIT_DIR)

from talo.subtask_extractor import SubtaskExtractor
from talo.model_wrapper import ModelWrapper
from talo.evaluation_engine import EvaluationEngine
from talo.probe_layer import probe_target_layer_vlmeval_enhanced
from talo.final_evaluation import final_evaluation_vlmeval_style
from talo.utils import setup_logging, setup_subtask_logging, filter_tasks
import numpy as np

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="VLM model layer probing and evaluation script")
    parser.add_argument('--model_name', type=str, required=False, default='llava_next_llama3',
                       help='Model name to evaluate')
    parser.add_argument('--dataset_name', type=str, required=False, default='MMStar',
                       help='Dataset name to evaluate')
    parser.add_argument('--shot', type=int, default=5,
                       help='Number of samples for probing')
    parser.add_argument('--output_root', type=str, default='./probe_results',
                       help='Root directory for results')
    parser.add_argument('--sampling_strategy', type=str, 
                       choices=['l2_priority', 'category_l2_stratified', 'l2_category_stratified', 
                               'category_random', 'category_skill_stratified', 'skill_stratified'],
                       default='l2_priority',
                       help='Sampling strategy')
    parser.add_argument('--target_tasks', type=str, nargs='*', default=None,
                       help='Specific task names to process, if not specified, process all tasks')
    parser.add_argument('--exclude_tasks', type=str, nargs='*', default=None,
                       help='Task names to exclude')
    parser.add_argument('--list_tasks', action='store_true',
                       help='List all tasks in the dataset without executing probing')
    parser.add_argument('--task_pattern', type=str, default=None,
                       help='Regular expression to match task names')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.join(args.output_root, args.model_name, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup main logger
    main_logger = setup_logging(output_dir)
    main_logger.info(f"Starting probing and evaluation task: model={args.model_name}, dataset={args.dataset_name}, shot={args.shot}")
    
    try:
        # 1. Initialize components
        main_logger.info("🔧 Initializing system components...")
        
        # Load subtask extractor with new sampling strategy
        subtask_extractor = SubtaskExtractor(args.dataset_name, args.sampling_strategy)
        if not subtask_extractor.load_dataset():
            main_logger.error("Dataset loading failed")
            return
        
        # Extract subtasks
        subtasks = subtask_extractor.extract_subtasks()
        if not subtasks:
            main_logger.error("No subtasks found")
            return
        
        # Apply task filtering
        filtered_subtasks = filter_tasks(subtasks, args)
        
        # If only listing tasks, return
        if args.list_tasks:
            return
        
        if not filtered_subtasks:
            main_logger.error("No tasks to process after filtering")
            return
        
        # Record classification strategy
        main_logger.info(f"📊 Classification strategy: Using '{subtask_extractor.primary_category_field}' field")
        main_logger.info(f"📋 Sampling strategy: {args.sampling_strategy}")

        # Load model
        model_wrapper = ModelWrapper(args.model_name)
        if not model_wrapper.load_model():
            main_logger.error("Model loading failed")
            return
        
        # Initialize evaluation engine
        evaluation_engine = EvaluationEngine(args.dataset_name)
        if not evaluation_engine.load_dataset():
            main_logger.error("Evaluation engine dataset loading failed")
            return
        
        # 2. Process all subtasks
        final_results = {}
        total_subtasks = len(filtered_subtasks)
        
        main_logger.info(f"🚀 Starting to process {total_subtasks} subtasks...")
        
        # Create summary results file
        summary_log_file = os.path.join(output_dir, "all_tasks_summary.log")
        summary_logger = setup_subtask_logging(output_dir, "summary")
        summary_logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        for handler in summary_logger.handlers[:]:
            summary_logger.removeHandler(handler)
            
        summary_handler = logging.FileHandler(summary_log_file, encoding='utf-8')
        summary_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        summary_handler.setFormatter(formatter)
        summary_logger.addHandler(summary_handler)
        
        summary_logger.info("=" * 80)
        summary_logger.info("All Subtasks Performance Evaluation Summary")
        summary_logger.info("=" * 80)
        summary_logger.info(f"Model: {args.model_name}")
        summary_logger.info(f"Dataset: {args.dataset_name}")
        summary_logger.info(f"Shot: {args.shot}")
        summary_logger.info(f"Total Subtasks: {total_subtasks}")
        summary_logger.info("-" * 80)
        summary_logger.info(f"{'Task Name':<20} {'Target Layer':<6} {'Baseline':<7} {'Processed':<7} {'Change':<10} {'Status':<8}")
        summary_logger.info("-" * 80)
        
        for subtask_idx, (subtask_name, samples) in enumerate(filtered_subtasks.items(), 1):
            # Setup independent logger for each subtask
            subtask_work_dir = os.path.join(output_dir, f"subtask_{subtask_name.replace(' ', '_')}")
            os.makedirs(subtask_work_dir, exist_ok=True)
            subtask_logger = setup_subtask_logging(subtask_work_dir, subtask_name)
            
            main_logger.info(f"Processing subtask {subtask_idx}/{total_subtasks}: {subtask_name} ({len(samples)} samples)")
            subtask_logger.info(f"Starting to process subtask {subtask_name} ({subtask_idx}/{total_subtasks})")
            subtask_logger.info(f"Total samples: {len(samples)}")
            
            try:
                # Check sample count
                if len(samples) < args.shot:
                    warning_msg = f"Subtask {subtask_name} sample count ({len(samples)}) is less than shot count ({args.shot}), skipping"
                    main_logger.warning(warning_msg)
                    subtask_logger.warning(warning_msg)
                    continue
                
                # Use new sampling strategy
                print(f"📊 Subtask {subtask_name} - Starting {args.sampling_strategy} sampling...")
                subtask_logger.info(f"Starting {args.sampling_strategy} sampling (target shot: {args.shot})")
                subtask_logger.info(f"Primary category field: {subtask_extractor.primary_category_field}")
                
                # Step 1: Strategically sample all samples to build probe pool
                max_probe_pool_size = min(len(samples) // 3, args.shot * 4)
                probe_pool_samples, probe_pool_stats = subtask_extractor.apply_sampling_strategy(
                    samples, max_probe_pool_size
                )
                
                subtask_logger.info(f"Probe pool sampling completed:")
                subtask_logger.info(f"  - Target size: {max_probe_pool_size}")
                subtask_logger.info(f"  - Actual size: {len(probe_pool_samples)}")
                subtask_logger.info(f"  - Sampling stats: {probe_pool_stats}")
                
                # Step 2: Resample from probe pool to get initial probe samples
                probe_samples, probe_samples_stats = subtask_extractor.apply_sampling_strategy(
                    probe_pool_samples, args.shot
                )
                
                subtask_logger.info(f"Initial probe sample sampling completed:")
                subtask_logger.info(f"  - Target size: {args.shot}")
                subtask_logger.info(f"  - Actual size: {len(probe_samples)}")
                subtask_logger.info(f"  - Sampling stats: {probe_samples_stats}")
                
                # Step 3: Remaining samples as evaluation set
                probe_pool_indices = {sample.get('index', -1) for sample in probe_pool_samples}
                eval_samples = [sample for sample in samples if sample.get('index', -1) not in probe_pool_indices]
                
                # Verify sample separation
                probe_indices = {sample.get('index', -1) for sample in probe_samples}
                eval_indices = {sample.get('index', -1) for sample in eval_samples}
                overlap = probe_indices & eval_indices
                
                if overlap:
                    error_msg = f"❌ Found sample overlap! Overlapping indices: {overlap}"
                    subtask_logger.error(error_msg)
                    main_logger.error(error_msg)
                    continue
                else:
                    subtask_logger.info(f"✅ Sample isolation verification passed: probe pool({len(probe_pool_indices)}) vs evaluation set({len(eval_indices)}), no overlap")
                
                print(f"📊 Subtask {subtask_name} sample allocation:")
                print(f"  - Total samples: {len(samples)}")
                print(f"  - Probe pool size: {len(probe_pool_samples)} ({args.sampling_strategy} sampling)")
                print(f"  - Initial probe samples: {len(probe_samples)} ({args.sampling_strategy} sampling)")
                print(f"  - Evaluation samples: {len(eval_samples)}")
                print(f"  - Sampling strategy: {args.sampling_strategy}")
                
                # Create work directory
                work_dir = subtask_work_dir
                
                # Phase 1: Probe target layer
                subtask_logger.info("Phase 1: Probing target layer")
                target_layer, base_score, probe_details = probe_target_layer_vlmeval_enhanced(
                    model_wrapper, evaluation_engine, probe_samples, work_dir,
                    all_samples=probe_pool_samples, shot=args.shot  # Pass probe pool for resampling
                )
                
                # Phase 2: Final performance evaluation (including baseline model comparison)
                subtask_logger.info("Phase 2: Final performance evaluation (including baseline model comparison)")
                base_eval_performance, target_eval_performance = final_evaluation_vlmeval_style(
                    model_wrapper, evaluation_engine, eval_samples, target_layer, work_dir
                )
                
                # Calculate performance change metrics
                performance_change = target_eval_performance - base_eval_performance
                performance_change_pct = (performance_change / base_eval_performance * 100) if base_eval_performance != 0 else 0
                
                # Evaluate effectiveness
                is_improvement = performance_change > 0
                is_significant = abs(performance_change) >= 0.01  # 1% change considered significant
                
                # Record detailed results
                final_results[subtask_name] = {
                    'target_layer': target_layer,
                    'probe_base_score': base_score,  # Baseline score from probing phase
                    'base_eval_performance': base_eval_performance,  # Baseline score from evaluation phase
                    'target_eval_performance': target_eval_performance,  # Score after target layer processing
                    'performance_change': performance_change,
                    'performance_change_pct': performance_change_pct,
                    'is_improvement': is_improvement,
                    'is_significant': is_significant,
                    'details': probe_details
                }
                
                # Calculate performance change metrics
                performance_change = target_eval_performance - base_eval_performance
                performance_change_pct = (performance_change / base_eval_performance * 100) if base_eval_performance != 0 else 0
                
                # Evaluate effectiveness
                is_improvement = performance_change > 0
                is_significant = abs(performance_change) >= 0.01  # 1% change considered significant
                
                # Record detailed results
                final_results[subtask_name] = {
                    'target_layer': target_layer,
                    'probe_base_score': base_score,  # Baseline score from probing phase
                    'base_eval_performance': base_eval_performance,  # Baseline score from evaluation phase
                    'target_eval_performance': target_eval_performance,  # Score after target layer processing
                    'performance_change': performance_change,
                    'performance_change_pct': performance_change_pct,
                    'is_improvement': is_improvement,
                    'is_significant': is_significant,
                    'details': probe_details
                }
                
                # Print single subtask result summary
                status_symbol = "📈" if is_improvement else "📉" if performance_change < 0 else "➡️"
                significance = "significant" if is_significant else "minor"
                
                result_msg = f"\n{status_symbol} Subtask {subtask_name} completed:"
                subtask_logger.info(result_msg)
                main_logger.info(result_msg)
                
                details_msg = f"  - Target layer: {target_layer}\n" \
                             f"  - Baseline performance: {base_eval_performance:.4f}\n" \
                             f"  - Processed performance: {target_eval_performance:.4f}\n" \
                             f"  - Performance change: {performance_change:+.4f} ({performance_change_pct:+.2f}%) - {significance}"
                subtask_logger.info(details_msg)
                main_logger.info(details_msg)
                
                subtask_logger.info(f"Subtask {subtask_name} completed: target_layer={target_layer}, "
                          f"base={base_eval_performance:.4f}, target={target_eval_performance:.4f}, "
                          f"change={performance_change:+.4f} ({performance_change_pct:+.2f}%)")
                
                # Record to summary log
                if target_layer >= 0:
                    if is_improvement:
                        status = "📈 Improvement" if is_significant else "📈 Minor impr"
                    elif performance_change < 0:
                        status = "📉 Decrease" if is_significant else "📉 Minor decr"
                    else:
                        status = "➡️ No change"
                else:
                    status = "❌ Failed"
                    
                summary_logger.info(f"{subtask_name:<20} {target_layer:<6} {base_eval_performance:<7.3f} {target_eval_performance:<7.3f} {performance_change_pct:+7.2f}% {status:<8}")
                
            except Exception as e:
                error_msg = f"Error processing subtask {subtask_name}: {e}"
                main_logger.error(error_msg)
                subtask_logger.error(error_msg)
                main_logger.error(traceback.format_exc())
                subtask_logger.error(traceback.format_exc())
                
                # Record error results
                final_results[subtask_name] = {
                    'target_layer': -1,
                    'probe_base_score': 0.0,
                    'base_eval_performance': 0.0,
                    'target_eval_performance': 0.0,
                    'performance_change': 0.0,
                    'performance_change_pct': 0.0,
                    'is_improvement': False,
                    'is_significant': False,
                    'details': {
                        'error': str(e)
                    }
                }
                
                # Error information also recorded to summary log
                summary_logger.error(f"{subtask_name:<20} {'ERROR':<6} {'N/A':<7} {'N/A':<7} {'N/A':<10} {'❌ Failed':<8}")
                continue
        
        # 3. Save final results
        results_file = os.path.join(output_dir, 'results_vlmeval_style.json')
        
        # Add metadata
        metadata = {
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'shot': args.shot,
            'total_subtasks': len(final_results),
            'results': final_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        main_logger.info(f"All results saved to: {results_file}")
        summary_logger.info("-" * 80)
        summary_logger.info(f"Results file: {results_file}")
        summary_logger.info("=" * 80)
        
        # 4. Generate comprehensive result analysis report
        report_msg = f"\n🎉 Probing and evaluation completed!"
        main_logger.info(report_msg)
        summary_logger.info(report_msg)
        report_msg = f"=" * 60
        main_logger.info(report_msg)
        summary_logger.info(report_msg)
        
        # Basic statistics
        successful_tasks = [name for name, result in final_results.items() 
                          if result['target_layer'] >= 0]
        improved_tasks = [name for name, result in final_results.items() 
                         if result['is_improvement'] and result['target_layer'] >= 0]
        significant_tasks = [name for name, result in final_results.items() 
                           if result['is_significant'] and result['target_layer'] >= 0]
        
        stats_msg = f"📊 Overall statistics:"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Total subtasks: {len(final_results)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Successfully found target layer: {len(successful_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Performance improved: {len(improved_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Significant improvement: {len(significant_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Success rate: {len(successful_tasks) / len(final_results) * 100:.1f}%"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - Improvement rate: {len(improved_tasks) / len(successful_tasks) * 100:.1f}%" if successful_tasks else "  - Improvement rate: N/A"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        
        # Performance change statistics
        if successful_tasks:
            all_changes = [final_results[name]['performance_change'] for name in successful_tasks]
            all_changes_pct = [final_results[name]['performance_change_pct'] for name in successful_tasks]
            
            perf_msg = f"\n📈 Performance change statistics:"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - Average performance change: {np.mean(all_changes):+.4f} ({np.mean(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - Maximum performance improvement: {np.max(all_changes):+.4f} ({np.max(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - Maximum performance decrease: {np.min(all_changes):+.4f} ({np.min(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - Performance change standard deviation: {np.std(all_changes):.4f}"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            
            # Baseline performance statistics
            base_performances = [final_results[name]['base_eval_performance'] for name in successful_tasks]
            target_performances = [final_results[name]['target_eval_performance'] for name in successful_tasks]
            
            base_msg = f"\n📊 Baseline performance statistics:"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - Average baseline performance: {np.mean(base_performances):.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - Average processed performance: {np.mean(target_performances):.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - Overall performance change: {np.mean(target_performances) - np.mean(base_performances):+.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
        
        # Detailed results list
        detail_msg = f"\n📋 Detailed results:"
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        detail_msg = "-" * 80
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        detail_msg = f"{'Task Name':<20} {'Target Layer':<6} {'Baseline':<7} {'Processed':<7} {'Change':<10} {'Status':<8}"
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        detail_msg = "-" * 80
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        
        for subtask_name, result in final_results.items():
            target_layer = result['target_layer']
            base_perf = result['base_eval_performance']
            target_perf = result['target_eval_performance']
            change_pct = result['performance_change_pct']
            
            if target_layer >= 0:
                if result['is_improvement']:
                    status = "📈 Improvement" if result['is_significant'] else "📈 Minor impr"
                elif result['performance_change'] < 0:
                    status = "📉 Decrease" if result['is_significant'] else "📉 Minor decr"
                else:
                    status = "➡️ No change"
            else:
                status = "❌ Failed"
                
            detail_msg = f"{subtask_name:<20} {target_layer:<6} {base_perf:<7.3f} {target_perf:<7.3f} {change_pct:+7.2f}% {status:<8}"
            main_logger.info(detail_msg)
            summary_logger.info(detail_msg)
        
        detail_msg = "-" * 80
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        
        # Best and worst results
        if successful_tasks:
            best_task = max(successful_tasks, key=lambda x: final_results[x]['performance_change'])
            worst_task = min(successful_tasks, key=lambda x: final_results[x]['performance_change'])
            
            best_msg = f"\n🏆 Best improvement:"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_result = final_results[best_task]
            best_msg = f"  - Task: {best_task}"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_msg = f"  - Target layer: {best_result['target_layer']}"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_msg = f"  - Performance improvement: {best_result['performance_change']:+.4f} ({best_result['performance_change_pct']:+.2f}%)"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            
            worst_msg = f"\n⚠️ Largest decrease:"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_result = final_results[worst_task]
            worst_msg = f"  - Task: {worst_task}"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_msg = f"  - Target layer: {worst_result['target_layer']}"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_msg = f"  - Performance change: {worst_result['performance_change']:+.4f} ({worst_result['performance_change_pct']:+.2f}%)"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
        
        end_msg = f"\n💾 Results file: {results_file}"
        main_logger.info(end_msg)
        summary_logger.info(end_msg)
        main_logger.info("Probing and evaluation task completed")
        summary_logger.info("Probing and evaluation task completed")
        summary_logger.info("=" * 80)
        
    except Exception as e:
        main_logger.error(f"Main program execution error: {e}")
        main_logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()