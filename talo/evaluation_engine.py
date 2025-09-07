#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch

# Add VLMEvalKit path
VLMEVALKIT_DIR = "dir2vlmevalkit" 
sys.path.append(VLMEVALKIT_DIR)

from vlmeval.dataset import build_dataset
from vlmeval.smp import dump, load


class EvaluationEngine:
    """
    Evaluation engine that fully uses VLMEvalKit mechanisms
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None
    
    def load_dataset(self):
        """Load dataset for evaluation"""
        try:
            self.dataset = build_dataset(self.dataset_name)
            return self.dataset is not None
        except Exception as e:
            print(f"❌ Evaluation engine failed to load dataset: {e}")
            return False
    
    def evaluate_samples_with_cut_layer(self, model_wrapper, samples: List[Dict], 
                                       cut_layer: int = -1, cut_module: str = "self_attn",
                                       work_dir: str = None) -> float:
        """
        Evaluate samples with cut_layer using VLMEvalKit internal mechanisms - load model only once
        """
        if len(samples) == 0:
            print("⚠️ Sample list is empty, returning 0 score")
            return 0.0
        
        try:
            # Create temporary work directory
            if work_dir is None:
                work_dir = f"./temp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            os.makedirs(work_dir, exist_ok=True)
            
            print(f"🚀 Using VLMEvalKit single-load inference (cut_layer={cut_layer}, cut_module={cut_module})")
            
            # Key: Use VLMEvalKit internal mechanisms, load only once
            result = self._run_vlmeval_inference_single_load(
                model_wrapper, samples, cut_layer, cut_module, work_dir
            )
            
            if not result:
                print("❌ VLMEvalKit inference failed")
                return 0.0
            
            # Execute evaluation
            score = self._run_evaluation_vlmeval_style(result, work_dir)
            
            print(f"✅ Evaluation completed, score: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            traceback.print_exc()
            return 0.0

    def _run_vlmeval_inference_single_load(self, model_wrapper, 
                                          samples: List[Dict], cut_layer: int, 
                                          cut_module: str, work_dir: str) -> str:
        """
        Perform inference using VLMEvalKit internal mechanisms - single model load version, optimized for large models
        """
        try:
            print(f"🔧 Preparing VLMEvalKit inference data (sample count: {len(samples)})")
            
            # Build DataFrame
            inference_data = []
            for i, sample in enumerate(samples):
                if 'raw_sample' in sample:
                    data_entry = sample['raw_sample'].copy()
                else:
                    data_entry = {
                        'index': sample.get('index'),
                        'question': sample.get('question', ''),
                        'hint': sample.get('hint', ''),
                        'choices': sample.get('choices', ''),
                        'answer': sample.get('answer', ''),
                        'image': sample.get('image', ''),
                        'category': sample.get('category', ''),
                        'l2-category': sample.get('l2-category', ''),
                    }
                inference_data.append(data_entry)
            
            df = pd.DataFrame(inference_data)
            temp_dataset = self._create_temp_dataset(df)
            
            # Key: Directly call infer_data_job, let it handle model loading, add large model optimization parameters
            from vlmeval.inference import infer_data_job
            
            # Build inference parameters, add large model optimization
            inference_kwargs = {
                'model': model_wrapper.model_name,  # Key: Only pass model name, let VLMEvalKit load
                'model_name': model_wrapper.model_name,
                'dataset': temp_dataset,
                'work_dir': work_dir,
                'verbose': False,
                'api_nproc': 1,  # Key: Large models use single process
                'ignore_failed': True,
            }
            
            # Key: For large models, add special environment variable control
            original_env = {}
            try:
                # Set large model optimization environment variables
                env_settings = {
                    'CUDA_LAUNCH_BLOCKING': '1',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                    'TOKENIZERS_PARALLELISM': 'false'
                }
                
                for key, value in env_settings.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                # Key: If cut_layer parameter is supported, add it
                if cut_layer >= 0:
                    print(f"🔪 Applying cut_layer: layer {cut_layer}, module {cut_module}")
                    inference_kwargs['cut_layer'] = cut_layer
                    inference_kwargs['cut_module'] = cut_module
                else:
                    print(f"📝 Normal inference, no cut_layer")
                
                # Key: Use VLMEvalKit standard method to call inference (single model load)
                print(f"🔄 Starting VLMEvalKit inference, model: {model_wrapper.model_name}")
                result = infer_data_job(**inference_kwargs)
                
            except TypeError as e:
                if 'cut_layer' in str(e):
                    print("⚠️ infer_data_job doesn't support cut_layer parameter, removing parameter and retrying")
                    # Remove cut_layer related parameters and retry
                    inference_kwargs.pop('cut_layer', None)
                    inference_kwargs.pop('cut_module', None)
                    result = infer_data_job(**inference_kwargs)
                    
                    if cut_layer >= 0:
                        print(f"⚠️ Note: cut_layer={cut_layer} parameter not effective, using normal inference")
                else:
                    raise e
            finally:
                # Restore environment variables
                for key, old_value in original_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
            
            # Key: Force GPU memory cleanup
            torch.cuda.empty_cache()
            
            # Find result file
            xlsx_files = [f for f in os.listdir(work_dir) 
                         if f.endswith('.xlsx') and 'temp_input_data' not in f]
            
            if xlsx_files:
                result_file = os.path.join(work_dir, xlsx_files[0])
                print(f"✅ VLMEvalKit inference completed: {result_file}")
                return result_file
            else:
                raise Exception("Result file not found")
                
        except Exception as e:
            print(f"❌ VLMEvalKit inference failed: {e}")
            traceback.print_exc()
            # Key: Clean up GPU memory even on error
            torch.cuda.empty_cache()
            raise e

    def _create_temp_dataset(self, target_df: pd.DataFrame):
        """Create temporary dataset"""
        import copy
        temp_dataset = copy.deepcopy(self.dataset)
        temp_dataset.data = target_df
        
        print(f"✅ Created temporary dataset, sample count: {len(target_df)}")
        return temp_dataset

    def _run_evaluation_vlmeval_style(self, inference_result_file: str, work_dir: str) -> float:
        """Execute evaluation using VLMEvalKit's dataset.evaluate method"""
        try:
            print(f"📊 Using VLMEvalKit standard evaluation process...")
            
            if not os.path.exists(inference_result_file):
                print(f"❌ Inference result file does not exist: {inference_result_file}")
                return 0.0
            
            # Use dataset's evaluate method
            if hasattr(self.dataset, 'evaluate'):
                try:
                    judge_kwargs = self._get_judge_kwargs()
                    print(f"🔄 Starting VLMEvalKit evaluation, parameters: {judge_kwargs}")
                    
                    eval_results = self.dataset.evaluate(inference_result_file, **judge_kwargs)
                    
                    # Save detailed evaluation results
                    eval_result_file = os.path.join(work_dir, "vlmeval_evaluation_results.json")
                    with open(eval_result_file, 'w', encoding='utf-8') as f:
                        if isinstance(eval_results, dict):
                            json.dump(eval_results, f, indent=2, ensure_ascii=False)
                        else:
                            json.dump(eval_results.to_dict(), f, indent=2, ensure_ascii=False)
                    
                    score = self._parse_eval_results(eval_results)
                    return score
                        
                except Exception as e:
                    print(f"⚠️ Using dataset.evaluate failed: {e}")
                    return self._fallback_evaluation(inference_result_file)
            else:
                print("⚠️ Dataset doesn't support evaluate method, using simple evaluation")
                return self._fallback_evaluation(inference_result_file)
                    
        except Exception as e:
            print(f"❌ VLMEvalKit evaluation failed: {e}")
            return self._fallback_evaluation(inference_result_file)

    def _get_judge_kwargs(self) -> Dict:
        """Get evaluation parameters"""
        judge_kwargs = {
            'nproc': 1,  # Key: Large models use single process
            'verbose': True,
            'retry': 3
        }
        
        has_openai_api = os.environ.get('OPENAI_API_KEY')
        
        if has_openai_api:
            if 'MMBench' in self.dataset_name:
                judge_kwargs['model'] = 'exact_matching'
            else:
                judge_kwargs['model'] = 'chatgpt-0125'
        else:
            judge_kwargs['model'] = 'exact_matching'
        
        return judge_kwargs

    def _parse_eval_results(self, eval_results) -> float:
        """Parse VLMEvalKit evaluation results"""
        try:
            if hasattr(eval_results, 'to_dict'):
                eval_dict = eval_results.iloc[0].to_dict() if len(eval_results) == 1 else eval_results.iloc[0].to_dict()
            elif isinstance(eval_results, dict):
                eval_dict = eval_results
            else:
                return 0.0
            
            possible_score_keys = ['Overall', 'overall', 'Accuracy', 'accuracy', 'Score', 'score']
            
            for key in possible_score_keys:
                if key in eval_dict:
                    score_value = eval_dict[key]
                    if isinstance(score_value, (int, float)):
                        return float(score_value) / 100.0 if score_value > 1.0 else float(score_value)
                    elif isinstance(score_value, str):
                        if '%' in score_value:
                            return float(score_value.replace('%', '')) / 100.0
                        else:
                            try:
                                val = float(score_value)
                                return val / 100.0 if val > 1.0 else val
                            except ValueError:
                                continue
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️ Failed to parse evaluation results: {e}")
            return 0.0

    def _fallback_evaluation(self, inference_result_file: str) -> float:
        """Fallback evaluation method"""
        try:
            df = pd.read_excel(inference_result_file)
            if 'prediction' in df.columns and 'answer' in df.columns:
                correct = (df['prediction'].astype(str).str.upper() == df['answer'].astype(str).str.upper()).sum()
                total = len(df)
                return correct / total if total > 0 else 0.0
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️ Fallback evaluation failed: {e}")
            return 0.0