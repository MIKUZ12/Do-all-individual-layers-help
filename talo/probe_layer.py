#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, List, Tuple

from talo.utils import setup_stage_logging


def probe_target_layer_vlmeval_enhanced(model_wrapper, evaluation_engine,
                                       probe_samples: List[Dict], work_dir: str, 
                                       all_samples: List[Dict] = None, shot: int = 5) -> Tuple[int, float, Dict]:
    """
    Probe target layer - VLMEvalKit style but maintains complete logic
    """
    print(f"🔍 Starting VLMEvalKit-style target layer probing (probe sample count: {len(probe_samples)})")
    
    probe_logger = setup_stage_logging(work_dir, "probe")
    probe_logger.info(f"Starting VLMEvalKit-style target layer probing (probe sample count: {len(probe_samples)})")
    
    # Get model layers
    num_layers = model_wrapper.get_num_layers()
    probe_logger.info(f"Model total layers: {num_layers}")
    print(f"📊 Model total layers: {num_layers}")
    
    # Keep original resampling logic
    resample_round = 0
    max_resample_rounds = 5
    current_probe_samples = probe_samples.copy()
    used_sample_indices = set()
    
    # If all_samples provided, record currently used sample indices
    if all_samples:
        current_indices = {sample.get('index', -1) for sample in current_probe_samples}
        used_sample_indices.update(current_indices)
    
    target_layer = -1
    best_improvement = -float('inf')
    layer_scores = {}
    base_score = 0.0
    
    while resample_round <= max_resample_rounds:
        round_suffix = f"_round_{resample_round}" if resample_round > 0 else ""
        round_work_dir = work_dir if resample_round == 0 else os.path.join(work_dir, f"resample_round_{resample_round}")
        
        if resample_round > 0:
            os.makedirs(round_work_dir, exist_ok=True)
            print(f"🔄 Resampling round {resample_round}")
            probe_logger.info(f"Resampling round {resample_round}")
        
        # Key: Phase 1 - Baseline evaluation (using VLMEvalKit method)
        print(f"📊 Executing baseline evaluation{round_suffix}...")
        probe_logger.info(f"Executing baseline evaluation{round_suffix}...")
        
        base_eval_dir = os.path.join(round_work_dir, f"base_eval{round_suffix}")
        base_score = evaluation_engine.evaluate_samples_with_cut_layer(
            model_wrapper, current_probe_samples, 
            cut_layer=-1,  # -1 means don't cut any layer
            work_dir=base_eval_dir
        )
        print(f"✅ Baseline score{round_suffix}: {base_score:.4f}")
        probe_logger.info(f"Baseline score{round_suffix}: {base_score:.4f}")
        
        # Key: First priority - Check if baseline score is 1.0 (keep original logic)
        if (resample_round == 0 and base_score >= 1.0 and all_samples and 
            resample_round < max_resample_rounds):
            print(f"⚠️ Baseline score is {base_score:.4f}, trying resampling...")
            probe_logger.info(f"Baseline score is {base_score:.4f}, trying resampling...")
            
            # Resample: Select from unused samples
            available_samples = [sample for sample in all_samples 
                               if sample.get('index', -1) not in used_sample_indices]
            
            if len(available_samples) >= shot:
                # Randomly select new samples
                import random
                new_probe_samples = random.sample(available_samples, shot)
                
                # Update used sample indices
                new_indices = {sample.get('index', -1) for sample in new_probe_samples}
                used_sample_indices.update(new_indices)
                
                current_probe_samples = new_probe_samples
                resample_round += 1
                
                print(f"🔄 Baseline score is 1, resampling successful, new sample indices: {sorted(list(new_indices))}")
                probe_logger.info(f"Baseline score is 1, resampling successful, new sample indices: {sorted(list(new_indices))}")
                continue
            else:
                print(f"⚠️ Insufficient available samples ({len(available_samples)} < {shot}), continuing with current samples")
                probe_logger.warning(f"Insufficient available samples ({len(available_samples)} < {shot}), continuing with current samples")
        
        # Key: Phase 2 - Layer-by-layer evaluation (using VLMEvalKit method)
        layer_scores = {}
        layer_improvements = {}
        
        final_round_suffix = f"_final_round_{resample_round}" if resample_round > 0 else ""
        print(f"🔄 Starting layer-by-layer evaluation (total {num_layers} layers){final_round_suffix}...")
        probe_logger.info(f"Starting layer-by-layer evaluation (total {num_layers} layers){final_round_suffix}...")
        
        for layer_idx in range(1, num_layers):
            try:
                layer_work_dir = os.path.join(round_work_dir, f"layer_{layer_idx}_eval{final_round_suffix}")
                layer_logger = setup_stage_logging(layer_work_dir, f"layer_{layer_idx}{final_round_suffix}")
                layer_logger.info(f"Evaluating layer {layer_idx}{final_round_suffix}...")
                print(f"  📊 Evaluating layer {layer_idx}{final_round_suffix}...")
                
                # Key: Use VLMEvalKit's cut_layer mechanism
                layer_score = evaluation_engine.evaluate_samples_with_cut_layer(
                    model_wrapper, current_probe_samples,
                    cut_layer=layer_idx,
                    cut_module="self_attn",
                    work_dir=layer_work_dir
                )
                
                layer_scores[layer_idx] = layer_score
                improvement = layer_score - base_score
                layer_improvements[layer_idx] = improvement
                
                print(f"    Layer {layer_idx} score: {layer_score:.4f} (improvement: {improvement:+.4f})")
                layer_logger.info(f"Layer {layer_idx} score: {layer_score:.4f} (improvement: {improvement:+.4f})")
                probe_logger.info(f"Layer {layer_idx} score: {layer_score:.4f} (improvement: {improvement:+.4f})")
                
            except Exception as e:
                print(f"❌ Error evaluating layer {layer_idx}: {e}")
                probe_logger.error(f"Error evaluating layer {layer_idx}: {e}")
                layer_scores[layer_idx] = 0.0
                layer_improvements[layer_idx] = -float('inf')
                continue
        
        # Key: Second priority - Find best improvement (keep original tie resolution logic)
        positive_improvements = {layer: improvement for layer, improvement 
                               in layer_improvements.items() if improvement > 0}
        
        if positive_improvements:
            # Find maximum improvement value
            best_improvement = max(positive_improvements.values())
            
            # Find all layers with maximum improvement value
            best_layers = [layer for layer, improvement in positive_improvements.items() 
                        if abs(improvement - best_improvement) < 1e-6]
            
            if len(best_layers) == 1:
                # Single target layer
                target_layer = best_layers[0]
                print(f"🎯 Found single target layer: {target_layer} (improvement: {best_improvement:+.4f})")
                probe_logger.info(f"Found single target layer: {target_layer} (improvement: {best_improvement:+.4f})")
                break  # Directly exit resampling loop
            else:
                # Key: Enhanced handling of multi-layer tie (keep original logic)
                print(f"🔍 Found multiple layers({best_layers}) with same best improvement {best_improvement:+.4f}")
                probe_logger.info(f"Found multiple layers({best_layers}) with same best improvement {best_improvement:+.4f}")
                
                # For now, we'll just break without tie resolution
                target_layer = best_layers[0]
                print(f"🎯 Selecting first layer as target: {target_layer}")
                probe_logger.info(f"Selecting first layer as target: {target_layer}")
                break
        else:
            # Key: Third priority - No layers with positive improvement, try resampling (keep original logic)
            best_improvement = max(layer_improvements.values()) if layer_improvements else -float('inf')
            print(f"⚠️ No layers produced positive improvement, best improvement: {best_improvement:+.4f}")
            probe_logger.warning(f"No layers produced positive improvement, best improvement: {best_improvement:+.4f}")
            
            if all_samples and resample_round < max_resample_rounds:
                available_samples = [sample for sample in all_samples 
                                   if sample.get('index', -1) not in used_sample_indices]
                
                if len(available_samples) >= shot:
                    import random
                    new_probe_samples = random.sample(available_samples, shot + resample_round + 1)
                    
                    # Update used sample indices
                    new_indices = {sample.get('index', -1) for sample in new_probe_samples}
                    used_sample_indices.update(new_indices)
                    
                    current_probe_samples = new_probe_samples
                    resample_round += 1
                    
                    print(f"🔄 Target layer not found, resampling round {resample_round}, new sample indices: {sorted(list(new_indices))}")
                    probe_logger.info(f"Target layer not found, resampling round {resample_round}, new sample indices: {sorted(list(new_indices))}")
                    continue
                else:
                    print(f"⚠️ Insufficient available samples ({len(available_samples)} < {shot}), stopping resampling")
                    probe_logger.warning(f"Insufficient available samples ({len(available_samples)} < {shot}), stopping resampling")
                    target_layer = -1
                    break
            else:
                target_layer = -1
                break
    
    # Result summary
    if target_layer == -1:
        print(f"⚠️ Still no valid target layer found after {resample_round} sampling rounds")
        probe_logger.warning(f"Still no valid target layer found after {resample_round} sampling rounds")
    
    details = {
        'probe_base_score': base_score,
        'layer_scores': layer_scores,
        'layer_improvements': layer_improvements if 'layer_improvements' in locals() else {},
        'best_improvement': best_improvement,
        'best_probe_score': layer_scores.get(target_layer, base_score) if target_layer >= 0 else base_score,
        'resample_rounds': resample_round,
        'final_probe_samples_indices': [sample.get('index', -1) for sample in current_probe_samples],
        'vlmeval_style': True
    }
    
    probe_logger.info("VLMEvalKit-style probing phase completed")
    return target_layer, base_score, details