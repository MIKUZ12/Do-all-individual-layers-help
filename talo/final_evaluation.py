#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

from talo.utils import setup_stage_logging


def final_evaluation_vlmeval_style(model_wrapper, evaluation_engine,
                                  eval_samples: List[Dict], target_layer: int, work_dir: str) -> Tuple[float, float]:
    """
    Perform final performance evaluation using VLMEvalKit internal mechanisms
    """
    final_logger = setup_stage_logging(work_dir, "final")
    final_logger.info(f"Starting VLMEvalKit-style final performance evaluation (evaluation sample count: {len(eval_samples)}, target layer: {target_layer})")
    
    print(f"🎯 Starting VLMEvalKit-style final performance evaluation (evaluation sample count: {len(eval_samples)}, target layer: {target_layer})")
    
    if len(eval_samples) == 0:
        print("⚠️ Evaluation samples are empty, skipping final evaluation")
        final_logger.warning("Evaluation samples are empty, skipping final evaluation")
        return 0.0, 0.0
    
    # Step 1: Evaluate baseline model (without using cut_layer)
    print(f"📊 Step 1: Evaluating baseline model...")
    final_logger.info(f"Evaluating baseline model...")
    
    base_eval_score = evaluation_engine.evaluate_samples_with_cut_layer(
        model_wrapper, eval_samples,
        cut_layer=-1,  # Don't cut any layer
        work_dir=os.path.join(work_dir, "base_model_eval")
    )
    
    print(f"✅ Baseline model score: {base_eval_score:.4f}")
    final_logger.info(f"Baseline model score: {base_eval_score:.4f}")
    
    # Step 2: Evaluate model after target layer processing
    if target_layer == -1:
        print("⚠️ Target layer is -1, skipping target layer evaluation")
        final_logger.info("Target layer is -1, skipping target layer evaluation")
        target_layer_score = base_eval_score
    else:
        print(f"🔧 Step 2: Using VLMEvalKit cut_layer mechanism to evaluate layer {target_layer}...")
        final_logger.info(f"Using VLMEvalKit cut_layer mechanism to evaluate layer {target_layer}...")
        
        target_layer_score = evaluation_engine.evaluate_samples_with_cut_layer(
            model_wrapper, eval_samples,
            cut_layer=target_layer,
            cut_module="self_attn",
            work_dir=os.path.join(work_dir, "target_layer_eval")
        )
        
        print(f"✅ Score after target layer processing: {target_layer_score:.4f}")
        final_logger.info(f"Score after target layer processing: {target_layer_score:.4f}")
    
    # Performance comparison
    performance_change = target_layer_score - base_eval_score
    performance_change_pct = (performance_change / base_eval_score * 100) if base_eval_score != 0 else 0
    
    print(f"\n📊 VLMEvalKit-style final evaluation results:")
    print(f"  - Baseline model score: {base_eval_score:.4f}")
    print(f"  - Score after target layer processing: {target_layer_score:.4f}")
    print(f"  - Performance change: {performance_change:+.4f} ({performance_change_pct:+.2f}%)")
    
    final_logger.info(f"VLMEvalKit-style final evaluation results:")
    final_logger.info(f"  - Baseline model score: {base_eval_score:.4f}")
    final_logger.info(f"  - Score after target layer processing: {target_layer_score:.4f}")
    final_logger.info(f"  - Performance change: {performance_change:+.4f} ({performance_change_pct:+.2f}%)")
    final_logger.info("VLMEvalKit-style final evaluation phase completed")
    
    return base_eval_score, target_layer_score