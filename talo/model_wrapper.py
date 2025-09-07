#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

# Add VLMEvalKit path
VLMEVALKIT_DIR = "dir2vlmevalkit" 
sys.path.append(VLMEVALKIT_DIR)

from vlmeval.config import supported_VLM


class ModelWrapper:
    """
    Model wrapper that provides a unified inference interface - streamlined version, doesn't manually load model
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # No longer manually load and save model instances
        
    def load_model(self):
        """Verify that the model name is in the supported list"""
        try:
            if self.model_name not in supported_VLM:
                available_models = list(supported_VLM.keys())
                print(f"❌ Model {self.model_name} not in supported list")
                print(f"Supported models: {available_models[:10]}{'...' if len(available_models) > 10 else ''}")
                return False
            
            print(f"✅ Model {self.model_name} is in supported list, will be loaded on-demand by VLMEvalKit")
            return True
        except Exception as e:
            print(f"❌ Error verifying model {self.model_name}: {e}")
            return False
    
    def get_num_layers(self) -> int:
        """Get the number of model layers - using preset values"""
        # Since we don't manually load the model, use preset layer mappings
        layer_mapping = {
            'InternVL2-26B': 48,
            'InternVL2-8B': 32,
            'InternVL2-4B': 32,
            'InternVL2-2B': 24,
            'Qwen2-VL-2B-Instruct': 28,
            'Qwen2-VL-7B-Instruct': 32,
            'Qwen2-VL-72B-Instruct': 80,
            'llava_next_llama3': 32,
            'llava_next_yi_34b': 60,
            'llava_next_mistral_7b': 32,
            'LLaVA-OneVision-Qwen2-0.5B-SI': 24,
            'LLaVA-OneVision-Qwen2-7B-SI': 32,
            'LLaVA-OneVision-Qwen2-72B-SI': 80,
            'MiniCPM-V-2_6': 40,
            'MiniCPM-Llama3-V-2_5': 32,
            # Add more model layer mappings
        }
        
        num_layers = layer_mapping.get(self.model_name, 32)  # Default 32 layers
        print(f"✅ Using preset layers: {self.model_name} -> {num_layers} layers")
        return num_layers