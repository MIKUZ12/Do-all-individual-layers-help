#!/bin/bash
# run_all_experiments.sh - Script to automatically run all experiment combinations

# --- Configuration ---
export VLMEVALKIT_DIR="Your_VLMEvalKit_path"
export HF_HOME="Your_hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_KERNEL_CACHE_PATH="Your_cache_path"

# GPU configuration
export CUDA_VISIBLE_DEVICES='0'

# Switch to the directory where this script is located
cd "$(dirname "$0")" || { echo "Error: Cannot switch to script directory"; exit 1; }

# 🔧 Task configuration - modify here to specify tasks to process

# --- Experiment configuration ---
MODEL_NAME="llava_next_llama3"  

# Define experiment combinations: dataset name and corresponding sampling strategy
declare -A DATASET_STRATEGIES
DATASET_STRATEGIES["MMStar"]="category_l2_stratified"
# Shot combinations - 🔑 Reduce shot count for large models
SHOT_ARRAY=(10)  # Start with small shot count

echo "🚀 Starting large model experiments..."
echo "🎯 Model: $MODEL_NAME (Large model)"
echo "🎯 Target tasks: $TARGET_TASKS"
echo "🔧 GPU memory optimization enabled"
echo "=================================="

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# Calculate total experiments
for DATASET_NAME in "${!DATASET_STRATEGIES[@]}"; do
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + ${#SHOT_ARRAY[@]}))
done

echo "📊 Total experiments: $TOTAL_EXPERIMENTS"
echo "=================================="

# Iterate through all experiment combinations
for DATASET_NAME in "${!DATASET_STRATEGIES[@]}"; do
    SAMPLING_STRATEGY=${DATASET_STRATEGIES[$DATASET_NAME]}
    
    for SHOT in "${SHOT_ARRAY[@]}"; do
        EXPERIMENT_NUM=$((COMPLETED_EXPERIMENTS + 1))
        echo "=================================="
        echo "🧪 Large model experiment $EXPERIMENT_NUM/$TOTAL_EXPERIMENTS"
        echo "  - Model: $MODEL_NAME"
        echo "  - Dataset: $DATASET_NAME"
        echo "  - Shot count: $SHOT"
        echo "  - Sampling strategy: $SAMPLING_STRATEGY"
        echo "  - Target tasks: $TARGET_TASKS"
        echo "=================================="
        
        # Set output directory
        OUTPUT_ROOT="./${SHOT}shot-lora"
        
        # 🔑 Clear GPU memory
        echo "🧹 Pre-clearing GPU memory..."
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # 🔑 Directly execute Python command, not using eval and timeout
        echo "🔄 Executing large model command..."
        echo "python -m talo.main --model_name \"$MODEL_NAME\" --dataset_name \"$DATASET_NAME\" --shot \"$SHOT\" --output_root \"$OUTPUT_ROOT\" --sampling_strategy \"$SAMPLING_STRATEGY\" --target_tasks $TARGET_TASKS"
        echo ""
        
        # 🔑 Fix: Execute directly, not using timeout and eval
        python -m talo.main \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME" \
            --shot "$SHOT" \
            --output_root "$OUTPUT_ROOT" \
            --sampling_strategy "$SAMPLING_STRATEGY" \
        
        EXIT_CODE=$?
        
        # 🔑 Clear GPU memory after experiment
        echo "🧹 Clearing GPU memory after experiment..."
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Large model experiment completed: $DATASET_NAME (Shot: $SHOT)"
            COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
        else
            echo "❌ Large model experiment failed: $DATASET_NAME (Shot: $SHOT) - Exit code: $EXIT_CODE"
            
            # 🔑 Extra cleanup on failure
            echo "🧹 Deep GPU memory cleanup after failure..."
            python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()
" 2>/dev/null || true
            
            # 🔑 Ask whether to continue
            echo ""
            echo "Continue to next experiment? (y/N): "
            read -r -t 30 continue_choice  # 30 second timeout, continue by default
            if [[ ! "$continue_choice" =~ ^[Yy]$ ]] && [[ -n "$continue_choice" ]]; then
                echo "User chose to exit"
                break 2
            fi
        fi
        
        echo "------------------------------"
        echo "Progress: $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS completed"
        echo "=============================="
        echo ""
        
        # 🔑 Wait between large model experiments to allow GPU memory to be fully released
        if [ $EXPERIMENT_NUM -lt $TOTAL_EXPERIMENTS ]; then
            echo "⏳ Waiting between large model experiments for GPU memory release..."
            sleep 10
        fi
    done
done

echo "=================================="
echo "🏁 All large model experiments completed!"
echo "Successfully completed: $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS"
echo "Results saved in corresponding shot directories"
echo "=================================="

# Final cleanup
echo "🧹 Final GPU memory cleanup..."
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()
" 2>/dev/null || true