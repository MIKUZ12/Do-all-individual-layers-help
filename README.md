# Do All Individual Layers Help? An Empirical Study of Task-Interfering Layers in Vision-Language Models

[Project Page](https://mikuz12.github.io/Do_All_Individual_Layers_Help/) [Paper](https://arxiv.org/abs/2602.01167)

## Overview

This repo works by:
1. Probing different layers of a VLM to identify those that have the most positive impact on task performance
2. Evaluating the effectiveness of modifications to these layers
3. Providing insights into model behavior and optimization opportunities

## Project Structure

```
TaLo/
├── talo/                    # Main TaLo package
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Main entry point
│   ├── subtask_extractor.py # Subtask extraction and sampling
│   ├── model_wrapper.py     # Model handling wrapper
│   ├── evaluation_engine.py # Evaluation engine using VLMEvalKit
│   ├── probe_layer.py       # Layer probing functionality
│   ├── final_evaluation.py  # Final performance evaluation
│   └── utils.py             # Utility functions
├── VLMEvalKit/              # Vision-Language Model Evaluation Toolkit
├── Run_TaLo.sh              # Script to run TaLo experiments
├── eval_Interfering_layer.sh # Script to reproduce task interfering layers experiments
└── requirements.txt         # Project dependencies
```

## Environment Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup VLMEvalKit**:
   Make sure VLMEvalKit is properly installed and configured. Follow the instructions in the VLMEvalKit directory.

3. **Environment Variables**:
   Set up the required environment variables in your shell or in the run scripts:
   ```bash
   export VLMEVALKIT_DIR="path/to/VLMEvalKit"
   export HF_HOME="path/to/huggingface/cache"
   export HF_DATASETS_CACHE="${HF_HOME}/datasets"
   export PYTORCH_KERNEL_CACHE_PATH="path/to/pytorch/cache"
   ```

## Usage

### Running TaLo

To run TaLo experiments, use the `Run_TaLo.sh` script:

```bash
./Run_TaLo.sh
```

This script will:
- Configure the environment
- Activate the conda environment
- Run experiments with different models, datasets, and shot counts
- Automatically manage GPU memory between experiments

You can customize the experiments by modifying the variables in the script:
- `MODEL_NAME`: The VLM to evaluate
- `DATASET_STRATEGIES`: Associative array mapping datasets to sampling strategies
- `SHOT_ARRAY`: Different shot counts to evaluate

### Reproducing Task Interfering Layers Experiments

To reproduce the task interfering layers experiments, use the `eval_Interfering_layer.sh` script:

```bash
./eval_Interfering_layer.sh
```

This script will:
- Evaluate model performance with different layers "cut" (disabled)
- Systematically test all layers from 1 to 31
- Save results in a structured format for analysis
- Generate logs and statistics for each experiment

Customize the evaluation by modifying:
- `BASE_MODEL_NAME_OR_PATH`: Path to the model to evaluate
- `LAYERS`: Array of layers to test
- `MODULES`: Modules to cut within each layer
- `tasks`: Datasets to evaluate on

## Sampling Strategies

TaLo supports multiple sampling strategies for subtask selection:
- `l2_priority`: Prioritizes l2-category, falls back to category
- `category_l2_stratified`: Stratified sampling by l2-category within each category
- `l2_category_stratified`: Stratified sampling by category within each l2-category
- `category_random`: Random sampling by category
- `category_skill_stratified`: Stratified sampling by skill within each category
- `skill_stratified`: Sampling by skill category

## Output

TaLo generates comprehensive output including:
- Detailed logs for each experiment phase
- JSON results with performance metrics for each subtask
- Summary statistics across all subtasks
- Identification of target layers for optimization

Results are organized in a hierarchical directory structure based on model name, dataset, and shot count.
