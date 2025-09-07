#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the TaLo system.
"""

import logging
import os
import sys
import re


def setup_logging(output_dir: str):
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "probe_evaluation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def setup_subtask_logging(output_dir: str, subtask_name: str):
    """Setup logging for a specific subtask"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{subtask_name}_evaluation.log")
    
    # Create a new logger instance
    logger = logging.getLogger(f"subtask_{subtask_name}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_stage_logging(work_dir: str, stage_name: str):
    """Setup logging for a specific stage"""
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, f"{stage_name}_log.txt")
    
    # Create a new logger instance
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def filter_tasks(subtasks: dict, args) -> dict:
    """
    Filter tasks based on command line arguments
    
    Args:
        subtasks: All subtasks dictionary
        args: Command line arguments
        
    Returns:
        dict: Filtered subtasks dictionary
    """
    all_task_names = list(subtasks.keys())
    
    # If only listing tasks, return directly
    if args.list_tasks:
        print("📋 All tasks in the dataset:")
        print("-" * 50)
        for i, task_name in enumerate(all_task_names, 1):
            sample_count = len(subtasks[task_name])
            print(f"{i:2d}. {task_name:<30} ({sample_count} samples)")
        print("-" * 50)
        print(f"Total: {len(all_task_names)} tasks")
        return {}
    
    filtered_tasks = {}
    
    # 1. If target_tasks is specified, only process these tasks
    if args.target_tasks:
        print(f"🎯 Specified tasks to process: {args.target_tasks}")
        
        for task_name in args.target_tasks:
            if task_name in subtasks:
                filtered_tasks[task_name] = subtasks[task_name]
                print(f"  ✅ Found task: {task_name} ({len(subtasks[task_name])} samples)")
            else:
                print(f"  ❌ Task not found: {task_name}")
                print(f"     Available tasks: {all_task_names}")
        
        if not filtered_tasks:
            print("❌ No specified tasks found")
            return {}
    
    # 2. If regex pattern matching is used
    elif args.task_pattern:
        print(f"🔍 Using regex pattern matching: {args.task_pattern}")
        
        try:
            pattern = re.compile(args.task_pattern, re.IGNORECASE)
            for task_name in all_task_names:
                if pattern.search(task_name):
                    filtered_tasks[task_name] = subtasks[task_name]
                    print(f"  ✅ Matched task: {task_name} ({len(subtasks[task_name])} samples)")
        except re.error as e:
            print(f"❌ Regex error: {e}")
            return {}
        
        if not filtered_tasks:
            print("❌ No tasks match the specified regex pattern")
            return {}
    
    # 3. Otherwise process all tasks
    else:
        filtered_tasks = subtasks.copy()
        print(f"📊 Processing all tasks ({len(filtered_tasks)} tasks)")
    
    # 4. Apply exclusion list
    if args.exclude_tasks:
        print(f"🚫 Excluding tasks: {args.exclude_tasks}")
        
        for task_name in args.exclude_tasks:
            if task_name in filtered_tasks:
                del filtered_tasks[task_name]
                print(f"  ✅ Excluded: {task_name}")
            else:
                print(f"  ⚠️ Task not found in filtered results: {task_name}")
    
    print(f"🎯 Final tasks to process:")
    for i, (task_name, samples) in enumerate(filtered_tasks.items(), 1):
        print(f"  {i}. {task_name} ({len(samples)} samples)")
    
    return filtered_tasks