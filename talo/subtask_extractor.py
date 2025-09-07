#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
import random
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# Add VLMEvalKit path
VLMEVALKIT_DIR = "dir2vlmevalkit" 
import sys
sys.path.append(VLMEVALKIT_DIR)

from vlmeval.dataset import build_dataset


def stratified_sampling_by_skill_in_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    Stratified sampling by skill within each category
    That is: within each category, stratified sampling is performed according to the proportion of skills
    
    Args:
        samples: Sample list
        shot: Total sampling quantity
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (sampled samples, stratified sampling statistics for each category)
    """
    # Group by category
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 Stratified sampling by skill within each category:")
    print(f"  - Found {len(category_to_samples)} categories")
    
    # Allocate sampling quantity for each category
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # Last category gets remaining shots
            category_shots[category] = remaining_shot
        else:
            # Allocate proportionally
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # Not exceeding the number of samples in that category
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # Stratified sampling within each category by skill
    final_samples = []
    detailed_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            detailed_stats[category] = {}
            continue
            
        print(f"  - {category}: Allocated {allocated_shot} samples")
        
        # Count skill distribution within this category
        skill_to_samples = defaultdict(list)
        for sample in cat_samples:
            # Get skill field from raw_sample
            skill = sample.get('raw_sample', {}).get('skills', 'unknown')
            if pd.isna(skill) or skill == '':
                skill = 'unknown'
            skill_to_samples[skill].append(sample)
        
        print(f"    Contains {len(skill_to_samples)} skills")
        
        # Allocate sampling quantity for each skill
        skill_shots = {}
        remaining_in_category = allocated_shot
        skill_list = list(skill_to_samples.keys())
        
        for i, skill in enumerate(skill_list):
            if i == len(skill_list) - 1:
                skill_shots[skill] = remaining_in_category
            else:
                proportion = len(skill_to_samples[skill]) / len(cat_samples)
                skill_allocated = int(round(proportion * allocated_shot))
                skill_allocated = min(skill_allocated, len(skill_to_samples[skill]))
                skill_shots[skill] = skill_allocated
                remaining_in_category -= skill_allocated
        
        # Sample within category by skill
        category_sampled = []
        skill_stats = {}
        
        for skill, skill_shot in skill_shots.items():
            if skill_shot > 0:
                skill_samples = skill_to_samples[skill]
                random.shuffle(skill_samples)
                selected = skill_samples[:skill_shot]
                category_sampled.extend(selected)
                skill_stats[skill] = len(selected)
                print(f"    - {skill}: {len(selected)} samples")
        
        final_samples.extend(category_sampled)
        detailed_stats[category] = skill_stats
    
    print(f"  - Total sampled: {len(final_samples)} samples")
    return final_samples, detailed_stats


def stratified_sampling_by_skill(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Sampling by skill category
    That is: allocate sampling quantity according to the proportion of skills, then randomly sample within each skill
    
    Args:
        samples: Sample list
        shot: Total sampling quantity
        
    Returns:
        Tuple[List[Dict], Dict[str, int]]: (sampled samples, skill sampling quantity statistics)
    """
    # Group by skill
    skill_to_samples = defaultdict(list)
    for sample in samples:
        # Get skill field from raw_sample
        skill = sample.get('raw_sample', {}).get('skills', 'unknown')
        if pd.isna(skill) or skill == '':
            skill = 'unknown'
        skill_to_samples[skill].append(sample)
    
    print(f"📊 Sampling by skill category:")
    print(f"  - Found {len(skill_to_samples)} skills")
    
    # Count distribution of each skill
    for skill, skill_samples in skill_to_samples.items():
        proportion = len(skill_samples) / len(samples)
        print(f"    - {skill}: {len(skill_samples)} samples ({proportion:.1%})")
    
    # Allocate sampling quantity for each skill
    total_samples = len(samples)
    skill_shots = {}
    remaining_shot = shot
    
    for i, (skill, skill_samples) in enumerate(skill_to_samples.items()):
        if i == len(skill_to_samples) - 1:
            # Last skill gets remaining shots
            skill_shots[skill] = remaining_shot
        else:
            # Allocate proportionally
            proportion = len(skill_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(skill_samples))  # Not exceeding the number of samples in that skill
            skill_shots[skill] = allocated_shot
            remaining_shot -= allocated_shot
    
    # Randomly sample within each skill
    final_samples = []
    sampling_stats = {}
    
    for skill, allocated_shot in skill_shots.items():
        skill_samples = skill_to_samples[skill]
        
        if allocated_shot == 0:
            sampling_stats[skill] = 0
            continue
            
        print(f"  - {skill}: Allocated {allocated_shot} samples")
        
        # Random sampling
        random.shuffle(skill_samples)
        selected = skill_samples[:allocated_shot]
        final_samples.extend(selected)
        sampling_stats[skill] = len(selected)
    
    print(f"  - Total sampled: {len(final_samples)} samples")
    return final_samples, sampling_stats


def stratified_sampling_by_l2_in_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    Stratified sampling by l2-category within each category
    That is: within each category, stratified sampling is performed according to the proportion of l2-categories
    
    Args:
        samples: Sample list
        shot: Total sampling quantity
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (sampled samples, stratified sampling statistics for each category)
    """
    # Group by category
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 Stratified sampling by l2-category within each category:")
    print(f"  - Found {len(category_to_samples)} categories")
    
    # Allocate sampling quantity for each category
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # Last category gets remaining shots
            category_shots[category] = remaining_shot
        else:
            # Allocate proportionally
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # Not exceeding the number of samples in that category
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # Stratified sampling within each category by l2-category
    final_samples = []
    detailed_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            detailed_stats[category] = {}
            continue
            
        print(f"  - {category}: Allocated {allocated_shot} samples")
        
        # Count l2-category distribution within this category
        l2cat_to_samples = defaultdict(list)
        for sample in cat_samples:
            # Handle field name differences
            l2cat = sample.get('l2-category', sample.get('l2_category', 'unknown'))
            l2cat_to_samples[l2cat].append(sample)
        
        print(f"    Contains {len(l2cat_to_samples)} l2-categories")
        
        # Allocate sampling quantity for each l2-category
        l2cat_shots = {}
        remaining_in_category = allocated_shot
        l2cat_list = list(l2cat_to_samples.keys())
        
        for i, l2cat in enumerate(l2cat_list):
            if i == len(l2cat_list) - 1:
                l2cat_shots[l2cat] = remaining_in_category
            else:
                proportion = len(l2cat_to_samples[l2cat]) / len(cat_samples)
                l2_allocated = int(round(proportion * allocated_shot))
                l2_allocated = min(l2_allocated, len(l2cat_to_samples[l2cat]))
                l2cat_shots[l2cat] = l2_allocated
                remaining_in_category -= l2_allocated
        
        # Sample within category by l2-category
        category_sampled = []
        l2_stats = {}
        
        for l2cat, l2_shot in l2cat_shots.items():
            if l2_shot > 0:
                l2_samples = l2cat_to_samples[l2cat]
                random.shuffle(l2_samples)
                selected = l2_samples[:l2_shot]
                category_sampled.extend(selected)
                l2_stats[l2cat] = len(selected)
                print(f"    - {l2cat}: {len(selected)} samples")
        
        final_samples.extend(category_sampled)
        detailed_stats[category] = l2_stats
    
    print(f"  - Total sampled: {len(final_samples)} samples")
    return final_samples, detailed_stats


def stratified_sampling_by_category_in_l2(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    Stratified sampling by category within each l2-category
    That is: within each l2-category, stratified sampling is performed according to the proportion of categories
    
    Args:
        samples: Sample list
        shot: Total sampling quantity
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (sampled samples, stratified sampling statistics for each category)
    """
    # Group by l2-category, handle field name differences
    l2cat_to_samples = defaultdict(list)
    for sample in samples:
        l2cat = sample.get('l2-category', sample.get('l2_category', 'unknown'))
        l2cat_to_samples[l2cat].append(sample)
    
    print(f"📊 Stratified sampling by category within each l2-category:")
    print(f"  - Found {len(l2cat_to_samples)} l2-categories")
    
    # Allocate sampling quantity for each l2-category
    total_samples = len(samples)
    l2cat_shots = {}
    remaining_shot = shot
    
    for i, (l2cat, l2_samples) in enumerate(l2cat_to_samples.items()):
        if i == len(l2cat_to_samples) - 1:
            # Last l2-category gets remaining shots
            l2cat_shots[l2cat] = remaining_shot
        else:
            # Allocate proportionally
            proportion = len(l2_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(l2_samples))  # Not exceeding the number of samples in that l2-category
            l2cat_shots[l2cat] = allocated_shot
            remaining_shot -= allocated_shot
    
    # Stratified sampling within each l2-category by category
    final_samples = []
    detailed_stats = {}
    
    for l2cat, allocated_shot in l2cat_shots.items():
        l2_samples = l2cat_to_samples[l2cat]
        
        if allocated_shot == 0:
            detailed_stats[l2cat] = {}
            continue
            
        print(f"  - {l2cat}: Allocated {allocated_shot} samples")
        
        # Count category distribution within this l2-category
        cat_to_samples = defaultdict(list)
        for sample in l2_samples:
            category = sample.get('category', 'unknown')
            cat_to_samples[category].append(sample)
        
        print(f"    Contains {len(cat_to_samples)} categories")
        
        # Allocate sampling quantity for each category
        cat_shots = {}
        remaining_in_l2cat = allocated_shot
        cat_list = list(cat_to_samples.keys())
        
        for i, category in enumerate(cat_list):
            if i == len(cat_list) - 1:
                cat_shots[category] = remaining_in_l2cat
            else:
                proportion = len(cat_to_samples[category]) / len(l2_samples)
                cat_allocated = int(round(proportion * allocated_shot))
                cat_allocated = min(cat_allocated, len(cat_to_samples[category]))
                cat_shots[category] = cat_allocated
                remaining_in_l2cat -= cat_allocated
        
        # Sample within l2-category by category
        l2cat_sampled = []
        cat_stats = {}
        
        for category, cat_shot in cat_shots.items():
            if cat_shot > 0:
                cat_samples = cat_to_samples[category]
                random.shuffle(cat_samples)
                selected = cat_samples[:cat_shot]
                l2cat_sampled.extend(selected)
                cat_stats[category] = len(selected)
                print(f"    - {category}: {len(selected)} samples")
        
        final_samples.extend(l2cat_sampled)
        detailed_stats[l2cat] = cat_stats
    
    print(f"  - Total sampled: {len(final_samples)} samples")
    return final_samples, detailed_stats


def random_sampling_by_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Random sampling by category
    That is: allocate sampling quantity according to the proportion of categories, then randomly sample within each category
    
    Args:
        samples: Sample list
        shot: Total sampling quantity
        
    Returns:
        Tuple[List[Dict], Dict[str, int]]: (sampled samples, category sampling quantity statistics)
    """
    # Group by category
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 Random sampling by category:")
    print(f"  - Found {len(category_to_samples)} categories")
    
    # Allocate sampling quantity for each category
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # Last category gets remaining shots
            category_shots[category] = remaining_shot
        else:
            # Allocate proportionally
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # Not exceeding the number of samples in that category
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # Randomly sample within each category
    final_samples = []
    sampling_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            sampling_stats[category] = 0
            continue
            
        print(f"  - {category}: Allocated {allocated_shot} samples")
        
        # Random sampling
        random.shuffle(cat_samples)
        selected = cat_samples[:allocated_shot]
        final_samples.extend(selected)
        sampling_stats[category] = len(selected)
        print(f"    - Actually sampled: {len(selected)} samples")
    
    print(f"  - Total sampled: {len(final_samples)} samples")
    return final_samples, sampling_stats


class SubtaskExtractor:
    def __init__(self, dataset_name: str, sampling_strategy: str = 'l2_priority'):
        """
        Initialize subtask extractor
        
        Args:
            dataset_name: Dataset name
            sampling_strategy: Sampling strategy
                - 'l2_priority': Prioritize l2-category, fallback to category (original logic)
                - 'category_l2_stratified': Stratified sampling by l2-category within each category
                - 'l2_category_stratified': Stratified sampling by category within each l2-category
                - 'category_random': Random sampling by category
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.index_mapping = {}  # Original index -> continuous index
        self.reverse_mapping = {}  # Continuous index -> original index
        self.primary_category_field = None  # Record primary classification field
        self.sampling_strategy = sampling_strategy
        
    def load_dataset(self):
        """Load dataset and create index mapping"""
        try:
            self.dataset = build_dataset(self.dataset_name)
            if self.dataset is None:
                raise ValueError(f"Dataset {self.dataset_name} build failed")
            
            # Check dataset classification fields
            self._analyze_category_fields()
            
            # Create index mapping
            original_indices = sorted(self.dataset.data['index'].unique())
            self.index_mapping = {orig: new for new, orig in enumerate(original_indices)}
            self.reverse_mapping = {new: orig for orig, new in self.index_mapping.items()}
            
            print(f"🔄 Creating index mapping:")
            print(f"  - Original index range: {min(original_indices)} - {max(original_indices)}")
            print(f"  - Mapped index range: 0 - {len(original_indices) - 1}")
            print(f"  - Mapping example: {list(self.index_mapping.items())[:5]}")
            
            print(f"✅ Dataset {self.dataset_name} loaded successfully")
            print(f"📋 Sampling strategy: {self.sampling_strategy}")
            return True
        except Exception as e:
            print(f"❌ Dataset {self.dataset_name} loading failed: {e}")
            return False
    
    def _analyze_category_fields(self):
        """Analyze dataset classification fields to determine which field to use as primary"""
        data_columns = self.dataset.data.columns.tolist()
        
        # Check for possible field names
        has_l2_category = 'l2-category' in data_columns or 'l2_category' in data_columns
        has_category = 'category' in data_columns
        
        # Check skill field
        has_skill = False
        if len(self.dataset.data) > 0:
            try:
                # Check first few samples' raw_sample for skill field
                for i in range(min(5, len(self.dataset.data))):
                    sample_data = self.dataset.data.iloc[i]
                    # If it's dict format
                    if isinstance(sample_data.get('raw_sample'), dict):
                        if 'skills' in sample_data['raw_sample']:
                            has_skill = True
                            break
                    # If raw_sample is JSON string format, try parsing
                    elif isinstance(sample_data.get('raw_sample'), str):
                        try:
                            raw_data = json.loads(sample_data['raw_sample'])
                            if 'skills' in raw_data:
                                has_skill = True
                                break
                        except:
                            continue
                    # Directly check if there's a skill column
                    elif 'skills' in sample_data:
                        has_skill = True
                        break
            except Exception as e:
                print(f"⚠️ Error checking skill field: {e}")
                has_skill = False
        
        print(f"📊 Analyzing dataset classification fields:")
        print(f"  - Contains 'l2-category' or 'l2_category': {has_l2_category}")
        print(f"  - Contains 'category': {has_category}")
        print(f"  - 'skills' in raw_sample: {has_skill}")
        
        # Determine primary classification field based on sampling strategy
        if self.sampling_strategy == 'l2_priority':
            # Original logic: prioritize l2-category
            if has_l2_category:
                # Determine actual field name used
                l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
                l2_cat_values = self.dataset.data[l2_field_name].dropna().unique()
                if len(l2_cat_values) > 0 and not all(v == 'unknown' for v in l2_cat_values):
                    self.primary_category_field = l2_field_name
                    print(f"  ✅ Using '{l2_field_name}' as primary classification field")
                    print(f"  - l2-category unique values: {len(l2_cat_values)}")
                    print(f"  - l2-category examples: {list(l2_cat_values)[:5]}")
                elif has_category:
                    self.primary_category_field = 'category'
                    print(f"  ⚠️ l2-category has no valid values, fallback to 'category'")
                else:
                    self.primary_category_field = None
                    print(f"  ❌ No valid classification fields")
            elif has_category:
                self.primary_category_field = 'category'
                print(f"  ✅ Using 'category' as primary classification field")
                cat_values = self.dataset.data['category'].dropna().unique()
                print(f"  - category unique values: {len(cat_values)}")
                print(f"  - category examples: {list(cat_values)[:5]}")
            else:
                self.primary_category_field = None
                print(f"  ❌ Dataset does not contain 'category' or 'l2-category' fields")
        
        elif self.sampling_strategy in ['category_l2_stratified', 'category_random', 'category_skill_stratified']:
            # Category-prioritized strategies
            if has_category:
                self.primary_category_field = 'category'
                print(f"  ✅ Using 'category' as primary classification field (strategy: {self.sampling_strategy})")
                cat_values = self.dataset.data['category'].dropna().unique()
                print(f"  - category unique values: {len(cat_values)}")
                print(f"  - category examples: {list(cat_values)[:5]}")
                
                # For skill-related strategies, additionally check skill field
                if self.sampling_strategy == 'category_skill_stratified':
                    if has_skill:
                        print(f"  ✅ raw_sample contains skill field, supporting skill stratified sampling")
                        # Get skill value examples
                        skill_examples = []
                        skill_counts = {}
                        for i in range(min(50, len(self.dataset.data))):  # Check more samples to get complete skill distribution
                            try:
                                sample_data = self.dataset.data.iloc[i]
                                if isinstance(sample_data.get('raw_sample'), dict):
                                    skill = sample_data['raw_sample'].get('skills', 'unknown')
                                else:
                                    skill = 'unknown'
                                
                                if skill != 'unknown':
                                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
                                    if skill not in skill_examples:
                                        skill_examples.append(skill)
                            except:
                                continue
                        
                        print(f"  - skill types: {len(skill_examples)}")
                        print(f"  - skill examples: {skill_examples[:10]}")  # Show first 10
                        if skill_counts:
                            print(f"  - skill distribution examples: {dict(list(skill_counts.items())[:5])}")
                    else:
                        print(f"  ⚠️ Strategy '{self.sampling_strategy}' requires 'skill' field, but not found in raw_sample")
            else:
                self.primary_category_field = None
                print(f"  ❌ Strategy '{self.sampling_strategy}' requires 'category' field, but not found in dataset")
        
        elif self.sampling_strategy == 'l2_category_stratified':
            # l2-category-prioritized strategy
            if has_l2_category:
                # Determine actual field name used
                l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
                l2_cat_values = self.dataset.data[l2_field_name].dropna().unique()
                if len(l2_cat_values) > 0 and not all(v == 'unknown' for v in l2_cat_values):
                    self.primary_category_field = l2_field_name
                    print(f"  ✅ Using '{l2_field_name}' as primary classification field (strategy: {self.sampling_strategy})")
                    print(f"  - l2-category unique values: {len(l2_cat_values)}")
                    print(f"  - l2-category examples: {list(l2_cat_values)[:5]}")
                else:
                    self.primary_category_field = None
                    print(f"  ❌ Strategy '{self.sampling_strategy}' requires valid 'l2-category' field, but no valid values in dataset")
            else:
                self.primary_category_field = None
                print(f"  ❌ Strategy '{self.sampling_strategy}' requires 'l2-category' field, but not found in dataset")
        
        elif self.sampling_strategy == 'skill_stratified':
            # Skill-prioritized strategy
            if has_skill:
                self.primary_category_field = 'skills'
                print(f"  ✅ Using 'skill' as primary classification field (strategy: {self.sampling_strategy})")
                # Get skill value examples
                skill_examples = []
                skill_counts = {}
                for i in range(min(50, len(self.dataset.data))):  # Check more samples to get complete skill distribution
                    try:
                        sample_data = self.dataset.data.iloc[i]
                        if isinstance(sample_data.get('raw_sample'), dict):
                            skill = sample_data['raw_sample'].get('skills', 'unknown')
                        else:
                            skill = 'unknown'
                        
                        if skill != 'unknown':
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
                            if skill not in skill_examples:
                                skill_examples.append(skill)
                    except:
                        continue
                
                print(f"  - skill types: {len(skill_examples)}")
                print(f"  - skill examples: {skill_examples[:10]}")  # Show first 10
                if skill_counts:
                    print(f"  - skill distribution examples: {dict(list(skill_counts.items())[:5])}")
            else:
                self.primary_category_field = None
                print(f"  ❌ Strategy '{self.sampling_strategy}' requires 'skill' field, but not found in raw_sample")

    def extract_subtasks(self) -> Dict[str, List[Dict]]:
        """Extract subtasks, using mapped continuous indices, determine classification method based on sampling strategy"""
        if self.dataset is None:
            print("❌ Dataset not loaded, cannot extract subtasks")
            return {}
        
        if self.primary_category_field is None:
            print("❌ No valid classification field, cannot extract subtasks")
            return {}
        
        # Determine actual l2-category field name
        data_columns = self.dataset.data.columns.tolist()
        l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
        l2_field_exists = l2_field_name in data_columns
        
        subtasks = {}
        total_samples = len(self.dataset.data)
        print(f"📊 Using '{self.primary_category_field}' field to process {total_samples} samples...")
        print(f"📋 Sampling strategy: {self.sampling_strategy}")
        
        for idx in range(total_samples):
            try:
                sample_dict = self.dataset.data.iloc[idx].to_dict()
                
                # Get original index and map to continuous index
                original_index = sample_dict.get('index')
                mapped_index = self.index_mapping.get(original_index, idx)
                
                # Get task category based on primary classification field
                if self.primary_category_field == 'skills':
                    # Get skill field from raw_sample
                    if isinstance(sample_dict.get('raw_sample'), dict):
                        subtask_name = sample_dict['raw_sample'].get('skills', 'unknown')
                    else:
                        subtask_name = 'unknown'
                else:
                    # Use other classification fields
                    subtask_name = sample_dict.get(self.primary_category_field, 'unknown')
                
                if subtask_name == 'unknown' or pd.isna(subtask_name):
                    # If primary field is invalid, try backup field
                    if self.primary_category_field == 'skills':
                        # If skill strategy fails, try category field
                        backup_field = 'category'
                    elif self.primary_category_field in ['l2-category', 'l2_category']:
                        backup_field = 'category'
                    else:
                        backup_field = l2_field_name  # Use actual l2 field name
                    
                    if backup_field in sample_dict:
                        subtask_name = sample_dict.get(backup_field, 'unknown')
                    
                    if subtask_name == 'unknown' or pd.isna(subtask_name):
                        continue
                
                if subtask_name not in subtasks:
                    subtasks[subtask_name] = []
                
                # Create sample data using mapped continuous index
                sample_data = {
                    'index': mapped_index,  # Use mapped continuous index
                    'original_index': original_index,  # Keep original index for tracing
                    'category': sample_dict.get('category', 'unknown'),
                    'l2-category': sample_dict.get(l2_field_name, 'unknown') if l2_field_exists else 'unknown',
                    'primary_category': subtask_name,  # Record primary category used
                    'primary_category_field': self.primary_category_field,  # Record field used
                    'sampling_strategy': self.sampling_strategy,  # Record sampling strategy
                    'question': sample_dict.get('question', ''),
                    'choices': sample_dict.get('choices', ''),
                    'answer': sample_dict.get('answer', ''),
                    'image': sample_dict.get('image', ''),
                    'raw_sample': sample_dict.copy()
                }
                
                # Update index in raw_sample to mapped index
                sample_data['raw_sample']['index'] = mapped_index
                
                subtasks[subtask_name].append(sample_data)
                
            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue
        
        # Statistics
        print(f"📊 Subtask extraction completed:")
        print(f"  - Using classification field: {self.primary_category_field}")
        print(f"  - Sampling strategy: {self.sampling_strategy}")
        print(f"  - Extracted {len(subtasks)} subtasks")
        for task_name, samples in subtasks.items():
            print(f"  - {task_name}: {len(samples)} samples")
        
        return subtasks
    
    def apply_sampling_strategy(self, samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict]:
        """
        Sample samples according to sampling strategy
        
        Args:
            samples: Sample list
            shot: Sampling quantity
            
        Returns:
            Tuple[List[Dict], Dict]: (sampled samples, sampling statistics)
        """
        if self.sampling_strategy == 'l2_priority':
            # Original logic: simple random sampling
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)
            sampled = shuffled_samples[:shot]
            stats = {'strategy': 'l2_priority', 'sampled_count': len(sampled)}
            return sampled, stats
        
        elif self.sampling_strategy == 'category_l2_stratified':
            # Stratified sampling by l2-category within each category
            sampled, detailed_stats = stratified_sampling_by_l2_in_category(samples, shot)
            stats = {
                'strategy': 'category_l2_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'l2_category_stratified':
            # Stratified sampling by category within each l2-category
            sampled, detailed_stats = stratified_sampling_by_category_in_l2(samples, shot)
            stats = {
                'strategy': 'l2_category_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'category_random':
            # Random sampling by category
            sampled, sampling_stats = random_sampling_by_category(samples, shot)
            stats = {
                'strategy': 'category_random',
                'sampled_count': len(sampled),
                'sampling_stats': sampling_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'category_skill_stratified':
            # New: stratified sampling by skill within each category
            sampled, detailed_stats = stratified_sampling_by_skill_in_category(samples, shot)
            stats = {
                'strategy': 'category_skill_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'skill_stratified':
            # New: sampling by skill category
            sampled, sampling_stats = stratified_sampling_by_skill(samples, shot)
            stats = {
                'strategy': 'skill_stratified',
                'sampled_count': len(sampled),
                'sampling_stats': sampling_stats
            }
            return sampled, stats
        
        else:
            # Unknown strategy, fallback to random sampling
            print(f"⚠️ Unknown sampling strategy '{self.sampling_strategy}', fallback to random sampling")
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)
            sampled = shuffled_samples[:shot]
            stats = {'strategy': 'random_fallback', 'sampled_count': len(sampled)}
            return sampled, stats