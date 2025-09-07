#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

针对VLM模型进行分层探测和评估的自动化脚本。
该脚本通过两阶段评估找到模型的关键层，并测试中性化操作的效果。

阶段一：探测目标层 (Probing Phase)
- 使用前shot个样本进行N+1次推理（base + 每层cut_layer）
- 找到正向增益最大的层作为target_layer

阶段二：最终性能评估 (Final Evaluation Phase)  
- 对target_layer进行中性化，在剩余样本上评估最终性能

"""

import argparse
import json
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import traceback
from datetime import datetime
import logging

# 添加VLMEvalKit路径
VLMEVALKIT_DIR = "dir2vlmevalkit"  # 请根据实际路径修改
sys.path.append(VLMEVALKIT_DIR)

# 导入VLMEvalKit相关模块
try:
    from vlmeval.dataset import build_dataset
    from vlmeval.config import supported_VLM
    from vlmeval.utils import track_progress_rich
    from vlmeval.smp import *
    import pandas as pd
    from vlmeval.smp.misc import get_rank_and_world_size
    
    # 定义辅助函数来获取rank和world_size
    def get_rank():
        rank, _ = get_rank_and_world_size()
        return rank
    
    def get_world_size():
        _, world_size = get_rank_and_world_size()
        return world_size
    print("✅ VLMEvalKit模块导入成功")
except ImportError as e:
    print(f"❌ VLMEvalKit模块导入失败: {e}")
    sys.exit(1)

def stratified_sampling_by_skill_in_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    以category为类别，按照skill分层采样
    即：每个category内部按照skill的比例进行分层采样
    
    Args:
        samples: 样本列表
        shot: 总采样数量
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (采样后的样本, 各类别的分层采样统计)
    """
    from collections import defaultdict
    
    # 按category分组
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 以category为类别，按skill分层采样:")
    print(f"  - 发现 {len(category_to_samples)} 个category")
    
    # 为每个category分配采样数量
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # 最后一个category获得剩余的shot
            category_shots[category] = remaining_shot
        else:
            # 按比例分配
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # 不超过该category的样本数
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # 对每个category内部按skill分层采样
    final_samples = []
    detailed_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            detailed_stats[category] = {}
            continue
            
        print(f"  - {category}: 分配 {allocated_shot} 个样本")
        
        # 统计该category内的skill分布
        skill_to_samples = defaultdict(list)
        for sample in cat_samples:
            # 从raw_sample中获取skill字段
            skill = sample.get('raw_sample', {}).get('skills', 'unknown')
            if pd.isna(skill) or skill == '':
                skill = 'unknown'
            skill_to_samples[skill].append(sample)
        
        print(f"    包含 {len(skill_to_samples)} 个skill")
        
        # 为每个skill分配采样数量
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
        
        # 在category内按skill采样
        category_sampled = []
        skill_stats = {}
        
        for skill, skill_shot in skill_shots.items():
            if skill_shot > 0:
                skill_samples = skill_to_samples[skill]
                import random
                random.shuffle(skill_samples)
                selected = skill_samples[:skill_shot]
                category_sampled.extend(selected)
                skill_stats[skill] = len(selected)
                print(f"    - {skill}: {len(selected)} 个样本")
        
        final_samples.extend(category_sampled)
        detailed_stats[category] = skill_stats
    
    print(f"  - 总计采样: {len(final_samples)} 个样本")
    return final_samples, detailed_stats


def stratified_sampling_by_skill(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, int]]:
    """
    以skill为类别进行采样
    即：按照skill的比例分配采样数量，然后在每个skill内随机采样
    
    Args:
        samples: 样本列表
        shot: 总采样数量
        
    Returns:
        Tuple[List[Dict], Dict[str, int]]: (采样后的样本, 各skill采样数量统计)
    """
    from collections import defaultdict
    
    # 按skill分组
    skill_to_samples = defaultdict(list)
    for sample in samples:
        # 从raw_sample中获取skill字段
        skill = sample.get('raw_sample', {}).get('skills', 'unknown')
        if pd.isna(skill) or skill == '':
            skill = 'unknown'
        skill_to_samples[skill].append(sample)
    
    print(f"📊 以skill为类别采样:")
    print(f"  - 发现 {len(skill_to_samples)} 个skill")
    
    # 统计各skill的分布
    for skill, skill_samples in skill_to_samples.items():
        proportion = len(skill_samples) / len(samples)
        print(f"    - {skill}: {len(skill_samples)} 个样本 ({proportion:.1%})")
    
    # 为每个skill分配采样数量
    total_samples = len(samples)
    skill_shots = {}
    remaining_shot = shot
    
    for i, (skill, skill_samples) in enumerate(skill_to_samples.items()):
        if i == len(skill_to_samples) - 1:
            # 最后一个skill获得剩余的shot
            skill_shots[skill] = remaining_shot
        else:
            # 按比例分配
            proportion = len(skill_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(skill_samples))  # 不超过该skill的样本数
            skill_shots[skill] = allocated_shot
            remaining_shot -= allocated_shot
    
    # 对每个skill内部随机采样
    final_samples = []
    sampling_stats = {}
    
    for skill, allocated_shot in skill_shots.items():
        skill_samples = skill_to_samples[skill]
        
        if allocated_shot == 0:
            sampling_stats[skill] = 0
            continue
            
        print(f"  - {skill}: 分配 {allocated_shot} 个样本")
        
        # 随机采样
        import random
        random.shuffle(skill_samples)
        selected = skill_samples[:allocated_shot]
        final_samples.extend(selected)
        sampling_stats[skill] = len(selected)
    
    print(f"  - 总计采样: {len(final_samples)} 个样本")
    return final_samples, sampling_stats

def stratified_sampling_by_l2_in_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    以category为类别，按照l2-category分层采样
    即：每个category内部按照l2-category的比例进行分层采样
    
    Args:
        samples: 样本列表
        shot: 总采样数量
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (采样后的样本, 各类别的分层采样统计)
    """
    from collections import defaultdict
    
    # 按category分组
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 以category为类别，按l2-category分层采样:")
    print(f"  - 发现 {len(category_to_samples)} 个category")
    
    # 为每个category分配采样数量
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # 最后一个category获得剩余的shot
            category_shots[category] = remaining_shot
        else:
            # 按比例分配
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # 不超过该category的样本数
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # 对每个category内部按l2-category分层采样
    final_samples = []
    detailed_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            detailed_stats[category] = {}
            continue
            
        print(f"  - {category}: 分配 {allocated_shot} 个样本")
        
        # 统计该category内的l2-category分布
        l2cat_to_samples = defaultdict(list)
        for sample in cat_samples:
            # 处理字段名差异
            l2cat = sample.get('l2-category', sample.get('l2_category', 'unknown'))
            l2cat_to_samples[l2cat].append(sample)
        
        print(f"    包含 {len(l2cat_to_samples)} 个l2-category")
        
        # 为每个l2-category分配采样数量
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
        
        # 在category内按l2-category采样
        category_sampled = []
        l2_stats = {}
        
        for l2cat, l2_shot in l2cat_shots.items():
            if l2_shot > 0:
                l2_samples = l2cat_to_samples[l2cat]
                import random
                random.shuffle(l2_samples)
                selected = l2_samples[:l2_shot]
                category_sampled.extend(selected)
                l2_stats[l2cat] = len(selected)
                print(f"    - {l2cat}: {len(selected)} 个样本")
        
        final_samples.extend(category_sampled)
        detailed_stats[category] = l2_stats
    
    print(f"  - 总计采样: {len(final_samples)} 个样本")
    return final_samples, detailed_stats


def stratified_sampling_by_category_in_l2(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    """
    以l2-category为类别，按照category分层采样
    即：每个l2-category内部按照category的比例进行分层采样
    
    Args:
        samples: 样本列表
        shot: 总采样数量
        
    Returns:
        Tuple[List[Dict], Dict[str, Dict[str, int]]]: (采样后的样本, 各类别的分层采样统计)
    """
    from collections import defaultdict
    
    # 按l2-category分组，处理字段名差异
    l2cat_to_samples = defaultdict(list)
    for sample in samples:
        l2cat = sample.get('l2-category', sample.get('l2_category', 'unknown'))
        l2cat_to_samples[l2cat].append(sample)
    
    print(f"📊 以l2-category为类别，按category分层采样:")
    print(f"  - 发现 {len(l2cat_to_samples)} 个l2-category")
    
    # 为每个l2-category分配采样数量
    total_samples = len(samples)
    l2cat_shots = {}
    remaining_shot = shot
    
    for i, (l2cat, l2_samples) in enumerate(l2cat_to_samples.items()):
        if i == len(l2cat_to_samples) - 1:
            # 最后一个l2-category获得剩余的shot
            l2cat_shots[l2cat] = remaining_shot
        else:
            # 按比例分配
            proportion = len(l2_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(l2_samples))  # 不超过该l2-category的样本数
            l2cat_shots[l2cat] = allocated_shot
            remaining_shot -= allocated_shot
    
    # 对每个l2-category内部按category分层采样
    final_samples = []
    detailed_stats = {}
    
    for l2cat, allocated_shot in l2cat_shots.items():
        l2_samples = l2cat_to_samples[l2cat]
        
        if allocated_shot == 0:
            detailed_stats[l2cat] = {}
            continue
            
        print(f"  - {l2cat}: 分配 {allocated_shot} 个样本")
        
        # 统计该l2-category内的category分布
        cat_to_samples = defaultdict(list)
        for sample in l2_samples:
            category = sample.get('category', 'unknown')
            cat_to_samples[category].append(sample)
        
        print(f"    包含 {len(cat_to_samples)} 个category")
        
        # 为每个category分配采样数量
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
        
        # 在l2-category内按category采样
        l2cat_sampled = []
        cat_stats = {}
        
        for category, cat_shot in cat_shots.items():
            if cat_shot > 0:
                cat_samples = cat_to_samples[category]
                import random
                random.shuffle(cat_samples)
                selected = cat_samples[:cat_shot]
                l2cat_sampled.extend(selected)
                cat_stats[category] = len(selected)
                print(f"    - {category}: {len(selected)} 个样本")
        
        final_samples.extend(l2cat_sampled)
        detailed_stats[l2cat] = cat_stats
    
    print(f"  - 总计采样: {len(final_samples)} 个样本")
    return final_samples, detailed_stats


def random_sampling_by_category(samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict[str, int]]:
    """
    以category为类别，随机采样
    即：每个category按比例分配采样数量，然后在category内随机采样
    
    Args:
        samples: 样本列表
        shot: 总采样数量
        
    Returns:
        Tuple[List[Dict], Dict[str, int]]: (采样后的样本, 各类别采样数量统计)
    """
    from collections import defaultdict
    
    # 按category分组
    category_to_samples = defaultdict(list)
    for sample in samples:
        category = sample.get('category', 'unknown')
        category_to_samples[category].append(sample)
    
    print(f"📊 以category为类别，随机采样:")
    print(f"  - 发现 {len(category_to_samples)} 个category")
    
    # 为每个category分配采样数量
    total_samples = len(samples)
    category_shots = {}
    remaining_shot = shot
    
    for i, (category, cat_samples) in enumerate(category_to_samples.items()):
        if i == len(category_to_samples) - 1:
            # 最后一个category获得剩余的shot
            category_shots[category] = remaining_shot
        else:
            # 按比例分配
            proportion = len(cat_samples) / total_samples
            allocated_shot = int(round(proportion * shot))
            allocated_shot = min(allocated_shot, len(cat_samples))  # 不超过该category的样本数
            category_shots[category] = allocated_shot
            remaining_shot -= allocated_shot
    
    # 对每个category内部随机采样
    final_samples = []
    sampling_stats = {}
    
    for category, allocated_shot in category_shots.items():
        cat_samples = category_to_samples[category]
        
        if allocated_shot == 0:
            sampling_stats[category] = 0
            continue
            
        print(f"  - {category}: 分配 {allocated_shot} 个样本")
        
        # 随机采样
        import random
        random.shuffle(cat_samples)
        selected = cat_samples[:allocated_shot]
        final_samples.extend(selected)
        sampling_stats[category] = len(selected)
        print(f"    - 实际采样: {len(selected)} 个样本")
    
    print(f"  - 总计采样: {len(final_samples)} 个样本")
    return final_samples, sampling_stats


class SubtaskExtractor:
    def __init__(self, dataset_name: str, sampling_strategy: str = 'l2_priority'):
        """
        初始化子任务提取器
        
        Args:
            dataset_name: 数据集名称
            sampling_strategy: 采样策略
                - 'l2_priority': 优先使用l2-category，回退到category（原始逻辑）
                - 'category_l2_stratified': 以category为类别，按l2-category分层采样
                - 'l2_category_stratified': 以l2-category为类别，按category分层采样
                - 'category_random': 以category为类别，随机采样
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.index_mapping = {}  # 原始索引 -> 连续索引
        self.reverse_mapping = {}  # 连续索引 -> 原始索引
        self.primary_category_field = None  # 用于记录主要分类字段
        self.sampling_strategy = sampling_strategy
        
    def load_dataset(self):
        """加载数据集并创建索引映射"""
        try:
            self.dataset = build_dataset(self.dataset_name)
            if self.dataset is None:
                raise ValueError(f"数据集 {self.dataset_name} 构建失败")
            
            # 🔑 检查数据集的分类字段情况
            self._analyze_category_fields()
            
            # 创建索引映射
            original_indices = sorted(self.dataset.data['index'].unique())
            self.index_mapping = {orig: new for new, orig in enumerate(original_indices)}
            self.reverse_mapping = {new: orig for orig, new in self.index_mapping.items()}
            
            print(f"🔄 创建索引映射:")
            print(f"  - 原始索引范围: {min(original_indices)} - {max(original_indices)}")
            print(f"  - 映射后索引范围: 0 - {len(original_indices) - 1}")
            print(f"  - 映射示例: {list(self.index_mapping.items())[:5]}")
            
            print(f"✅ 数据集 {self.dataset_name} 加载成功")
            print(f"📋 采样策略: {self.sampling_strategy}")
            return True
        except Exception as e:
            print(f"❌ 数据集 {self.dataset_name} 加载失败: {e}")
            return False
    
    def _analyze_category_fields(self):
        """分析数据集的分类字段情况，决定使用哪个字段作为主要分类"""
        data_columns = self.dataset.data.columns.tolist()
        
        # 检查两种可能的字段名
        has_l2_category = 'l2-category' in data_columns or 'l2_category' in data_columns
        has_category = 'category' in data_columns
        
        # 检查skill字段
        has_skill = False
        if len(self.dataset.data) > 0:
            try:
                # 检查前几个样本的raw_sample中是否包含skill字段
                for i in range(min(5, len(self.dataset.data))):
                    sample_data = self.dataset.data.iloc[i]
                    # 如果是字典格式
                    if isinstance(sample_data.get('raw_sample'), dict):
                        if 'skills' in sample_data['raw_sample']:
                            has_skill = True
                            break
                    # 如果raw_sample是字符串形式的JSON，尝试解析
                    elif isinstance(sample_data.get('raw_sample'), str):
                        try:
                            import json
                            raw_data = json.loads(sample_data['raw_sample'])
                            if 'skills' in raw_data:
                                has_skill = True
                                break
                        except:
                            continue
                    # 直接检查是否有skill列
                    elif 'skills' in sample_data:
                        has_skill = True
                        break
            except Exception as e:
                print(f"⚠️ 检测skill字段时出错: {e}")
                has_skill = False
        
        print(f"📊 分析数据集分类字段:")
        print(f"  - 包含 'l2-category' 或 'l2_category': {has_l2_category}")
        print(f"  - 包含 'category': {has_category}")
        print(f"  - raw_sample中包含 'skills': {has_skill}")
        
        # 根据采样策略确定主要分类字段
        if self.sampling_strategy == 'l2_priority':
            # 原始逻辑：优先l2-category
            if has_l2_category:
                # 确定实际使用的字段名
                l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
                l2_cat_values = self.dataset.data[l2_field_name].dropna().unique()
                if len(l2_cat_values) > 0 and not all(v == 'unknown' for v in l2_cat_values):
                    self.primary_category_field = l2_field_name
                    print(f"  ✅ 使用 '{l2_field_name}' 作为主要分类字段")
                    print(f"  - l2-category 唯一值数: {len(l2_cat_values)}")
                    print(f"  - l2-category 示例: {list(l2_cat_values)[:5]}")
                elif has_category:
                    self.primary_category_field = 'category'
                    print(f"  ⚠️ l2-category 无有效值，回退到 'category'")
                else:
                    self.primary_category_field = None
                    print(f"  ❌ 无有效的分类字段")
            elif has_category:
                self.primary_category_field = 'category'
                print(f"  ✅ 使用 'category' 作为主要分类字段")
                cat_values = self.dataset.data['category'].dropna().unique()
                print(f"  - category 唯一值数: {len(cat_values)}")
                print(f"  - category 示例: {list(cat_values)[:5]}")
            else:
                self.primary_category_field = None
                print(f"  ❌ 数据集不包含 'category' 或 'l2-category' 字段")
        
        elif self.sampling_strategy in ['category_l2_stratified', 'category_random', 'category_skill_stratified']:
            # 以category为主的策略
            if has_category:
                self.primary_category_field = 'category'
                print(f"  ✅ 使用 'category' 作为主要分类字段 (策略: {self.sampling_strategy})")
                cat_values = self.dataset.data['category'].dropna().unique()
                print(f"  - category 唯一值数: {len(cat_values)}")
                print(f"  - category 示例: {list(cat_values)[:5]}")
                
                # 对于skill相关策略，额外检查skill字段
                if self.sampling_strategy == 'category_skill_stratified':
                    if has_skill:
                        print(f"  ✅ raw_sample中包含skill字段，支持skill分层采样")
                        # 获取skill值示例
                        skill_examples = []
                        skill_counts = {}
                        for i in range(min(50, len(self.dataset.data))):  # 检查更多样本以获得完整的skill分布
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
                        
                        print(f"  - skill 类型数: {len(skill_examples)}")
                        print(f"  - skill 示例: {skill_examples[:10]}")  # 显示前10个
                        if skill_counts:
                            print(f"  - skill 分布示例: {dict(list(skill_counts.items())[:5])}")
                    else:
                        print(f"  ⚠️ 策略 '{self.sampling_strategy}' 需要 'skill' 字段，但raw_sample中不存在")
            else:
                self.primary_category_field = None
                print(f"  ❌ 策略 '{self.sampling_strategy}' 需要 'category' 字段，但数据集中不存在")
        
        elif self.sampling_strategy == 'l2_category_stratified':
            # 以l2-category为主的策略
            if has_l2_category:
                # 确定实际使用的字段名
                l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
                l2_cat_values = self.dataset.data[l2_field_name].dropna().unique()
                if len(l2_cat_values) > 0 and not all(v == 'unknown' for v in l2_cat_values):
                    self.primary_category_field = l2_field_name
                    print(f"  ✅ 使用 '{l2_field_name}' 作为主要分类字段 (策略: {self.sampling_strategy})")
                    print(f"  - l2-category 唯一值数: {len(l2_cat_values)}")
                    print(f"  - l2-category 示例: {list(l2_cat_values)[:5]}")
                else:
                    self.primary_category_field = None
                    print(f"  ❌ 策略 '{self.sampling_strategy}' 需要有效的 'l2-category' 字段，但数据集中无有效值")
            else:
                self.primary_category_field = None
                print(f"  ❌ 策略 '{self.sampling_strategy}' 需要 'l2-category' 字段，但数据集中不存在")
        
        elif self.sampling_strategy == 'skill_stratified':
            # 以skill为主的策略
            if has_skill:
                self.primary_category_field = 'skills'
                print(f"  ✅ 使用 'skill' 作为主要分类字段 (策略: {self.sampling_strategy})")
                # 获取skill值示例
                skill_examples = []
                skill_counts = {}
                for i in range(min(50, len(self.dataset.data))):  # 检查更多样本以获得完整的skill分布
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
                
                print(f"  - skill 类型数: {len(skill_examples)}")
                print(f"  - skill 示例: {skill_examples[:10]}")  # 显示前10个
                if skill_counts:
                    print(f"  - skill 分布示例: {dict(list(skill_counts.items())[:5])}")
            else:
                self.primary_category_field = None
                print(f"  ❌ 策略 '{self.sampling_strategy}' 需要 'skill' 字段，但raw_sample中不存在")

    def extract_subtasks(self) -> Dict[str, List[Dict]]:
        """提取子任务，使用映射后的连续索引，根据采样策略决定分类方式"""
        if self.dataset is None:
            print("❌ 数据集未加载，无法提取子任务")
            return {}
        
        if self.primary_category_field is None:
            print("❌ 无有效的分类字段，无法提取子任务")
            return {}
        
        # 确定l2-category字段的实际名称
        data_columns = self.dataset.data.columns.tolist()
        l2_field_name = 'l2-category' if 'l2-category' in data_columns else 'l2_category'
        l2_field_exists = l2_field_name in data_columns
        
        subtasks = {}
        total_samples = len(self.dataset.data)
        print(f"📊 使用 '{self.primary_category_field}' 字段处理 {total_samples} 个样本...")
        print(f"📋 采样策略: {self.sampling_strategy}")
        
        for idx in range(total_samples):
            try:
                sample_dict = self.dataset.data.iloc[idx].to_dict()
                
                # 获取原始索引并映射为连续索引
                original_index = sample_dict.get('index')
                mapped_index = self.index_mapping.get(original_index, idx)
                
                # 🔑 根据主要分类字段提取任务类别
                if self.primary_category_field == 'skills':
                    # 🔑 从raw_sample中获取skill字段
                    if isinstance(sample_dict.get('raw_sample'), dict):
                        subtask_name = sample_dict['raw_sample'].get('skills', 'unknown')
                    else:
                        subtask_name = 'unknown'
                else:
                    # 使用其他分类字段
                    subtask_name = sample_dict.get(self.primary_category_field, 'unknown')
                
                if subtask_name == 'unknown' or pd.isna(subtask_name):
                    # 如果主要字段无效，尝试备用字段
                    if self.primary_category_field == 'skills':
                        # skill策略失败时，尝试category字段
                        backup_field = 'category'
                    elif self.primary_category_field in ['l2-category', 'l2_category']:
                        backup_field = 'category'
                    else:
                        backup_field = l2_field_name  # 使用实际的l2字段名
                    
                    if backup_field in sample_dict:
                        subtask_name = sample_dict.get(backup_field, 'unknown')
                    
                    if subtask_name == 'unknown' or pd.isna(subtask_name):
                        continue
                
                if subtask_name not in subtasks:
                    subtasks[subtask_name] = []
                
                # 创建样本数据，使用映射后的连续索引
                sample_data = {
                    'index': mapped_index,  # 🔑 使用映射后的连续索引
                    'original_index': original_index,  # 保留原始索引用于追溯
                    'category': sample_dict.get('category', 'unknown'),
                    'l2-category': sample_dict.get(l2_field_name, 'unknown') if l2_field_exists else 'unknown',
                    'primary_category': subtask_name,  # 记录用于分类的主要类别
                    'primary_category_field': self.primary_category_field,  # 记录使用的字段
                    'sampling_strategy': self.sampling_strategy,  # 记录采样策略
                    'question': sample_dict.get('question', ''),
                    'choices': sample_dict.get('choices', ''),
                    'answer': sample_dict.get('answer', ''),
                    'image': sample_dict.get('image', ''),
                    'raw_sample': sample_dict.copy()
                }
                
                # 🔑 更新raw_sample中的索引为映射后的索引
                sample_data['raw_sample']['index'] = mapped_index
                
                subtasks[subtask_name].append(sample_data)
                
            except Exception as e:
                print(f"⚠️ 处理样本 {idx} 时出错: {e}")
                continue
        
        # 统计结果
        print(f"📊 子任务提取完成:")
        print(f"  - 使用分类字段: {self.primary_category_field}")
        print(f"  - 采样策略: {self.sampling_strategy}")
        print(f"  - 提取到 {len(subtasks)} 个子任务")
        for task_name, samples in subtasks.items():
            print(f"  - {task_name}: {len(samples)} 个样本")
        
        return subtasks
    
    def apply_sampling_strategy(self, samples: List[Dict], shot: int) -> Tuple[List[Dict], Dict]:
        """
        根据采样策略对样本进行采样
        
        Args:
            samples: 样本列表
            shot: 采样数量
            
        Returns:
            Tuple[List[Dict], Dict]: (采样后的样本, 采样统计信息)
        """
        if self.sampling_strategy == 'l2_priority':
            # 原始逻辑：简单随机采样
            import random
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)
            sampled = shuffled_samples[:shot]
            stats = {'strategy': 'l2_priority', 'sampled_count': len(sampled)}
            return sampled, stats
        
        elif self.sampling_strategy == 'category_l2_stratified':
            # 以category为类别，按l2-category分层采样
            sampled, detailed_stats = stratified_sampling_by_l2_in_category(samples, shot)
            stats = {
                'strategy': 'category_l2_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'l2_category_stratified':
            # 以l2-category为类别，按category分层采样
            sampled, detailed_stats = stratified_sampling_by_category_in_l2(samples, shot)
            stats = {
                'strategy': 'l2_category_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'category_random':
            # 以category为类别，随机采样
            sampled, sampling_stats = random_sampling_by_category(samples, shot)
            stats = {
                'strategy': 'category_random',
                'sampled_count': len(sampled),
                'sampling_stats': sampling_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'category_skill_stratified':
            # 🔑 新增：以category为类别，按skill分层采样
            sampled, detailed_stats = stratified_sampling_by_skill_in_category(samples, shot)
            stats = {
                'strategy': 'category_skill_stratified',
                'sampled_count': len(sampled),
                'detailed_stats': detailed_stats
            }
            return sampled, stats
        
        elif self.sampling_strategy == 'skill_stratified':
            # 🔑 新增：以skill为类别采样
            sampled, sampling_stats = stratified_sampling_by_skill(samples, shot)
            stats = {
                'strategy': 'skill_stratified',
                'sampled_count': len(sampled),
                'sampling_stats': sampling_stats
            }
            return sampled, stats
        
        else:
            # 未知策略，回退到随机采样
            print(f"⚠️ 未知采样策略 '{self.sampling_strategy}'，回退到随机采样")
            import random
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)
            sampled = shuffled_samples[:shot]
            stats = {'strategy': 'random_fallback', 'sampled_count': len(sampled)}
            return sampled, stats

class ModelWrapper:
    """
    模型包装器，提供统一的推理接口 - 精简版，不手动加载模型
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # 🔑 不再手动加载和保存模型实例
        
    def load_model(self):
        """验证模型名称是否在支持列表中"""
        try:
            if self.model_name not in supported_VLM:
                available_models = list(supported_VLM.keys())
                print(f"❌ 模型 {self.model_name} 不在支持列表中")
                print(f"支持的模型: {available_models[:10]}{'...' if len(available_models) > 10 else ''}")
                return False
            
            print(f"✅ 模型 {self.model_name} 在支持列表中，将由VLMEvalKit按需加载")
            return True
        except Exception as e:
            print(f"❌ 验证模型 {self.model_name} 时出错: {e}")
            return False
    
    def get_num_layers(self) -> int:
        """获取模型的层数 - 使用预设值"""
        # 🔑 由于不手动加载模型，使用预设的层数映射
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
            # 添加更多模型的层数映射
        }
        
        num_layers = layer_mapping.get(self.model_name, 32)  # 默认32层
        print(f"✅ 使用预设层数: {self.model_name} -> {num_layers} 层")
        return num_layers

class EvaluationEngine:
    """
    评估引擎，完全使用VLMEvalKit的机制
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None
    
    def load_dataset(self):
        """加载数据集用于评估"""
        try:
            self.dataset = build_dataset(self.dataset_name)
            return self.dataset is not None
        except Exception as e:
            print(f"❌ 评估引擎加载数据集失败: {e}")
            return False
    
    def evaluate_samples_with_cut_layer(self, model_wrapper: ModelWrapper, samples: List[Dict], 
                                       cut_layer: int = -1, cut_module: str = "self_attn",
                                       work_dir: str = None) -> float:
        """
        使用VLMEvalKit内部机制评估带有cut_layer的样本 - 只加载一次模型
        """
        if len(samples) == 0:
            print("⚠️ 样本列表为空，返回0分")
            return 0.0
        
        try:
            # 创建临时工作目录
            if work_dir is None:
                work_dir = f"./temp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            os.makedirs(work_dir, exist_ok=True)
            
            print(f"🚀 使用VLMEvalKit单次加载推理 (cut_layer={cut_layer}, cut_module={cut_module})")
            
            # 🔑 关键：使用VLMEvalKit的内部机制，只加载一次
            result = self._run_vlmeval_inference_single_load(
                model_wrapper, samples, cut_layer, cut_module, work_dir
            )
            
            if not result:
                print("❌ VLMEvalKit推理失败")
                return 0.0
            
            # 执行评估
            score = self._run_evaluation_vlmeval_style(result, work_dir)
            
            print(f"✅ 评估完成，得分: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"❌ 评估过程中出错: {e}")
            traceback.print_exc()
            return 0.0

    def _run_vlmeval_inference_single_load(self, model_wrapper: ModelWrapper, 
                                          samples: List[Dict], cut_layer: int, 
                                          cut_module: str, work_dir: str) -> str:
        """
        使用VLMEvalKit内部机制进行推理 - 单次模型加载版本，支持大模型优化
        """
        try:
            print(f"🔧 准备VLMEvalKit推理数据 (样本数: {len(samples)})")
            
            # 构建DataFrame
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
            
            # 🔑 关键：直接调用infer_data_job，让它处理模型加载，添加大模型优化参数
            from vlmeval.inference import infer_data_job
            
            # 构建推理参数，添加大模型优化
            inference_kwargs = {
                'model': model_wrapper.model_name,  # 🔑 只传递模型名称，由VLMEvalKit加载
                'model_name': model_wrapper.model_name,
                'dataset': temp_dataset,
                'work_dir': work_dir,
                'verbose': False,
                'api_nproc': 1,  # 🔑 大模型使用单进程
                'ignore_failed': True,
            }
            
            # 🔑 对于大模型，添加特殊的环境变量控制
            original_env = {}
            try:
                # 设置大模型优化环境变量
                env_settings = {
                    'CUDA_LAUNCH_BLOCKING': '1',
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                    'TOKENIZERS_PARALLELISM': 'false'
                }
                
                for key, value in env_settings.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                # 🔑 如果支持cut_layer参数，则添加
                if cut_layer >= 0:
                    print(f"🔪 应用cut_layer: 层{cut_layer}, 模块{cut_module}")
                    inference_kwargs['cut_layer'] = cut_layer
                    inference_kwargs['cut_module'] = cut_module
                else:
                    print(f"📝 正常推理，无cut_layer")
                
                # 🔑 使用VLMEvalKit的标准方式调用推理（单次模型加载）
                print(f"🔄 开始VLMEvalKit推理，模型: {model_wrapper.model_name}")
                result = infer_data_job(**inference_kwargs)
                
            except TypeError as e:
                if 'cut_layer' in str(e):
                    print("⚠️ infer_data_job不支持cut_layer参数，移除该参数重试")
                    # 移除cut_layer相关参数重试
                    inference_kwargs.pop('cut_layer', None)
                    inference_kwargs.pop('cut_module', None)
                    result = infer_data_job(**inference_kwargs)
                    
                    if cut_layer >= 0:
                        print(f"⚠️ 注意：cut_layer={cut_layer}参数未生效，使用正常推理")
                else:
                    raise e
            finally:
                # 恢复环境变量
                for key, old_value in original_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
            
            # 🔑 强制清理显存
            torch.cuda.empty_cache()
            
            # 查找结果文件
            xlsx_files = [f for f in os.listdir(work_dir) 
                         if f.endswith('.xlsx') and 'temp_input_data' not in f]
            
            if xlsx_files:
                result_file = os.path.join(work_dir, xlsx_files[0])
                print(f"✅ VLMEvalKit推理完成: {result_file}")
                return result_file
            else:
                raise Exception("未找到推理结果文件")
                
        except Exception as e:
            print(f"❌ VLMEvalKit推理失败: {e}")
            traceback.print_exc()
            # 🔑 出错时也清理显存
            torch.cuda.empty_cache()
            raise e

    def _create_temp_dataset(self, target_df: pd.DataFrame):
        """创建临时数据集"""
        import copy
        temp_dataset = copy.deepcopy(self.dataset)
        temp_dataset.data = target_df
        
        print(f"✅ 创建临时数据集，样本数: {len(target_df)}")
        return temp_dataset

    def _run_evaluation_vlmeval_style(self, inference_result_file: str, work_dir: str) -> float:
        """使用VLMEvalKit的dataset.evaluate方法执行评估"""
        try:
            print(f"📊 使用VLMEvalKit标准评估流程...")
            
            if not os.path.exists(inference_result_file):
                print(f"❌ 推理结果文件不存在: {inference_result_file}")
                return 0.0
            
            # 使用dataset的evaluate方法
            if hasattr(self.dataset, 'evaluate'):
                try:
                    judge_kwargs = self._get_judge_kwargs()
                    print(f"🔄 开始VLMEvalKit评估，参数: {judge_kwargs}")
                    
                    eval_results = self.dataset.evaluate(inference_result_file, **judge_kwargs)
                    
                    # 保存详细评估结果
                    eval_result_file = os.path.join(work_dir, "vlmeval_evaluation_results.json")
                    with open(eval_result_file, 'w', encoding='utf-8') as f:
                        if isinstance(eval_results, dict):
                            json.dump(eval_results, f, indent=2, ensure_ascii=False)
                        else:
                            json.dump(eval_results.to_dict(), f, indent=2, ensure_ascii=False)
                    
                    score = self._parse_eval_results(eval_results)
                    return score
                        
                except Exception as e:
                    print(f"⚠️ 使用dataset.evaluate失败: {e}")
                    return self._fallback_evaluation(inference_result_file)
            else:
                print("⚠️ 数据集不支持evaluate方法，使用简单评估")
                return self._fallback_evaluation(inference_result_file)
                    
        except Exception as e:
            print(f"❌ VLMEvalKit评估失败: {e}")
            return self._fallback_evaluation(inference_result_file)

    def _get_judge_kwargs(self) -> Dict:
        """获取评估参数"""
        judge_kwargs = {
            'nproc': 1,  # 🔑 大模型使用单进程
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
        """解析VLMEvalKit的评估结果"""
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
            print(f"⚠️ 解析评估结果失败: {e}")
            return 0.0

    def _fallback_evaluation(self, inference_result_file: str) -> float:
        """备用评估方法"""
        try:
            df = pd.read_excel(inference_result_file)
            if 'prediction' in df.columns and 'answer' in df.columns:
                correct = (df['prediction'].astype(str).str.upper() == df['answer'].astype(str).str.upper()).sum()
                total = len(df)
                return correct / total if total > 0 else 0.0
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️ 备用评估失败: {e}")
            return 0.0

def setup_logging(output_dir: str):
    """设置日志记录"""
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
    """为特定子任务设置日志记录"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{subtask_name}_evaluation.log")
    
    # 创建一个新的logger实例
    logger = logging.getLogger(f"subtask_{subtask_name}")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_stage_logging(work_dir: str, stage_name: str):
    """为特定阶段设置日志记录"""
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, f"{stage_name}_log.txt")
    
    # 创建一个新的logger实例
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def probe_target_layer_vlmeval_enhanced(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                                       probe_samples: List[Dict], work_dir: str, 
                                       all_samples: List[Dict] = None, shot: int = 5) -> Tuple[int, float, Dict]:
    """
    探测目标层 - VLMEvalKit风格但保持完整逻辑
    """
    print(f"🔍 开始VLMEvalKit风格探测目标层 (探测样本数: {len(probe_samples)})")
    
    probe_logger = setup_stage_logging(work_dir, "probe")
    probe_logger.info(f"开始VLMEvalKit风格探测目标层 (探测样本数: {len(probe_samples)})")
    
    # 获取模型层数
    num_layers = model_wrapper.get_num_layers()
    probe_logger.info(f"模型总层数: {num_layers}")
    print(f"📊 模型总层数: {num_layers}")
    
    # 🔑 保持原始的重新采样逻辑
    resample_round = 0
    max_resample_rounds = 5
    current_probe_samples = probe_samples.copy()
    used_sample_indices = set()
    
    # 如果提供了all_samples，记录当前使用的样本索引
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
            print(f"🔄 重新采样第 {resample_round} 轮")
            probe_logger.info(f"重新采样第 {resample_round} 轮")
        
        # 🔑 阶段一：基准评估（使用VLMEvalKit方式）
        print(f"📊 执行基准评估{round_suffix}...")
        probe_logger.info(f"执行基准评估{round_suffix}...")
        
        base_eval_dir = os.path.join(round_work_dir, f"base_eval{round_suffix}")
        base_score = evaluation_engine.evaluate_samples_with_cut_layer(
            model_wrapper, current_probe_samples, 
            cut_layer=-1,  # -1表示不切断任何层
            work_dir=base_eval_dir
        )
        print(f"✅ 基准得分{round_suffix}: {base_score:.4f}")
        probe_logger.info(f"基准得分{round_suffix}: {base_score:.4f}")
        
        # 🔑 第一优先级：检查基准得分是否为1.0（保持原始逻辑）
        if (resample_round == 0 and base_score >= 1.0 and all_samples and 
            resample_round < max_resample_rounds):
            print(f"⚠️ 基准得分为 {base_score:.4f}，尝试重新采样...")
            probe_logger.info(f"基准得分为 {base_score:.4f}，尝试重新采样...")
            
            # 重新采样：从未使用的样本中选择
            available_samples = [sample for sample in all_samples 
                               if sample.get('index', -1) not in used_sample_indices]
            
            if len(available_samples) >= shot:
                # 随机选择新的样本
                import random
                new_probe_samples = random.sample(available_samples, shot)
                
                # 更新使用的样本索引
                new_indices = {sample.get('index', -1) for sample in new_probe_samples}
                used_sample_indices.update(new_indices)
                
                current_probe_samples = new_probe_samples
                resample_round += 1
                
                print(f"🔄 基准得分为1，重新采样成功，新样本索引: {sorted(list(new_indices))}")
                probe_logger.info(f"基准得分为1，重新采样成功，新样本索引: {sorted(list(new_indices))}")
                continue
            else:
                print(f"⚠️ 可用样本不足 ({len(available_samples)} < {shot})，继续使用当前样本")
                probe_logger.warning(f"可用样本不足 ({len(available_samples)} < {shot})，继续使用当前样本")
        
        # 🔑 阶段二：逐层评估（使用VLMEvalKit方式）
        layer_scores = {}
        layer_improvements = {}
        
        final_round_suffix = f"_final_round_{resample_round}" if resample_round > 0 else ""
        print(f"🔄 开始逐层评估 (共 {num_layers} 层){final_round_suffix}...")
        probe_logger.info(f"开始逐层评估 (共 {num_layers} 层){final_round_suffix}...")
        
        for layer_idx in range(1, num_layers):
            try:
                layer_work_dir = os.path.join(round_work_dir, f"layer_{layer_idx}_eval{final_round_suffix}")
                layer_logger = setup_stage_logging(layer_work_dir, f"layer_{layer_idx}{final_round_suffix}")
                layer_logger.info(f"评估第 {layer_idx} 层{final_round_suffix}...")
                print(f"  📊 评估第 {layer_idx} 层{final_round_suffix}...")
                
                # 🔑 关键：使用VLMEvalKit的cut_layer机制
                layer_score = evaluation_engine.evaluate_samples_with_cut_layer(
                    model_wrapper, current_probe_samples,
                    cut_layer=layer_idx,
                    cut_module="self_attn",
                    work_dir=layer_work_dir
                )
                
                layer_scores[layer_idx] = layer_score
                improvement = layer_score - base_score
                layer_improvements[layer_idx] = improvement
                
                print(f"    层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
                layer_logger.info(f"层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
                probe_logger.info(f"层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
                
            except Exception as e:
                print(f"❌ 评估第 {layer_idx} 层时出错: {e}")
                probe_logger.error(f"评估第 {layer_idx} 层时出错: {e}")
                layer_scores[layer_idx] = 0.0
                layer_improvements[layer_idx] = -float('inf')
                continue
        
        # 🔑 第二优先级：寻找最佳改进（保持原始tie解决逻辑）
        positive_improvements = {layer: improvement for layer, improvement 
                               in layer_improvements.items() if improvement > 0}
        
        if positive_improvements:
            # 找到最大改进值
            best_improvement = max(positive_improvements.values())
            
            # 找到所有具有最大改进值的层
            best_layers = [layer for layer, improvement in positive_improvements.items() 
                        if abs(improvement - best_improvement) < 1e-6]
            
            if len(best_layers) == 1:
                # 单一目标层
                target_layer = best_layers[0]
                print(f"🎯 找到单一目标层: {target_layer} (改进: {best_improvement:+.4f})")
                probe_logger.info(f"找到单一目标层: {target_layer} (改进: {best_improvement:+.4f})")
                break  # 直接退出重采样循环
            else:
                # 🔑 多层tie的增强处理（保持原始逻辑）
                print(f"🔍 发现多个层({best_layers})具有相同最佳改进 {best_improvement:+.4f}")
                probe_logger.info(f"发现多个层({best_layers})具有相同最佳改进 {best_improvement:+.4f}")
                
                target_layer, tie_resolution_details = _resolve_multi_layer_tie_vlmeval(
                    model_wrapper, evaluation_engine, best_layers, best_improvement,
                    current_probe_samples, base_score, all_samples, used_sample_indices,
                    round_work_dir, probe_logger, shot
                )
                
                # 更新相关变量
                current_probe_samples = tie_resolution_details.get('final_probe_samples', current_probe_samples)
                base_score = tie_resolution_details.get('final_base_score', base_score)
                layer_scores.update(tie_resolution_details.get('additional_layer_scores', {}))
                layer_improvements.update(tie_resolution_details.get('additional_layer_improvements', {}))
                best_improvement = tie_resolution_details.get('final_best_improvement', best_improvement)
                used_sample_indices.update(tie_resolution_details.get('additional_used_indices', set()))
                
                break  # Tie解决完成，退出重采样循环
        else:
            # 🔑 第三优先级：没有正向改进的层，尝试重新采样（保持原始逻辑）
            best_improvement = max(layer_improvements.values()) if layer_improvements else -float('inf')
            print(f"⚠️ 没有层产生正向改进，最佳改进: {best_improvement:+.4f}")
            probe_logger.warning(f"没有层产生正向改进，最佳改进: {best_improvement:+.4f}")
            
            if all_samples and resample_round < max_resample_rounds:
                available_samples = [sample for sample in all_samples 
                                   if sample.get('index', -1) not in used_sample_indices]
                
                if len(available_samples) >= shot:
                    import random
                    new_probe_samples = random.sample(available_samples, shot + resample_round + 1)
                    
                    # 更新使用的样本索引
                    new_indices = {sample.get('index', -1) for sample in new_probe_samples}
                    used_sample_indices.update(new_indices)
                    
                    current_probe_samples = new_probe_samples
                    resample_round += 1
                    
                    print(f"🔄 未找到目标层，重新采样第 {resample_round} 轮，新样本索引: {sorted(list(new_indices))}")
                    probe_logger.info(f"未找到目标层，重新采样第 {resample_round} 轮，新样本索引: {sorted(list(new_indices))}")
                    continue
                else:
                    print(f"⚠️ 可用样本不足 ({len(available_samples)} < {shot})，停止重新采样")
                    probe_logger.warning(f"可用样本不足 ({len(available_samples)} < {shot})，停止重新采样")
                    target_layer = -1
                    break
            else:
                target_layer = -1
                break
    
    # 结果总结
    if target_layer == -1:
        print(f"⚠️ 经过 {resample_round} 轮采样后仍未找到有效目标层")
        probe_logger.warning(f"经过 {resample_round} 轮采样后仍未找到有效目标层")
    
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
    
    # 记录tie解决的详细信息（如果有的话）
    if target_layer >= 0 and 'tie_resolution_details' in locals():
        details['multi_tie_resolution'] = tie_resolution_details
    
    probe_logger.info("VLMEvalKit风格探测阶段完成")
    return target_layer, base_score, details

def _resolve_multi_layer_tie_vlmeval(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                                    best_layers: List[int], best_improvement: float,
                                    current_probe_samples: List[Dict], base_score: float,
                                    all_samples: List[Dict], used_sample_indices: set,
                                    work_dir: str, logger, shot: int) -> Tuple[int, Dict]:
    """
    解决多层tie的问题 - VLMEvalKit版本
    """
    print(f"🔧 开始VLMEvalKit风格多层tie解决过程...")
    logger.info(f"开始VLMEvalKit风格多层tie解决过程 - tied层: {best_layers}")
    
    tie_resolution_details = {
        'original_tied_layers': best_layers.copy(),
        'original_best_improvement': best_improvement,
        'resolution_rounds': [],
        'additional_used_indices': set(),
        'final_target_layer': -1,
        'final_probe_samples': current_probe_samples,
        'final_base_score': base_score,
        'additional_layer_scores': {},
        'additional_layer_improvements': {},
        'final_best_improvement': best_improvement
    }
    
    if not all_samples:
        # 没有额外样本，直接选择最后一层
        final_target_layer = max(best_layers)
        print(f"⚠️ 没有额外样本，选择最后一层: {final_target_layer}")
        logger.info(f"没有额外样本，选择最后一层: {final_target_layer}")
        tie_resolution_details['final_target_layer'] = final_target_layer
        return final_target_layer, tie_resolution_details
    
    # 第一次增样：增加 shot/2 个样本
    available_samples = [sample for sample in all_samples 
                        if sample.get('index', -1) not in used_sample_indices]
    
    additional_samples_needed_1 = max(1, shot // 2)
    if len(available_samples) >= additional_samples_needed_1:
        print(f"📈 第一轮VLMEvalKit风格tie解决：增加 {additional_samples_needed_1} 个样本...")
        logger.info(f"第一轮VLMEvalKit风格tie解决：增加 {additional_samples_needed_1} 个样本...")
        
        import random
        additional_samples_1 = random.sample(available_samples, additional_samples_needed_1)
        expanded_samples_1 = current_probe_samples + additional_samples_1
        
        additional_indices_1 = {sample.get('index', -1) for sample in additional_samples_1}
        used_sample_indices.update(additional_indices_1)
        tie_resolution_details['additional_used_indices'].update(additional_indices_1)
        
        # 第一轮解决（使用VLMEvalKit方式）
        round_1_result = _evaluate_tie_resolution_round_vlmeval(
            model_wrapper, evaluation_engine, best_layers, expanded_samples_1,
            work_dir, logger, 1, "增加shot/2样本"
        )
        
        tie_resolution_details['resolution_rounds'].append(round_1_result)
        tie_resolution_details['additional_layer_scores'].update(round_1_result['layer_scores'])
        tie_resolution_details['additional_layer_improvements'].update(round_1_result['layer_improvements'])
        
        if len(round_1_result['final_best_layers']) == 1:
            # 第一轮就解决了tie
            final_target_layer = round_1_result['final_best_layers'][0]
            print(f"🎯 第一轮VLMEvalKit风格tie解决成功！目标层: {final_target_layer}")
            logger.info(f"第一轮VLMEvalKit风格tie解决成功！目标层: {final_target_layer}")
            
            tie_resolution_details['final_target_layer'] = final_target_layer
            tie_resolution_details['final_probe_samples'] = expanded_samples_1
            tie_resolution_details['final_base_score'] = round_1_result['base_score']
            tie_resolution_details['final_best_improvement'] = round_1_result['best_improvement']
            
            return final_target_layer, tie_resolution_details
        else:
            # 仍然tie，准备第二轮
            remaining_tied_layers = round_1_result['final_best_layers']
            print(f"⚠️ 第一轮后仍有tie: {remaining_tied_layers}")
            logger.info(f"第一轮后仍有tie: {remaining_tied_layers}")
            
            # 第二次增样：再增加 shot/4 个样本
            available_samples = [sample for sample in all_samples 
                               if sample.get('index', -1) not in used_sample_indices]
            
            additional_samples_needed_2 = max(1, shot // 4)
            if len(available_samples) >= additional_samples_needed_2:
                print(f"📈 第二轮VLMEvalKit风格tie解决：再增加 {additional_samples_needed_2} 个样本...")
                logger.info(f"第二轮VLMEvalKit风格tie解决：再增加 {additional_samples_needed_2} 个样本...")
                
                additional_samples_2 = random.sample(available_samples, additional_samples_needed_2)
                expanded_samples_2 = expanded_samples_1 + additional_samples_2
                
                additional_indices_2 = {sample.get('index', -1) for sample in additional_samples_2}
                used_sample_indices.update(additional_indices_2)
                tie_resolution_details['additional_used_indices'].update(additional_indices_2)
                
                # 第二轮解决（使用VLMEvalKit方式）
                round_2_result = _evaluate_tie_resolution_round_vlmeval(
                    model_wrapper, evaluation_engine, remaining_tied_layers, expanded_samples_2,
                    work_dir, logger, 2, "再增加shot/4样本"
                )
                
                tie_resolution_details['resolution_rounds'].append(round_2_result)
                tie_resolution_details['additional_layer_scores'].update(round_2_result['layer_scores'])
                tie_resolution_details['additional_layer_improvements'].update(round_2_result['layer_improvements'])
                
                if len(round_2_result['final_best_layers']) == 1:
                    # 第二轮解决了tie
                    final_target_layer = round_2_result['final_best_layers'][0]
                    print(f"🎯 第二轮VLMEvalKit风格tie解决成功！目标层: {final_target_layer}")
                    logger.info(f"第二轮VLMEvalKit风格tie解决成功！目标层: {final_target_layer}")
                    
                    tie_resolution_details['final_target_layer'] = final_target_layer
                    tie_resolution_details['final_probe_samples'] = expanded_samples_2
                    tie_resolution_details['final_base_score'] = round_2_result['base_score']
                    tie_resolution_details['final_best_improvement'] = round_2_result['best_improvement']
                    
                    return final_target_layer, tie_resolution_details
                else:
                    # 仍然tie，选择最后一层
                    final_tied_layers = round_2_result['final_best_layers']
                    final_target_layer = max(final_tied_layers)
                    print(f"⚠️ 第二轮后仍有tie: {final_tied_layers}，选择最后一层: {final_target_layer}")
                    logger.info(f"第二轮后仍有tie: {final_tied_layers}，选择最后一层: {final_target_layer}")
                    
                    tie_resolution_details['final_target_layer'] = final_target_layer
                    tie_resolution_details['final_probe_samples'] = expanded_samples_2
                    tie_resolution_details['final_base_score'] = round_2_result['base_score']
                    tie_resolution_details['final_best_improvement'] = round_2_result['best_improvement']
                    
                    return final_target_layer, tie_resolution_details
            else:
                # 第二轮样本不足，选择最后一层
                final_target_layer = max(remaining_tied_layers)
                print(f"⚠️ 第二轮样本不足，选择最后一层: {final_target_layer}")
                logger.info(f"第二轮样本不足，选择最后一层: {final_target_layer}")
                
                tie_resolution_details['final_target_layer'] = final_target_layer
                tie_resolution_details['final_probe_samples'] = expanded_samples_1
                tie_resolution_details['final_base_score'] = round_1_result['base_score']
                tie_resolution_details['final_best_improvement'] = round_1_result['best_improvement']
                
                return final_target_layer, tie_resolution_details
    else:
        # 第一轮样本不足，选择最后一层
        final_target_layer = max(best_layers)
        print(f"⚠️ 第一轮样本不足，选择最后一层: {final_target_layer}")
        logger.info(f"第一轮样本不足，选择最后一层: {final_target_layer}")
        
        tie_resolution_details['final_target_layer'] = final_target_layer
        return final_target_layer, tie_resolution_details

def _evaluate_tie_resolution_round_vlmeval(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                                          candidate_layers: List[int], samples: List[Dict],
                                          work_dir: str, logger, round_num: int, 
                                          description: str) -> Dict:
    """
    评估tie解决的一轮结果 - VLMEvalKit版本
    """
    print(f"📊 VLMEvalKit风格Tie解决第{round_num}轮 ({description})：评估 {len(candidate_layers)} 个候选层...")
    logger.info(f"VLMEvalKit风格Tie解决第{round_num}轮 ({description})：评估候选层 {candidate_layers}")
    
    round_work_dir = os.path.join(work_dir, f"tie_resolution_round_{round_num}")
    os.makedirs(round_work_dir, exist_ok=True)
    
    # 🔑 重新评估基准得分（使用VLMEvalKit方式）
    base_score = evaluation_engine.evaluate_samples_with_cut_layer(
        model_wrapper, samples,
        cut_layer=-1,  # 不切断任何层
        work_dir=os.path.join(round_work_dir, "base_eval")
    )
    
    print(f"✅ 第{round_num}轮基准得分: {base_score:.4f}")
    logger.info(f"第{round_num}轮基准得分: {base_score:.4f}")
    
    # 🔑 重新评估候选层（使用VLMEvalKit方式）
    layer_scores = {}
    layer_improvements = {}
    
    for layer_idx in candidate_layers:
        try:
            layer_work_dir = os.path.join(round_work_dir, f"layer_{layer_idx}_eval")
            
            layer_score = evaluation_engine.evaluate_samples_with_cut_layer(
                model_wrapper, samples,
                cut_layer=layer_idx,
                cut_module="self_attn",
                work_dir=layer_work_dir
            )
            
            layer_scores[layer_idx] = layer_score
            improvement = layer_score - base_score
            layer_improvements[layer_idx] = improvement
            
            print(f"  📊 第{round_num}轮 - 层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
            logger.info(f"第{round_num}轮 - 层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
            
        except Exception as e:
            print(f"❌ 第{round_num}轮评估第 {layer_idx} 层时出错: {e}")
            logger.error(f"第{round_num}轮评估第 {layer_idx} 层时出错: {e}")
            layer_scores[layer_idx] = 0.0
            layer_improvements[layer_idx] = -float('inf')
    
    # 分析结果
    positive_improvements = {layer: improvement for layer, improvement 
                           in layer_improvements.items() if improvement > 0}
    
    if positive_improvements:
        best_improvement = max(positive_improvements.values())
        best_layers = [layer for layer, improvement in positive_improvements.items() 
                      if abs(improvement - best_improvement) < 1e-6]
    else:
        best_improvement = max(layer_improvements.values()) if layer_improvements else -float('inf')
        best_layers = [layer for layer, improvement in layer_improvements.items() 
                      if abs(improvement - best_improvement) < 1e-6]
    
    result = {
        'round': round_num,
        'description': description,
        'candidate_layers': candidate_layers,
        'samples_count': len(samples),
        'base_score': base_score,
        'layer_scores': layer_scores,
        'layer_improvements': layer_improvements,
        'best_improvement': best_improvement,
        'final_best_layers': best_layers,
        'tie_resolved': len(best_layers) == 1
    }
    
    print(f"📊 第{round_num}轮结果：最佳层 {best_layers}，改进 {best_improvement:+.4f}")
    logger.info(f"第{round_num}轮结果：最佳层 {best_layers}，改进 {best_improvement:+.4f}，tie已解决: {len(best_layers) == 1}")
    
    return result


def _verify_single_target_layer(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                               initial_target_layer: int, current_probe_samples: List[Dict], 
                               base_score: float, all_samples: List[Dict], used_sample_indices: set,
                               work_dir: str, logger) -> Tuple[int, Dict]:
    """
    验证单一目标层的稳定性 - 修改版：当目标层与历史层重合时直接确定
    
    Returns:
        Tuple[int, Dict]: (确认的目标层, 验证详情)
    """
    print(f"🔍 开始单一目标层验证过程...")
    logger.info(f"开始单一目标层验证过程 - 初始目标层: {initial_target_layer}")
    
    verification_details = {
        'initial_target_layer': initial_target_layer,
        'verification_rounds': [],
        'additional_used_indices': set(),
        'converged': False,
        'convergence_reason': '',
        'final_target_layer': initial_target_layer,
        'historical_target_layers': [initial_target_layer]  # 记录所有历史出现的目标层
    }
    
    if not all_samples:
        print(f"⚠️ 没有额外样本可用于验证，直接确认目标层: {initial_target_layer}")
        logger.info(f"没有额外样本可用于验证，直接确认目标层: {initial_target_layer}")
        verification_details['convergence_reason'] = '无额外样本'
        return initial_target_layer, verification_details
    
    verification_samples = current_probe_samples.copy()
    verification_round = 0
    max_verification_rounds = 5
    historical_target_layers = [initial_target_layer]  # 记录历史出现的目标层
    
    while verification_round < max_verification_rounds:
        verification_round += 1
        
        # 获取可用样本
        available_samples = [sample for sample in all_samples 
                           if sample.get('index', -1) not in used_sample_indices]
        
        if len(available_samples) == 0:
            print(f"⚠️ 验证第{verification_round}轮：没有更多样本可用")
            logger.info(f"验证第{verification_round}轮：没有更多样本可用")
            verification_details['convergence_reason'] = '无更多样本可用'
            break
        
        # 添加1个样本
        import random
        additional_sample = random.choice(available_samples)
        verification_samples.append(additional_sample)
        
        additional_index = additional_sample.get('index', -1)
        used_sample_indices.add(additional_index)
        verification_details['additional_used_indices'].add(additional_index)
        
        print(f"🎲 验证第{verification_round}轮：添加样本索引={additional_index}，当前样本数={len(verification_samples)}")
        logger.info(f"验证第{verification_round}轮：添加样本索引={additional_index}，当前样本数={len(verification_samples)}")
        
        # 重新评估基准得分
        verify_work_dir = os.path.join(work_dir, f"verification_round_{verification_round}")
        os.makedirs(verify_work_dir, exist_ok=True)
        
        model_wrapper.restore_original_state()
        verify_base_score = evaluation_engine.evaluate_samples(
            model_wrapper, verification_samples,
            work_dir=os.path.join(verify_work_dir, "base_eval")
        )
        
        # 重新评估所有有正向改进的层
        layer_scores = {}
        layer_improvements = {}
        num_layers = model_wrapper.get_num_layers()
        
        print(f"🔄 验证第{verification_round}轮：重新评估所有层...")
        logger.info(f"验证第{verification_round}轮：重新评估所有层...")
        
        for layer_idx in range(1, num_layers):
            try:
                layer_work_dir = os.path.join(verify_work_dir, f"layer_{layer_idx}_eval")
                
                model_wrapper.apply_cut_layer(layer_idx, module_type="self_attn")
                layer_score = evaluation_engine.evaluate_samples(
                    model_wrapper, verification_samples,
                    work_dir=layer_work_dir
                )
                
                layer_scores[layer_idx] = layer_score
                improvement = layer_score - verify_base_score
                layer_improvements[layer_idx] = improvement
                
            except Exception as e:
                print(f"❌ 验证第{verification_round}轮评估第{layer_idx}层时出错: {e}")
                logger.error(f"验证第{verification_round}轮评估第{layer_idx}层时出错: {e}")
                layer_scores[layer_idx] = 0.0
                layer_improvements[layer_idx] = -float('inf')
                continue
        
        # 找到新的最佳层
        positive_improvements = {layer: improvement for layer, improvement 
                               in layer_improvements.items() if improvement > 0}
        
        if positive_improvements:
            best_improvement = max(positive_improvements.values())
            best_layers = [layer for layer, improvement in positive_improvements.items() 
                          if abs(improvement - best_improvement) < 1e-6]
            
            if len(best_layers) == 1:
                new_target_layer = best_layers[0]
            else:
                # 如果有多个最佳层，选择中位数层
                new_target_layer = sorted(best_layers)[len(best_layers) // 2]
        else:
            new_target_layer = -1
        
        # 记录本轮验证结果
        round_details = {
            'round': verification_round,
            'added_sample_index': additional_index,
            'total_samples': len(verification_samples),
            'base_score': verify_base_score,
            'best_improvement': best_improvement if positive_improvements else -float('inf'),
            'best_layers': best_layers if positive_improvements else [],
            'new_target_layer': new_target_layer,
            'historical_layers_before': historical_target_layers.copy(),
            'matches_history': new_target_layer in historical_target_layers
        }
        verification_details['verification_rounds'].append(round_details)
        
        print(f"📊 验证第{verification_round}轮结果：")
        print(f"  - 新目标层: {new_target_layer}")
        print(f"  - 历史层: {historical_target_layers}")
        print(f"  - 与历史层重合: {new_target_layer in historical_target_layers}")
        print(f"  - 最佳改进: {best_improvement if positive_improvements else -float('inf'):+.4f}")
        
        logger.info(f"验证第{verification_round}轮结果：新目标层={new_target_layer}, 历史层={historical_target_layers}, 重合={new_target_layer in historical_target_layers}")
        
        # 🔑 新逻辑：检查是否与历史层重合
        if new_target_layer in historical_target_layers:
            # 与历史层重合，直接确定为该层
            print(f"🎯 目标层 {new_target_layer} 与历史层重合！直接确定为目标层")
            logger.info(f"目标层 {new_target_layer} 与历史层重合！直接确定为目标层")
            
            verification_details['converged'] = True
            verification_details['convergence_reason'] = f'与第{historical_target_layers.index(new_target_layer) + 1}轮历史层重合'
            verification_details['final_target_layer'] = new_target_layer
            verification_details['historical_target_layers'] = historical_target_layers
            
            break
        else:
            # 不重合，添加到历史记录中，继续验证
            historical_target_layers.append(new_target_layer)
            print(f"🔄 目标层 {new_target_layer} 未与历史层重合，添加到历史记录，继续验证")
            logger.info(f"目标层 {new_target_layer} 未与历史层重合，添加到历史记录: {historical_target_layers}")
    
    # 如果验证轮次用完仍未收敛
    if not verification_details['converged']:
        # 使用最后确定的目标层
        final_target_layer = historical_target_layers[-1] if historical_target_layers else initial_target_layer
        print(f"⚠️ 验证未收敛，使用最后确定的目标层: {final_target_layer}")
        logger.info(f"验证未收敛，使用最后确定的目标层: {final_target_layer}")
        
        verification_details['final_target_layer'] = final_target_layer
        verification_details['convergence_reason'] = '达到最大验证轮次'
        verification_details['historical_target_layers'] = historical_target_layers
    
    print(f"📊 验证过程总结:")
    print(f"  - 初始目标层: {initial_target_layer}")
    print(f"  - 历史目标层序列: {verification_details['historical_target_layers']}")
    print(f"  - 最终目标层: {verification_details['final_target_layer']}")
    print(f"  - 收敛原因: {verification_details['convergence_reason']}")
    
    logger.info(f"验证过程总结: 初始={initial_target_layer}, 历史序列={verification_details['historical_target_layers']}, 最终={verification_details['final_target_layer']}, 收敛原因={verification_details['convergence_reason']}")
    
    return verification_details['final_target_layer'], verification_details

def _resolve_multi_layer_tie(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                            best_layers: List[int], best_improvement: float,
                            current_probe_samples: List[Dict], base_score: float,
                            all_samples: List[Dict], used_sample_indices: set,
                            work_dir: str, logger, shot: int) -> Tuple[int, Dict]:
    """
    解决多层tie的问题 - 分步增加样本进行解决
    
    Returns:
        Tuple[int, Dict]: (最终目标层, 解决详情)
    """
    print(f"🔧 开始多层tie解决过程...")
    logger.info(f"开始多层tie解决过程 - tied层: {best_layers}")
    
    tie_resolution_details = {
        'original_tied_layers': best_layers.copy(),
        'original_best_improvement': best_improvement,
        'resolution_rounds': [],
        'additional_used_indices': set(),
        'final_target_layer': -1,
        'final_probe_samples': current_probe_samples,
        'final_base_score': base_score,
        'additional_layer_scores': {},
        'additional_layer_improvements': {},
        'final_best_improvement': best_improvement
    }
    
    if not all_samples:
        # 没有额外样本，直接选择最后一层
        final_target_layer = max(best_layers)
        print(f"⚠️ 没有额外样本，选择最后一层: {final_target_layer}")
        logger.info(f"没有额外样本，选择最后一层: {final_target_layer}")
        tie_resolution_details['final_target_layer'] = final_target_layer
        return final_target_layer, tie_resolution_details
    
    # 第一次增样：增加 shot/2 个样本
    available_samples = [sample for sample in all_samples 
                        if sample.get('index', -1) not in used_sample_indices]
    
    additional_samples_needed_1 = max(1, shot // 2)
    if len(available_samples) >= additional_samples_needed_1:
        print(f"📈 第一轮tie解决：增加 {additional_samples_needed_1} 个样本...")
        logger.info(f"第一轮tie解决：增加 {additional_samples_needed_1} 个样本...")
        
        import random
        additional_samples_1 = random.sample(available_samples, additional_samples_needed_1)
        expanded_samples_1 = current_probe_samples + additional_samples_1
        
        additional_indices_1 = {sample.get('index', -1) for sample in additional_samples_1}
        used_sample_indices.update(additional_indices_1)
        tie_resolution_details['additional_used_indices'].update(additional_indices_1)
        
        # 第一轮解决
        round_1_result = _evaluate_tie_resolution_round(
            model_wrapper, evaluation_engine, best_layers, expanded_samples_1,
            work_dir, logger, 1, "增加shot/2样本"
        )
        
        tie_resolution_details['resolution_rounds'].append(round_1_result)
        tie_resolution_details['additional_layer_scores'].update(round_1_result['layer_scores'])
        tie_resolution_details['additional_layer_improvements'].update(round_1_result['layer_improvements'])
        
        if len(round_1_result['final_best_layers']) == 1:
            # 第一轮就解决了tie
            final_target_layer = round_1_result['final_best_layers'][0]
            print(f"🎯 第一轮tie解决成功！目标层: {final_target_layer}")
            logger.info(f"第一轮tie解决成功！目标层: {final_target_layer}")
            
            tie_resolution_details['final_target_layer'] = final_target_layer
            tie_resolution_details['final_probe_samples'] = expanded_samples_1
            tie_resolution_details['final_base_score'] = round_1_result['base_score']
            tie_resolution_details['final_best_improvement'] = round_1_result['best_improvement']
            
            return final_target_layer, tie_resolution_details
        else:
            # 仍然tie，准备第二轮
            remaining_tied_layers = round_1_result['final_best_layers']
            print(f"⚠️ 第一轮后仍有tie: {remaining_tied_layers}")
            logger.info(f"第一轮后仍有tie: {remaining_tied_layers}")
            
            # 第二次增样：再增加 shot/4 个样本
            available_samples = [sample for sample in all_samples 
                               if sample.get('index', -1) not in used_sample_indices]
            
            additional_samples_needed_2 = max(1, shot // 4)
            if len(available_samples) >= additional_samples_needed_2:
                print(f"📈 第二轮tie解决：再增加 {additional_samples_needed_2} 个样本...")
                logger.info(f"第二轮tie解决：再增加 {additional_samples_needed_2} 个样本...")
                
                additional_samples_2 = random.sample(available_samples, additional_samples_needed_2)
                expanded_samples_2 = expanded_samples_1 + additional_samples_2
                
                additional_indices_2 = {sample.get('index', -1) for sample in additional_samples_2}
                used_sample_indices.update(additional_indices_2)
                tie_resolution_details['additional_used_indices'].update(additional_indices_2)
                
                # 第二轮解决
                round_2_result = _evaluate_tie_resolution_round(
                    model_wrapper, evaluation_engine, remaining_tied_layers, expanded_samples_2,
                    work_dir, logger, 2, "再增加shot/4样本"
                )
                
                tie_resolution_details['resolution_rounds'].append(round_2_result)
                tie_resolution_details['additional_layer_scores'].update(round_2_result['layer_scores'])
                tie_resolution_details['additional_layer_improvements'].update(round_2_result['layer_improvements'])
                
                if len(round_2_result['final_best_layers']) == 1:
                    # 第二轮解决了tie
                    final_target_layer = round_2_result['final_best_layers'][0]
                    print(f"🎯 第二轮tie解决成功！目标层: {final_target_layer}")
                    logger.info(f"第二轮tie解决成功！目标层: {final_target_layer}")
                    
                    tie_resolution_details['final_target_layer'] = final_target_layer
                    tie_resolution_details['final_probe_samples'] = expanded_samples_2
                    tie_resolution_details['final_base_score'] = round_2_result['base_score']
                    tie_resolution_details['final_best_improvement'] = round_2_result['best_improvement']
                    
                    return final_target_layer, tie_resolution_details
                else:
                    # 仍然tie，选择最后一层
                    final_tied_layers = round_2_result['final_best_layers']
                    final_target_layer = max(final_tied_layers)
                    print(f"⚠️ 第二轮后仍有tie: {final_tied_layers}，选择最后一层: {final_target_layer}")
                    logger.info(f"第二轮后仍有tie: {final_tied_layers}，选择最后一层: {final_target_layer}")
                    
                    tie_resolution_details['final_target_layer'] = final_target_layer
                    tie_resolution_details['final_probe_samples'] = expanded_samples_2
                    tie_resolution_details['final_base_score'] = round_2_result['base_score']
                    tie_resolution_details['final_best_improvement'] = round_2_result['best_improvement']
                    
                    return final_target_layer, tie_resolution_details
            else:
                # 第二轮样本不足，选择最后一层
                final_target_layer = max(remaining_tied_layers)
                print(f"⚠️ 第二轮样本不足，选择最后一层: {final_target_layer}")
                logger.info(f"第二轮样本不足，选择最后一层: {final_target_layer}")
                
                tie_resolution_details['final_target_layer'] = final_target_layer
                tie_resolution_details['final_probe_samples'] = expanded_samples_1
                tie_resolution_details['final_base_score'] = round_1_result['base_score']
                tie_resolution_details['final_best_improvement'] = round_1_result['best_improvement']
                
                return final_target_layer, tie_resolution_details
    else:
        # 第一轮样本不足，选择最后一层
        final_target_layer = max(best_layers)
        print(f"⚠️ 第一轮样本不足，选择最后一层: {final_target_layer}")
        logger.info(f"第一轮样本不足，选择最后一层: {final_target_layer}")
        
        tie_resolution_details['final_target_layer'] = final_target_layer
        return final_target_layer, tie_resolution_details


def _evaluate_tie_resolution_round(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                                  candidate_layers: List[int], samples: List[Dict],
                                  work_dir: str, logger, round_num: int, 
                                  description: str) -> Dict:
    """
    评估tie解决的一轮结果
    
    Returns:
        Dict: 本轮评估结果
    """
    print(f"📊 Tie解决第{round_num}轮 ({description})：评估 {len(candidate_layers)} 个候选层...")
    logger.info(f"Tie解决第{round_num}轮 ({description})：评估候选层 {candidate_layers}")
    
    round_work_dir = os.path.join(work_dir, f"tie_resolution_round_{round_num}")
    os.makedirs(round_work_dir, exist_ok=True)
    
    # 重新评估基准得分
    model_wrapper.restore_original_state()
    base_score = evaluation_engine.evaluate_samples(
        model_wrapper, samples,
        work_dir=os.path.join(round_work_dir, "base_eval")
    )
    
    print(f"✅ 第{round_num}轮基准得分: {base_score:.4f}")
    logger.info(f"第{round_num}轮基准得分: {base_score:.4f}")
    
    # 重新评估候选层
    layer_scores = {}
    layer_improvements = {}
    
    for layer_idx in candidate_layers:
        try:
            layer_work_dir = os.path.join(round_work_dir, f"layer_{layer_idx}_eval")
            
            model_wrapper.apply_cut_layer(layer_idx, module_type="self_attn")
            layer_score = evaluation_engine.evaluate_samples(
                model_wrapper, samples,
                work_dir=layer_work_dir
            )
            
            layer_scores[layer_idx] = layer_score
            improvement = layer_score - base_score
            layer_improvements[layer_idx] = improvement
            
            print(f"  📊 第{round_num}轮 - 层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
            logger.info(f"第{round_num}轮 - 层 {layer_idx} 得分: {layer_score:.4f} (改进: {improvement:+.4f})")
            
        except Exception as e:
            print(f"❌ 第{round_num}轮评估第 {layer_idx} 层时出错: {e}")
            logger.error(f"第{round_num}轮评估第 {layer_idx} 层时出错: {e}")
            layer_scores[layer_idx] = 0.0
            layer_improvements[layer_idx] = -float('inf')
    
    # 分析结果
    positive_improvements = {layer: improvement for layer, improvement 
                           in layer_improvements.items() if improvement > 0}
    
    if positive_improvements:
        best_improvement = max(positive_improvements.values())
        best_layers = [layer for layer, improvement in positive_improvements.items() 
                      if abs(improvement - best_improvement) < 1e-6]
    else:
        best_improvement = max(layer_improvements.values()) if layer_improvements else -float('inf')
        best_layers = [layer for layer, improvement in layer_improvements.items() 
                      if abs(improvement - best_improvement) < 1e-6]
    
    result = {
        'round': round_num,
        'description': description,
        'candidate_layers': candidate_layers,
        'samples_count': len(samples),
        'base_score': base_score,
        'layer_scores': layer_scores,
        'layer_improvements': layer_improvements,
        'best_improvement': best_improvement,
        'final_best_layers': best_layers,
        'tie_resolved': len(best_layers) == 1
    }
    
    print(f"📊 第{round_num}轮结果：最佳层 {best_layers}，改进 {best_improvement:+.4f}")
    logger.info(f"第{round_num}轮结果：最佳层 {best_layers}，改进 {best_improvement:+.4f}，tie已解决: {len(best_layers) == 1}")
    
    return result


def final_evaluation_vlmeval_style(model_wrapper: ModelWrapper, evaluation_engine: EvaluationEngine,
                                  eval_samples: List[Dict], target_layer: int, work_dir: str) -> Tuple[float, float]:
    """
    使用VLMEvalKit内部机制进行最终性能评估
    """
    final_logger = setup_stage_logging(work_dir, "final")
    final_logger.info(f"开始VLMEvalKit风格最终性能评估 (评估样本数: {len(eval_samples)}, 目标层: {target_layer})")
    
    print(f"🎯 开始VLMEvalKit风格最终性能评估 (评估样本数: {len(eval_samples)}, 目标层: {target_layer})")
    
    if len(eval_samples) == 0:
        print("⚠️ 评估样本为空，跳过最终评估")
        final_logger.warning("评估样本为空，跳过最终评估")
        return 0.0, 0.0
    
    # 第一步：评估基准模型（不使用cut_layer）
    print(f"📊 第一步：评估基准模型...")
    final_logger.info(f"评估基准模型...")
    
    base_eval_score = evaluation_engine.evaluate_samples_with_cut_layer(
        model_wrapper, eval_samples,
        cut_layer=-1,  # 不切断任何层
        work_dir=os.path.join(work_dir, "base_model_eval")
    )
    
    print(f"✅ 基准模型得分: {base_eval_score:.4f}")
    final_logger.info(f"基准模型得分: {base_eval_score:.4f}")
    
    # 第二步：评估目标层处理后的模型
    if target_layer == -1:
        print("⚠️ 目标层为-1，跳过目标层评估")
        final_logger.info("目标层为-1，跳过目标层评估")
        target_layer_score = base_eval_score
    else:
        print(f"🔧 第二步：使用VLMEvalKit cut_layer机制评估第 {target_layer} 层...")
        final_logger.info(f"使用VLMEvalKit cut_layer机制评估第 {target_layer} 层...")
        
        target_layer_score = evaluation_engine.evaluate_samples_with_cut_layer(
            model_wrapper, eval_samples,
            cut_layer=target_layer,
            cut_module="self_attn",
            work_dir=os.path.join(work_dir, "target_layer_eval")
        )
        
        print(f"✅ 目标层处理后得分: {target_layer_score:.4f}")
        final_logger.info(f"目标层处理后得分: {target_layer_score:.4f}")
    
    # 性能比较
    performance_change = target_layer_score - base_eval_score
    performance_change_pct = (performance_change / base_eval_score * 100) if base_eval_score != 0 else 0
    
    print(f"\n📊 VLMEvalKit风格最终评估结果:")
    print(f"  - 基准模型得分: {base_eval_score:.4f}")
    print(f"  - 目标层处理后得分: {target_layer_score:.4f}")
    print(f"  - 性能变化: {performance_change:+.4f} ({performance_change_pct:+.2f}%)")
    
    final_logger.info(f"VLMEvalKit风格最终评估结果:")
    final_logger.info(f"  - 基准模型得分: {base_eval_score:.4f}")
    final_logger.info(f"  - 目标层处理后得分: {target_layer_score:.4f}")
    final_logger.info(f"  - 性能变化: {performance_change:+.4f} ({performance_change_pct:+.2f}%)")
    final_logger.info("VLMEvalKit风格最终评估阶段完成")
    
    return base_eval_score, target_layer_score

def filter_tasks(subtasks: Dict[str, List[Dict]], args) -> Dict[str, List[Dict]]:
    """
    根据命令行参数过滤任务
    
    Args:
        subtasks: 所有子任务字典
        args: 命令行参数
        
    Returns:
        Dict[str, List[Dict]]: 过滤后的子任务字典
    """
    all_task_names = list(subtasks.keys())
    
    # 如果只是列出任务，直接返回
    if args.list_tasks:
        print("📋 数据集中的所有任务:")
        print("-" * 50)
        for i, task_name in enumerate(all_task_names, 1):
            sample_count = len(subtasks[task_name])
            print(f"{i:2d}. {task_name:<30} ({sample_count} 个样本)")
        print("-" * 50)
        print(f"总计: {len(all_task_names)} 个任务")
        return {}
    
    filtered_tasks = {}
    
    # 1. 如果指定了target_tasks，只处理这些任务
    if args.target_tasks:
        print(f"🎯 指定处理任务: {args.target_tasks}")
        
        for task_name in args.target_tasks:
            if task_name in subtasks:
                filtered_tasks[task_name] = subtasks[task_name]
                print(f"  ✅ 找到任务: {task_name} ({len(subtasks[task_name])} 个样本)")
            else:
                print(f"  ❌ 未找到任务: {task_name}")
                print(f"     可用任务: {all_task_names}")
        
        if not filtered_tasks:
            print("❌ 没有找到任何指定的任务")
            return {}
    
    # 2. 如果使用了正则表达式匹配
    elif args.task_pattern:
        import re
        print(f"🔍 使用正则表达式匹配: {args.task_pattern}")
        
        try:
            pattern = re.compile(args.task_pattern, re.IGNORECASE)
            for task_name in all_task_names:
                if pattern.search(task_name):
                    filtered_tasks[task_name] = subtasks[task_name]
                    print(f"  ✅ 匹配任务: {task_name} ({len(subtasks[task_name])} 个样本)")
        except re.error as e:
            print(f"❌ 正则表达式错误: {e}")
            return {}
        
        if not filtered_tasks:
            print("❌ 没有任务匹配指定的正则表达式")
            return {}
    
    # 3. 否则处理所有任务
    else:
        filtered_tasks = subtasks.copy()
        print(f"📊 处理所有任务 ({len(filtered_tasks)} 个)")
    
    # 4. 应用排除列表
    if args.exclude_tasks:
        print(f"🚫 排除任务: {args.exclude_tasks}")
        
        for task_name in args.exclude_tasks:
            if task_name in filtered_tasks:
                del filtered_tasks[task_name]
                print(f"  ✅ 已排除: {task_name}")
            else:
                print(f"  ⚠️ 任务不存在于过滤结果中: {task_name}")
    
    print(f"🎯 最终处理任务列表:")
    for i, (task_name, samples) in enumerate(filtered_tasks.items(), 1):
        print(f"  {i}. {task_name} ({len(samples)} 个样本)")
    
    return filtered_tasks



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLM模型分层探测和评估脚本")
    parser.add_argument('--model_name', type=str, required=False,default='llava_next_llama3',
                       help='要评估的模型名称')
    parser.add_argument('--dataset_name', type=str, required=False,default='MMStar',
                       help='要评估的数据集名称')
    parser.add_argument('--shot', type=int, default=5,
                       help='用于探测的样本数量')
    parser.add_argument('--output_root', type=str, default='./probe_results',
                       help='结果输出根目录')
    parser.add_argument('--sampling_strategy', type=str, 
                       choices=['l2_priority', 'category_l2_stratified', 'l2_category_stratified', 
                               'category_random', 'category_skill_stratified', 'skill_stratified'],
                       default='l2_priority',
                       help='采样策略选择')
    # 🆕 新增参数：指定特定任务
    parser.add_argument('--target_tasks', type=str, nargs='*', default=None,
                       help='指定要处理的特定任务名称列表，如果不指定则处理所有任务')
    parser.add_argument('--exclude_tasks', type=str, nargs='*', default=None,
                       help='指定要排除的任务名称列表')
    parser.add_argument('--list_tasks', action='store_true',
                       help='仅列出数据集中的所有任务，不执行探测')
    parser.add_argument('--task_pattern', type=str, default=None,
                       help='使用正则表达式匹配任务名称')
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = os.path.join(args.output_root, args.model_name, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置主日志
    main_logger = setup_logging(output_dir)
    main_logger.info(f"开始探测和评估任务: 模型={args.model_name}, 数据集={args.dataset_name}, shot={args.shot}")
    
    try:
        # 1. 初始化组件
        main_logger.info("🔧 初始化系统组件...")
        
        # 🔑 使用新的采样策略参数加载子任务提取器
        subtask_extractor = SubtaskExtractor(args.dataset_name, args.sampling_strategy)
        if not subtask_extractor.load_dataset():
            main_logger.error("数据集加载失败")
            return
        
        # 提取子任务
        subtasks = subtask_extractor.extract_subtasks()
        if not subtasks:
            main_logger.error("没有找到任何子任务")
            return
        
        # 🆕 应用任务过滤
        filtered_subtasks = filter_tasks(subtasks, args)
        
        # 如果只是列出任务，直接返回
        if args.list_tasks:
            return
        
        if not filtered_subtasks:
            main_logger.error("经过过滤后没有任务需要处理")
            return
        
        # 记录使用的分类策略
        main_logger.info(f"📊 分类策略: 使用 '{subtask_extractor.primary_category_field}' 字段")
        main_logger.info(f"📋 采样策略: {args.sampling_strategy}")

        # 加载模型
        model_wrapper = ModelWrapper(args.model_name)
        if not model_wrapper.load_model():
            main_logger.error("模型加载失败")
            return
        
        # 初始化评估引擎
        evaluation_engine = EvaluationEngine(args.dataset_name)
        if not evaluation_engine.load_dataset():
            main_logger.error("评估引擎数据集加载失败")
            return
        
        # 2. 遍历所有子任务
        final_results = {}
        total_subtasks = len(filtered_subtasks)
        
        main_logger.info(f"🚀 开始处理 {total_subtasks} 个子任务...")
        
        # 创建汇总结果文件
        summary_log_file = os.path.join(output_dir, "all_tasks_summary.log")
        summary_logger = logging.getLogger("summary")
        summary_logger.setLevel(logging.INFO)
        
        # 清除已有的handlers，避免重复
        for handler in summary_logger.handlers[:]:
            summary_logger.removeHandler(handler)
            
        summary_handler = logging.FileHandler(summary_log_file, encoding='utf-8')
        summary_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        summary_handler.setFormatter(formatter)
        summary_logger.addHandler(summary_handler)
        
        summary_logger.info("=" * 80)
        summary_logger.info("所有子任务性能评估汇总")
        summary_logger.info("=" * 80)
        summary_logger.info(f"模型: {args.model_name}")
        summary_logger.info(f"数据集: {args.dataset_name}")
        summary_logger.info(f"Shot: {args.shot}")
        summary_logger.info(f"子任务总数: {total_subtasks}")
        summary_logger.info("-" * 80)
        summary_logger.info(f"{'任务名称':<20} {'目标层':<6} {'基准':<7} {'处理后':<7} {'变化':<10} {'状态':<8}")
        summary_logger.info("-" * 80)
        
        for subtask_idx, (subtask_name, samples) in enumerate(filtered_subtasks.items(), 1):
            # 为每个子任务设置独立的日志记录器
            subtask_work_dir = os.path.join(output_dir, f"subtask_{subtask_name.replace(' ', '_')}")
            os.makedirs(subtask_work_dir, exist_ok=True)
            subtask_logger = setup_subtask_logging(subtask_work_dir, subtask_name)
            
            main_logger.info(f"处理子任务 {subtask_idx}/{total_subtasks}: {subtask_name} ({len(samples)} 个样本)")
            subtask_logger.info(f"开始处理子任务 {subtask_name} ({subtask_idx}/{total_subtasks})")
            subtask_logger.info(f"样本总数: {len(samples)}")
            
            try:
                # 检查样本数量
                if len(samples) < args.shot:
                    warning_msg = f"子任务 {subtask_name} 样本数量 ({len(samples)}) 少于shot数量 ({args.shot})，跳过"
                    main_logger.warning(warning_msg)
                    subtask_logger.warning(warning_msg)
                    continue
                
                # 🔑 使用新的采样策略
                print(f"📊 子任务 {subtask_name} - 开始 {args.sampling_strategy} 采样...")
                subtask_logger.info(f"开始 {args.sampling_strategy} 采样 (目标shot: {args.shot})")
                subtask_logger.info(f"主要分类字段: {subtask_extractor.primary_category_field}")
                
                # 🔑 第一步：对所有样本进行策略性采样，构建探测池
                max_probe_pool_size = min(len(samples) // 3, args.shot * 4)
                probe_pool_samples, probe_pool_stats = subtask_extractor.apply_sampling_strategy(
                    samples, max_probe_pool_size
                )
                
                subtask_logger.info(f"探测池采样完成:")
                subtask_logger.info(f"  - 目标大小: {max_probe_pool_size}")
                subtask_logger.info(f"  - 实际大小: {len(probe_pool_samples)}")
                subtask_logger.info(f"  - 采样统计: {probe_pool_stats}")
                
                # 🔑 第二步：从探测池中再次采样，获取初始探测样本
                probe_samples, probe_samples_stats = subtask_extractor.apply_sampling_strategy(
                    probe_pool_samples, args.shot
                )
                
                subtask_logger.info(f"初始探测样本采样完成:")
                subtask_logger.info(f"  - 目标大小: {args.shot}")
                subtask_logger.info(f"  - 实际大小: {len(probe_samples)}")
                subtask_logger.info(f"  - 采样统计: {probe_samples_stats}")
                
                # 🔑 第三步：剩余样本作为评估集
                probe_pool_indices = {sample.get('index', -1) for sample in probe_pool_samples}
                eval_samples = [sample for sample in samples if sample.get('index', -1) not in probe_pool_indices]
                
                # 验证样本分离
                probe_indices = {sample.get('index', -1) for sample in probe_samples}
                eval_indices = {sample.get('index', -1) for sample in eval_samples}
                overlap = probe_indices & eval_indices
                
                if overlap:
                    error_msg = f"❌ 发现样本重叠！重叠的索引: {overlap}"
                    subtask_logger.error(error_msg)
                    main_logger.error(error_msg)
                    continue
                else:
                    subtask_logger.info(f"✅ 样本隔离验证通过：探测池({len(probe_pool_indices)}) vs 评估集({len(eval_indices)})，无重叠")
                
                print(f"📊 子任务 {subtask_name} 样本分配:")
                print(f"  - 总样本数: {len(samples)}")
                print(f"  - 探测池大小: {len(probe_pool_samples)} ({args.sampling_strategy}采样)")
                print(f"  - 初始探测样本: {len(probe_samples)} ({args.sampling_strategy}采样)")
                print(f"  - 评估样本数: {len(eval_samples)}")
                print(f"  - 采样策略: {args.sampling_strategy}")
                
                # 创建工作目录
                work_dir = subtask_work_dir
                
                # 阶段一：探测目标层
                subtask_logger.info("阶段一：探测目标层")
                target_layer, base_score, probe_details = probe_target_layer_vlmeval_enhanced(
                    model_wrapper, evaluation_engine, probe_samples, work_dir,
                    all_samples=probe_pool_samples, shot=args.shot  # 传入探测池用于重新采样
                )
                
                # 阶段二：最终性能评估（包含基准模型对比）
                subtask_logger.info("阶段二：最终性能评估（包含基准模型对比）")
                base_eval_performance, target_eval_performance = final_evaluation_vlmeval_style(
                    model_wrapper, evaluation_engine, eval_samples, target_layer, work_dir
                )
                
                # 🔑 计算性能变化指标
                performance_change = target_eval_performance - base_eval_performance
                performance_change_pct = (performance_change / base_eval_performance * 100) if base_eval_performance != 0 else 0
                
                # 🔑 评估有效性
                is_improvement = performance_change > 0
                is_significant = abs(performance_change) >= 0.01  # 1%的变化视为显著
                
                # 记录详细结果
                final_results[subtask_name] = {
                    'target_layer': target_layer,
                    'probe_base_score': base_score,  # 探测阶段的基准得分
                    'base_eval_performance': base_eval_performance,  # 评估阶段的基准得分
                    'target_eval_performance': target_eval_performance,  # 目标层处理后的得分
                    'performance_change': performance_change,
                    'performance_change_pct': performance_change_pct,
                    'is_improvement': is_improvement,
                    'is_significant': is_significant,
                    'details': probe_details
                }
                
                # 🔑 计算性能变化指标
                performance_change = target_eval_performance - base_eval_performance
                performance_change_pct = (performance_change / base_eval_performance * 100) if base_eval_performance != 0 else 0
                
                # 🔑 评估有效性
                is_improvement = performance_change > 0
                is_significant = abs(performance_change) >= 0.01  # 1%的变化视为显著
                
                # 记录详细结果
                final_results[subtask_name] = {
                    'target_layer': target_layer,
                    'probe_base_score': base_score,  # 探测阶段的基准得分
                    'base_eval_performance': base_eval_performance,  # 评估阶段的基准得分
                    'target_eval_performance': target_eval_performance,  # 目标层处理后的得分
                    'performance_change': performance_change,
                    'performance_change_pct': performance_change_pct,
                    'is_improvement': is_improvement,
                    'is_significant': is_significant,
                    'details': probe_details
                }
                
                # 打印单个子任务的结果摘要
                status_symbol = "📈" if is_improvement else "📉" if performance_change < 0 else "➡️"
                significance = "显著" if is_significant else "轻微"
                
                result_msg = f"\n{status_symbol} 子任务 {subtask_name} 完成:"
                subtask_logger.info(result_msg)
                main_logger.info(result_msg)
                
                details_msg = f"  - 目标层: {target_layer}\n" \
                             f"  - 基准性能: {base_eval_performance:.4f}\n" \
                             f"  - 处理后性能: {target_eval_performance:.4f}\n" \
                             f"  - 性能变化: {performance_change:+.4f} ({performance_change_pct:+.2f}%) - {significance}"
                subtask_logger.info(details_msg)
                main_logger.info(details_msg)
                
                subtask_logger.info(f"子任务 {subtask_name} 完成: target_layer={target_layer}, "
                          f"base={base_eval_performance:.4f}, target={target_eval_performance:.4f}, "
                          f"change={performance_change:+.4f} ({performance_change_pct:+.2f}%)")
                
                # 记录到汇总日志
                if target_layer >= 0:
                    if is_improvement:
                        status = "📈 改进" if is_significant else "📈 微改进"
                    elif performance_change < 0:
                        status = "📉 下降" if is_significant else "📉 微下降"
                    else:
                        status = "➡️ 无变化"
                else:
                    status = "❌ 失败"
                    
                summary_logger.info(f"{subtask_name:<20} {target_layer:<6} {base_eval_performance:<7.3f} {target_eval_performance:<7.3f} {performance_change_pct:+7.2f}% {status:<8}")
                
            except Exception as e:
                error_msg = f"处理子任务 {subtask_name} 时出错: {e}"
                main_logger.error(error_msg)
                subtask_logger.error(error_msg)
                main_logger.error(traceback.format_exc())
                subtask_logger.error(traceback.format_exc())
                
                # 记录错误结果
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
                
                # 错误信息也记录到汇总日志
                summary_logger.error(f"{subtask_name:<20} {'ERROR':<6} {'N/A':<7} {'N/A':<7} {'N/A':<10} {'❌ 失败':<8}")
                continue
        
        # 3. 保存最终结果
        results_file = os.path.join(output_dir, 'results_vlmeval_style.json')
        
        # 添加元数据
        metadata = {
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'shot': args.shot,
            'total_subtasks': len(final_results),
            
            'results': final_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        main_logger.info(f"所有结果已保存到: {results_file}")
        summary_logger.info("-" * 80)
        summary_logger.info(f"结果文件: {results_file}")
        summary_logger.info("=" * 80)
        
        # 4. 生成全面的结果分析报告
        report_msg = f"\n🎉 探测和评估完成！"
        main_logger.info(report_msg)
        summary_logger.info(report_msg)
        report_msg = f"=" * 60
        main_logger.info(report_msg)
        summary_logger.info(report_msg)
        
        # 基础统计
        successful_tasks = [name for name, result in final_results.items() 
                          if result['target_layer'] >= 0]
        improved_tasks = [name for name, result in final_results.items() 
                         if result['is_improvement'] and result['target_layer'] >= 0]
        significant_tasks = [name for name, result in final_results.items() 
                           if result['is_significant'] and result['target_layer'] >= 0]
        
        stats_msg = f"📊 总体统计:"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 总子任务数: {len(final_results)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 成功找到目标层: {len(successful_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 性能有改进: {len(improved_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 显著性改进: {len(significant_tasks)}"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 成功率: {len(successful_tasks) / len(final_results) * 100:.1f}%"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        stats_msg = f"  - 改进率: {len(improved_tasks) / len(successful_tasks) * 100:.1f}%" if successful_tasks else "  - 改进率: N/A"
        main_logger.info(stats_msg)
        summary_logger.info(stats_msg)
        
        # 性能变化统计
        if successful_tasks:
            all_changes = [final_results[name]['performance_change'] for name in successful_tasks]
            all_changes_pct = [final_results[name]['performance_change_pct'] for name in successful_tasks]
            
            perf_msg = f"\n📈 性能变化统计:"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - 平均性能变化: {np.mean(all_changes):+.4f} ({np.mean(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - 最大性能提升: {np.max(all_changes):+.4f} ({np.max(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - 最大性能下降: {np.min(all_changes):+.4f} ({np.min(all_changes_pct):+.2f}%)"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            perf_msg = f"  - 性能变化标准差: {np.std(all_changes):.4f}"
            main_logger.info(perf_msg)
            summary_logger.info(perf_msg)
            
            # 基准性能统计
            base_performances = [final_results[name]['base_eval_performance'] for name in successful_tasks]
            target_performances = [final_results[name]['target_eval_performance'] for name in successful_tasks]
            
            base_msg = f"\n📊 基准性能统计:"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - 平均基准性能: {np.mean(base_performances):.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - 平均处理后性能: {np.mean(target_performances):.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
            base_msg = f"  - 整体性能变化: {np.mean(target_performances) - np.mean(base_performances):+.4f}"
            main_logger.info(base_msg)
            summary_logger.info(base_msg)
        
        # 详细结果列表
        detail_msg = f"\n📋 详细结果:"
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        detail_msg = "-" * 80
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        detail_msg = f"{'任务名称':<20} {'目标层':<6} {'基准':<7} {'处理后':<7} {'变化':<10} {'状态':<8}"
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
                    status = "📈 改进" if result['is_significant'] else "📈 微改进"
                elif result['performance_change'] < 0:
                    status = "📉 下降" if result['is_significant'] else "📉 微下降"
                else:
                    status = "➡️ 无变化"
            else:
                status = "❌ 失败"
                
            detail_msg = f"{subtask_name:<20} {target_layer:<6} {base_perf:<7.3f} {target_perf:<7.3f} {change_pct:+7.2f}% {status:<8}"
            main_logger.info(detail_msg)
            summary_logger.info(detail_msg)
        
        detail_msg = "-" * 80
        main_logger.info(detail_msg)
        summary_logger.info(detail_msg)
        
        # 最佳和最差结果
        if successful_tasks:
            best_task = max(successful_tasks, key=lambda x: final_results[x]['performance_change'])
            worst_task = min(successful_tasks, key=lambda x: final_results[x]['performance_change'])
            
            best_msg = f"\n🏆 最佳改进:"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_result = final_results[best_task]
            best_msg = f"  - 任务: {best_task}"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_msg = f"  - 目标层: {best_result['target_layer']}"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            best_msg = f"  - 性能提升: {best_result['performance_change']:+.4f} ({best_result['performance_change_pct']:+.2f}%)"
            main_logger.info(best_msg)
            summary_logger.info(best_msg)
            
            worst_msg = f"\n⚠️ 最大下降:"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_result = final_results[worst_task]
            worst_msg = f"  - 任务: {worst_task}"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_msg = f"  - 目标层: {worst_result['target_layer']}"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
            worst_msg = f"  - 性能变化: {worst_result['performance_change']:+.4f} ({worst_result['performance_change_pct']:+.2f}%)"
            main_logger.info(worst_msg)
            summary_logger.info(worst_msg)
        
        end_msg = f"\n💾 结果文件: {results_file}"
        main_logger.info(end_msg)
        summary_logger.info(end_msg)
        main_logger.info("探测和评估任务完成")
        summary_logger.info("探测和评估任务完成")
        summary_logger.info("=" * 80)
        
    except Exception as e:
        main_logger.error(f"主程序执行出错: {e}")
        main_logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()