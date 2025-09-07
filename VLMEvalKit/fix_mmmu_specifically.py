#!/usr/bin/env python3
"""
专门修复 MMMU_DEV_VAL 数据集读取问题
"""

import os
import sys
sys.path.append('/data_all/intern05/VLM_Merging/VLMEvalKit')

def fix_mmmu_local_loading():
    """修复 MMMU 本地加载"""
    
    # 方法1：直接复制到期望位置
    from vlmeval.smp import LMUDataRoot
    
    data_root = LMUDataRoot()
    local_file = '/home/intern05/LMUData/MMMU_DEV_VAL.tsv'
    target_file = os.path.join(data_root, 'MMMU_DEV_VAL.tsv')
    
    if os.path.exists(local_file):
        os.makedirs(data_root, exist_ok=True)
        
        # 如果目标文件不存在或者比本地文件旧，就复制
        if not os.path.exists(target_file) or \
           os.path.getmtime(local_file) > os.path.getmtime(target_file):
            
            import shutil
            shutil.copy2(local_file, target_file)
            print(f"✅ 复制 MMMU_DEV_VAL.tsv 到: {target_file}")
        else:
            print(f"✅ MMMU_DEV_VAL.tsv 已存在: {target_file}")
    else:
        print(f"❌ 本地文件不存在: {local_file}")

def test_mmmu_loading():
    """测试 MMMU 加载"""
    try:
        from vlmeval.dataset import build_dataset
        
        dataset = build_dataset('MMMU_DEV_VAL')
        if dataset:
            print(f"✅ MMMU_DEV_VAL 加载成功!")
            print(f"   数据量: {len(dataset.data)}")
            print(f"   列数: {len(dataset.data.columns)}")
            return True
        else:
            print("❌ MMMU_DEV_VAL 加载失败")
            return False
    except Exception as e:
        print(f"❌ MMMU_DEV_VAL 加载异常: {e}")
        return False

if __name__ == '__main__':
    print("🔧 修复 MMMU_DEV_VAL 数据集...")
    fix_mmmu_local_loading()
    print("\n🧪 测试数据集加载...")
    test_mmmu_loading()
