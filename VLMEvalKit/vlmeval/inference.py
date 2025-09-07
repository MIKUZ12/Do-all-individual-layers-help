import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import pdb
import json
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'

# # 🔧 重要：添加全局变量定义
# captured_data = {
#     'samples': [],  # 存储多个样本
#     'max_samples': 1500,  # 最大采样数量
#     'captured_count': 0
# }

# def patch_attention_forward(original_forward, layer_num):
#     """
#     直接修改attention模块的forward方法来捕获输入输出
#     """
#     def patched_forward(self, *args, **kwargs):
#         global captured_data
        
#         # 检查是否已经收集了足够的样本
#         if captured_data['captured_count'] >= captured_data['max_samples']:
#             return original_forward(*args, **kwargs)
        
#         try:
#             # 获取输入的hidden_states
#             if 'hidden_states' in kwargs:
#                 hidden_states_input = kwargs['hidden_states']
                
#                 # 🔧 可选：只捕获长序列（通常是完整输入，不是单token生成）
#                 # 如果您只想要初始的完整序列处理，可以启用这个过滤
#                 sequence_length = hidden_states_input.shape[1]
#                 if sequence_length < 100:  # 跳过短序列（单token生成）
#                     return original_forward(*args, **kwargs)
#             print(f"🔍 Forward方法被调用 - 第{layer_num}层")
#             print(f"   self类型: {type(self)}")
#             print(f"   args数量: {len(args)}")
#             print(f"   kwargs键: {list(kwargs.keys())}")
            
#             # 详细打印参数信息
#             for i, arg in enumerate(args):
#                 if hasattr(arg, 'shape'):
#                     print(f"   args[{i}]: tensor, 形状={arg.shape}, 设备={arg.device}, dtype={arg.dtype}")
#                 else:
#                     print(f"   args[{i}]: {type(arg)}")
            
#             for key, value in kwargs.items():
#                 if hasattr(value, 'shape'):
#                     print(f"   kwargs[{key}]: tensor, 形状={value.shape}")
#                 elif value is not None:
#                     print(f"   kwargs[{key}]: {type(value)}")
#                 else:
#                     print(f"   kwargs[{key}]: None")
            
#             # 获取输入的hidden_states - 这是关键！
#             hidden_states_input = None
            
#             # 对于LlamaAttention，第一个参数通常是hidden_states
#             if args and len(args) > 0 and hasattr(args[0], 'shape'):
#                 hidden_states_input = args[0]
#                 print(f"✅ 从args[0]获取hidden_states: {hidden_states_input.shape}")
#             elif 'hidden_states' in kwargs:
#                 hidden_states_input = kwargs['hidden_states']
#                 print(f"✅ 从kwargs获取hidden_states: {hidden_states_input.shape}")
#             else:
#                 print("❌ 无法找到hidden_states输入")
#                 # 尝试调用原始方法并返回
#                 return original_forward(*args, **kwargs)
            
#             print(f"📥 确认输入hidden_states: {hidden_states_input.shape}")
            
#             # 调用原始的forward方法
#             print("🔄 调用原始forward方法...")
#             output = original_forward(*args, **kwargs)
#             print(f"📤 原始forward返回类型: {type(output)}")
            
#             # 获取输出 - 对于self_attn，输出通常是(hidden_states, attn_weights, past_key_value)
#             attention_output = None
#             if isinstance(output, tuple) and len(output) > 0:
#                 attention_output = output[0]
#                 print(f"📤 从output[0]获取attention输出: {attention_output.shape}")
#             elif hasattr(output, 'shape'):
#                 attention_output = output
#                 print(f"📤 直接使用output作为attention输出: {attention_output.shape}")
#             else:
#                 print(f"❌ 无法解析输出格式: {type(output)}")
#                 return output
            
#             # 检查形状是否匹配
#             print(f"🔍 比较形状:")
#             print(f"   输入: {hidden_states_input.shape}")
#             print(f"   输出: {attention_output.shape}")
            
#             if hidden_states_input.shape == attention_output.shape:
#                 # 计算差值
#                 input_cpu = hidden_states_input.detach().cpu()
#                 output_cpu = attention_output.detach().cpu()
#                 difference = output_cpu - input_cpu
                
#                 # 存储这个样本的数据
#                 sample_data = {
#                     'input_hidden_states': input_cpu,
#                     'attention_output': output_cpu,
#                     'difference': difference,
#                     'sample_id': captured_data['captured_count'],
#                     'layer_num': layer_num
#                 }
                
#                 captured_data['samples'].append(sample_data)
#                 captured_data['captured_count'] += 1
                
#                 print(f"✅ 样本 {captured_data['captured_count']}/{captured_data['max_samples']} 数据捕获成功！")
                
#                 # 如果收集够了，标记为完成
#                 if captured_data['captured_count'] >= captured_data['max_samples']:
#                     captured_data['captured'] = True
#                     print(f"🎯 已收集 {captured_data['max_samples']} 个样本，停止收集")
            
#             return output
            
#         except Exception as e:
#             print(f"❌ Forward方法执行出错: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return original_forward(*args, **kwargs)
    
#     return patched_forward

# def save_captured_data(save_path, layer_num, module_type):
#     """保存捕获的数据到JSON文件（不保存NPZ文件）"""
#     if not captured_data or not captured_data.get('samples', []):
#         print("⚠️ 没有捕获到有效数据")
#         return
    
#     # 处理多个样本的数据
#     samples = captured_data['samples']
#     if not samples:
#         print("⚠️ 样本列表为空")
#         return
    
#     print(f"📊 开始保存 {len(samples)} 个样本的统计信息...")
    
#     # 收集统计信息
#     all_diff_means = []
#     all_diff_stds = []
#     all_diff_norms = []
#     all_sample_shapes = []
    
#     # 收集所有样本的差值用于计算整体均值
#     all_differences = []
    
#     for i, sample in enumerate(samples):
#         input_tensor = sample['input_hidden_states']
#         output_tensor = sample['attention_output']
#         diff_tensor = sample['difference']
        
#         # 收集差值数据用于计算统计信息
#         all_differences.append(diff_tensor)
#         all_sample_shapes.append(list(input_tensor.shape))
        
#         # 统计信息
#         all_diff_means.append(float(torch.mean(diff_tensor)))
#         all_diff_stds.append(float(torch.std(diff_tensor)))
#         all_diff_norms.append(float(torch.norm(diff_tensor, p=2)))
    
#     # 计算所有样本差值的整体均值
#     concatenated_differences = torch.cat([diff.flatten() for diff in all_differences])
#     overall_difference_mean = float(torch.mean(concatenated_differences))
#     overall_difference_std = float(torch.std(concatenated_differences))
#     overall_difference_min = float(torch.min(concatenated_differences))
#     overall_difference_max = float(torch.max(concatenated_differences))
    
#     # 保存基本信息到JSON
#     info = {
#         'layer_num': layer_num,
#         'module_type': module_type,
#         'num_samples': len(samples),
#         'sample_shapes': all_sample_shapes,
#         'statistics': {
#             # 每个样本的差值均值
#             'per_sample_difference_means': all_diff_means,
#             'per_sample_difference_stds': all_diff_stds,
#             'per_sample_difference_l2_norms': all_diff_norms,
            
#             # 所有样本差值的整体统计
#             'overall_difference_mean': overall_difference_mean,
#             'overall_difference_std': overall_difference_std,
#             'overall_difference_min': overall_difference_min,
#             'overall_difference_max': overall_difference_max,
            
#             # 样本间的统计（基于每个样本的均值）
#             'sample_means_average': float(np.mean(all_diff_means)),
#             'sample_means_std': float(np.std(all_diff_means)),
#             'sample_l2norms_average': float(np.mean(all_diff_norms)),
#             'sample_l2norms_min': float(np.min(all_diff_norms)),
#             'sample_l2norms_max': float(np.max(all_diff_norms))
#         },
#         'shape_distribution': {
#             'unique_shapes': list(set(str(shape) for shape in all_sample_shapes)),
#             'shape_counts': {}
#         }
#     }
    
#     # 统计形状分布
#     from collections import Counter
#     shape_counter = Counter(str(list(s['input_hidden_states'].shape)) for s in samples)
#     info['shape_distribution']['shape_counts'] = dict(shape_counter)
    
#     with open(save_path, 'w') as f:
#         json.dump(info, f, indent=2)
    
#     print(f"✅ 信息已保存到: {save_path}")
#     print(f"📊 统计信息:")
#     print(f"   样本数量: {len(samples)}")
#     print(f"   不同形状数量: {len(info['shape_distribution']['unique_shapes'])}")
#     print(f"   🎯 所有样本差值的整体均值: {info['statistics']['overall_difference_mean']:.6f}")
#     print(f"   🎯 所有样本差值的整体标准差: {info['statistics']['overall_difference_std']:.6f}")
#     print(f"   每个样本差值均值的平均: {info['statistics']['sample_means_average']:.6f}")
#     print(f"   差值均值的标准差: {info['statistics']['sample_means_std']:.6f}")
#     print(f"   平均L2范数: {info['statistics']['sample_l2norms_average']:.6f}")
#     print(f"   L2范数范围: [{info['statistics']['sample_l2norms_min']:.6f}, {info['statistics']['sample_l2norms_max']:.6f}]")
    
#     # 显示形状分布
#     print(f"📏 形状分布:")
#     for shape_str, count in info['shape_distribution']['shape_counts'].items():
#         print(f"   {shape_str}: {count} 个样本")

# def patch_attention_module(model, layer_num, model_name):
#     """
#     直接修改attention模块的forward方法
#     """
#     global captured_data
    
#     try:
#         # 定位目标模块
#         print(f"🎯 寻找第{layer_num}层的self-attention模块...")
        
#         target_module = None
#         if 'llava' in model_name.lower() or 'intern' in model_name.lower():
#             target_module = model.model.language_model.model.layers[layer_num].self_attn
#             print(f"✅ 使用llava/intern路径找到模块")
#         elif 'Qwen' in model_name:
#             target_module = model.model.model.layers[layer_num].self_attn
#             print(f"✅ 使用Qwen路径找到模块")
#         elif 'idefics' in model_name.lower():
#             target_module = model.model.model.text_model.layers[layer_num].self_attn
#             print(f"✅ 使用idefics路径找到模块")
#         else:
#             target_module = model.model.language_model.model.layers[layer_num].self_attn
#             print(f"✅ 使用默认路径找到模块")
        
#         print(f"🎯 目标模块类型: {type(target_module).__name__}")
#         print(f"🎯 目标模块: {target_module}")
        
#         # 检查模块属性
#         print("🔍 检查模块属性:")
#         for attr_name in dir(target_module):
#             if not attr_name.startswith('_'):
#                 attr = getattr(target_module, attr_name)
#                 if callable(attr):
#                     print(f"   方法: {attr_name}")
#                 else:
#                     print(f"   属性: {attr_name}")
        
#         # 保存原始的forward方法
#         original_forward = target_module.forward
#         print(f"🔍 原始forward方法: {original_forward}")
        
#         # 用修改后的方法替换
#         patched_method = patch_attention_forward(original_forward, layer_num)
#         target_module.forward = patched_method.__get__(target_module, target_module.__class__)
        
#         print(f"🔧 已成功修改第{layer_num}层self-attention的forward方法")
#         print(f"🔍 新的forward方法: {target_module.forward}")
        
#         return (target_module, original_forward)
        
#     except Exception as e:
#         print(f"❌ 方法修改失败: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res
import torch

def modify_params(model, layer_num, module_type):
    # 获取对应层的参数路径
    layer_prefix = f"language_model.model.layers.{layer_num}"
    layer_prefix = f"model.text_model.layers.{layer_num}" if 'idefics' in model.__class__.__name__.lower() else layer_prefix
    # 定义模块名和对应的参数
    module_params = {
        'self_attn': [
            f"{layer_prefix}.self_attn.o_proj.weight",
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",

        ],
        # # 'self_attn': [
        # #     f"{layer_prefix}.attention.wo.weight",
        # #     f"{layer_prefix}.attention.wqkv.weight",

        # ],
        'mlp': [
            f"{layer_prefix}.mlp.gate_proj.weight",
            f"{layer_prefix}.mlp.up_proj.weight",
            f"{layer_prefix}.mlp.down_proj.weight"
        ],
        # 'self_attn': [
            
        #     f"{layer_prefix}.self_attn.v_proj.weight",
           
        # ],
        # 'mlp': [
        #     f"{layer_prefix}.mlp.down_proj.weight"
        # ]
    }
    # 根据qwen的结构来构建参数键值
    layer_prefix = f"model.layers.{layer_num}" if 'Qwen' in model.__class__.__name__ else layer_prefix
    if 'Qwen' in model.__class__.__name__:
        module_params = {
            'self_attn': [
            f"{layer_prefix}.self_attn.q_proj.bias",
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.bias",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.bias",
            f"{layer_prefix}.self_attn.v_proj.weight",
            f"{layer_prefix}.self_attn.o_proj.weight"
            ],
            'mlp': [
                f"{layer_prefix}.mlp.gate_proj.weight",
                f"{layer_prefix}.mlp.up_proj.weight",
                f"{layer_prefix}.mlp.down_proj.weight"
            ]
        }
    # 获取模块的参数
    params_to_modify = module_params.get(module_type)
    if params_to_modify is None:
        raise ValueError("Invalid module type. Choose 'self_attn' or 'mlp'.")
    
    # 遍历并修改参数
    for i, param_name in enumerate(params_to_modify, 1):
        if param_name in model.model.state_dict():
            param = model.model.state_dict()[param_name]
            
            # 修改参数
            # new_param = torch.full_like(param, param_mean)
            new_param = torch.zeros_like(param)
            model.model.state_dict()[param_name].copy_(new_param)
            
            print(f"   [✅] {param_name}: 参数已修改为:{new_param}")
            
        else:
            print(f"   [❌] {param_name}: 参数未找到")
    
    print(f"✅ 第{layer_num}层 {module_type} 参数修改完成\n")
    return model


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4,merge_model=None,Intervening_layer=None,Intervening_module=None):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'

    if Intervening_layer is not None and Intervening_module is not None:
        prev_file = f'{work_dir}/{model_name}_{dataset_name}_cut{Intervening_layer}_{Intervening_module}_PREV.pkl'
    else:
        prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'

    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    
    if merge_model: 
        if 'llava' in model_name or 'intern' in model_name.lower():
            device_map = {}
            print(model)
            print("-----------------------")
            for name, param in model.model.language_model.named_parameters():
                device_map[name] = param.device

            merged_state_dict = torch.load(merge_model, map_location='cpu')

            new_state_dict = {}
            for name, param in merged_state_dict.items():
                if name in device_map:
                    new_state_dict[name] = param.to(device_map[name])
                else:
                    print(f'Warning: Parameter {name} not found in original model.')

            model.model.language_model.load_state_dict(new_state_dict)

            del merged_state_dict
            torch.cuda.empty_cache()
        elif 'Qwen' in model_name:
            merged_state_dict = torch.load(merge_model)
            model.model.model.load_state_dict(merged_state_dict)
        elif 'idefics' in model_name:
            merged_state_dict = torch.load(merge_model)
            model.model.model.text_model.load_state_dict(merged_state_dict)
    

    # 🔑 修改：支持多层切断
    if Intervening_layer is not None and Intervening_module is not None:
        if isinstance(Intervening_layer, list):
            print(f"🔧 开始修改多层: {Intervening_layer} 的 {Intervening_module} 模块...")
            for layer in Intervening_layer:
                print(f"  - 修改第 {layer} 层...")
                model = modify_params(model, layer, Intervening_module)
        else:
            print(f"🔧 开始修改第 {Intervening_layer} 层的 {Intervening_module} 模块...")
            model = modify_params(model, Intervening_layer, Intervening_module)
        print(f"✅ 参数修改完成")

    is_api = getattr(model, 'is_api', False)
    if is_api:
        # API model handling
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    hook_data_captured = False

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])
        
        print(f"🔧 正在处理第{i+1}个样本 (索引: {idx})...")
        response = model.generate(message=struct, dataset=dataset_name)

        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
    
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model



def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, merge_model=None, Intervening_layer=None, Intervening_module=None):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    
    # 🔧 修复：根据 Intervening_layer 参数构建文件名
    if Intervening_layer is not None and Intervening_module is not None:
        if isinstance(Intervening_layer, list):
            layer_str = "_".join(map(str, Intervening_layer))
            result_file_base = f'{model_name}_{dataset_name}_cut{layer_str}_{Intervening_module}.xlsx'
            prev_file_base = f'{model_name}_{dataset_name}_cut{layer_str}_{Intervening_module}_PREV.pkl'
            template_base = f'{{}}{world_size}_{dataset_name}_cut{layer_str}_{Intervening_module}.pkl'
        else:
            result_file_base = f'{model_name}_{dataset_name}_cut{Intervening_layer}_{Intervening_module}.xlsx'
            prev_file_base = f'{model_name}_{dataset_name}_cut{Intervening_layer}_{Intervening_module}_PREV.pkl'
            template_base = f'{{}}{world_size}_{dataset_name}_cut{Intervening_layer}_{Intervening_module}.pkl'
    else:
        result_file_base = f'{model_name}_{dataset_name}.xlsx'
        prev_file_base = f'{model_name}_{dataset_name}_PREV.pkl'
        template_base = f'{{}}{world_size}_{dataset_name}.pkl'
    
    result_file = osp.join(work_dir, result_file_base)
    prev_file = osp.join(work_dir, prev_file_base)

    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    # 🔧 Intervening_layer
    tmpl = osp.join(work_dir, template_base)
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, merge_model=merge_model, 
        Intervening_layer=Intervening_layer, Intervening_module=Intervening_module)
    
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            temp_file = tmpl.format(i)
            if osp.exists(temp_file):
                data_all.update(load(temp_file))
            else:
                print(f"Warning: Temporary file {temp_file} not found")

        data = dataset.data
        
        # 检查是否所有数据都有预测结果
        missing_indices = []
        for x in data['index']:
            if x not in data_all:
                missing_indices.append(x)
        
        if missing_indices:
            print(f"Warning: Missing predictions for indices: {missing_indices[:10]}..." if len(missing_indices) > 10 else missing_indices)
        
        data['prediction'] = [str(data_all.get(x, 'MISSING_PREDICTION')) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        # 确保目录存在
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        dump(data, result_file)
        print(f"✅ Results saved to: {result_file}")
        
        # 清理临时文件
        for i in range(world_size):
            temp_file = tmpl.format(i)
            if osp.exists(temp_file):
                os.remove(temp_file)
    
    if world_size > 1:
        dist.barrier()
    
    return model
