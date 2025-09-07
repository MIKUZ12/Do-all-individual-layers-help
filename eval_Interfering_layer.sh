#!/bin/bash

# --- 配置 VLMEvalKit 和缓存 ---
export VLMEVALKIT_DIR="Your_VLMEvalKit_path"
export HF_HOME="Your_hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_KERNEL_CACHE_PATH="Your_cache_path"

# --- GPU 配置 ---
export CUDA_VISIBLE_DEVICES='0'
export OPENAI_API_KEY="Your openai_key"

# --- 路径和模型定义 ---
BASE_MODEL_NAME_OR_PATH="Your_model_path"
EVAL_OUTPUT_DIR="Your_output_path"

# --- 优化的文件结构定义 ---
# 主要结果目录
RESULTS_BASE_DIR="${EVAL_OUTPUT_DIR}/Intervening_layer_experiments"
RESULTS_DIR="${RESULTS_BASE_DIR}/results"          # 存储CSV结果文件
LOGS_DIR="${RESULTS_BASE_DIR}/Intervening_layer_logs"      # 存储所有日志文件
TEMP_DIR="${RESULTS_BASE_DIR}/temp"                # 临时文件目录
ANALYSIS_DIR="${RESULTS_BASE_DIR}/analysis"        # 分析结果目录
CACHE_DIR="${RESULTS_BASE_DIR}/cache"              # 缓存文件目录

# 创建所有必要的目录
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${TEMP_DIR}"
mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${CACHE_DIR}"

# 为当前实验创建时间戳目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_LOG_DIR="${LOGS_DIR}/${TIMESTAMP}"
mkdir -p "${CURRENT_LOG_DIR}"

echo "📁 文件结构初始化完成:"
echo "   结果文件: ${RESULTS_DIR}"
echo "   日志文件: ${CURRENT_LOG_DIR}"
echo "   临时文件: ${TEMP_DIR}"
echo "   分析结果: ${ANALYSIS_DIR}"
echo "   缓存文件: ${CACHE_DIR}"

cleanup_all_cache() {
    echo "🧹 清理所有可能的缓存..."
    
    # 清理结果文件（可选，根据需要）
    # rm -f "${RESULTS_DIR}"/*.csv
    
    # 清理临时文件和缓存
    rm -rf "${TEMP_DIR}"/*
    rm -rf "${CACHE_DIR}"/*
    
    # 重新创建目录
    mkdir -p "${TEMP_DIR}"
    mkdir -p "${CACHE_DIR}"
    
    echo "✅ 缓存清理完成"
}

# --- 层和模块配置 ---
LAYERS=($(seq 1 31))
MODULES=("self_attn")
tasks=(
    "MMStar"
    "MMBench_DEV_EN"
    "MMMU_DEV_VAL"
)

# --- GPU 数量 ---
IFS=',' read -ra GPUS_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${#GPUS_ARRAY[@]}
echo "Using ${NPROC_PER_NODE} GPU(s) for evaluation"

# --- 切换到 VLMEvalKit 项目目录 ---
cd "${VLMEVALKIT_DIR}" || { echo "错误：无法切换到 VLMEvalKit 目录: ${VLMEVALKIT_DIR}"; exit 1; }

echo "开始使用 VLMEvalKit 评估融合模型（遍历所有层）..."

# --- 函数：查找可用端口 ---
find_free_port() {
    local base_port=54321
    local port=$((base_port + RANDOM % 1000))
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
        if [ $port -gt 65535 ]; then
            port=$((base_port + RANDOM % 1000))
        fi
    done
    echo $port
}

# --- 函数：检查结果文件是否存在 ---
check_result_exists() {
    local weight="$1"
    local layer="$2"
    local module="$3"
    local dataset="$4"
    
    local expected_result="${RESULTS_DIR}/llava_${weight}_eval_${layer}_${module}_${dataset}.csv"
    
    if [ -f "${expected_result}" ]; then
        return 0  # 存在
    else
        return 1  # 不存在
    fi
}

# 在开始前清理缓存
cleanup_all_cache

# --- 创建实验配置文件 ---
cat > "${CURRENT_LOG_DIR}/experiment_config.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "base_model": "${BASE_MODEL_NAME_OR_PATH}",
    "weights_value": "${weights}",
    "layers": [$(IFS=','; echo "${LAYERS[*]}")],
    "modules": ["$(IFS='","'; echo "${MODULES[*]}")"],
    "tasks": ["$(IFS='","'; echo "${tasks[*]}")"],
    "total_experiments": $((${#LAYERS[@]} * ${#MODULES[@]} * ${#tasks[@]})),
    "gpu_config": "${CUDA_VISIBLE_DEVICES}",
    "nproc_per_node": ${NPROC_PER_NODE}
}
EOF

# --- 主要评估循环 ---
total_experiments=$((${#LAYERS[@]} * ${#MODULES[@]} * ${#tasks[@]}))
current_experiment=0
success_count=0
skip_count=0
fail_count=0

echo ""
echo "🔬 Intervening_layer实验配置"
echo "==============================================="
echo "基础模型: ${BASE_MODEL_NAME_OR_PATH}"
echo "权重参数: ${weights}"
echo "总实验数: ${total_experiments}"
echo "层数范围: 0-31 (共${#LAYERS[@]}层)"
echo "测试模块: ${MODULES[*]}"
echo "测试数据集: ${tasks[*]}"
echo "==============================================="
echo "📁 文件存储结构:"
echo "   结果文件: ${RESULTS_DIR}"
echo "   日志文件: ${CURRENT_LOG_DIR}"
echo "   分析结果: ${ANALYSIS_DIR}"
echo "==============================================="

# 记录开始时间
start_time=$(date +%s)

# 创建日志文件
main_log="${CURRENT_LOG_DIR}/main_experiment.log"
failed_log="${CURRENT_LOG_DIR}/failed_experiments.log"
success_log="${CURRENT_LOG_DIR}/successful_experiments.log"
progress_log="${CURRENT_LOG_DIR}/progress.log"

echo "# Cut Layer实验主日志 - $(date)" > "${main_log}"
echo "# 失败实验记录 - $(date)" > "${failed_log}"
echo "# 成功实验记录 - $(date)" > "${success_log}"
echo "# 进度记录 - $(date)" > "${progress_log}"

# 遍历所有层
for layer in "${LAYERS[@]}"; do
    for module in "${MODULES[@]}"; do
        for task in "${tasks[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            echo ""
            echo "🔬 实验 ${current_experiment}/${total_experiments}"
            echo "==============================================="
            echo "🔹 层: ${layer}"
            echo "🔹 模块: ${module}" 
            echo "🔹 任务: ${task}"
            echo "🔹 时间: $(date '+%Y-%m-%d %H:%M:%S')"
            
            # 记录到主日志
            echo "$(date '+%Y-%m-%d %H:%M:%S') - 实验 ${current_experiment}/${total_experiments}: 层${layer}, 模块${module}, 任务${task}" >> "${main_log}"
            
            # 检查结果是否已存在
            if check_result_exists "${weights}" "${layer}" "${module}" "${task}"; then
                echo "⏭️  结果已存在，跳过实验"
                skip_count=$((skip_count + 1))
                echo "$(date '+%Y-%m-%d %H:%M:%S') - 跳过: 层${layer}, 模块${module}, 任务${task} - 结果已存在" >> "${main_log}"
                
                # 显示进度
                if [ $current_experiment -gt 0 ]; then
                    progress=$((current_experiment * 100 / total_experiments))
                    elapsed_time=$(($(date +%s) - start_time))
                    estimated_total_time=$((elapsed_time * total_experiments / current_experiment))
                    remaining_time=$((estimated_total_time - elapsed_time))
                    
                    echo "📊 进度: ${progress}% | 成功: ${success_count} | 跳过: ${skip_count} | 失败: ${fail_count}"
                    echo "⏱️  已用时: $(date -d@${elapsed_time} -u +%H:%M:%S) | 预计剩余: $(date -d@${remaining_time} -u +%H:%M:%S)"
                    
                    # 记录进度
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - 进度: ${progress}% (${current_experiment}/${total_experiments}) | 成功: ${success_count} | 跳过: ${skip_count} | 失败: ${fail_count}" >> "${progress_log}"
                fi
                continue
            fi
            
            # 查找可用端口
            port=$(find_free_port)
            echo "🌐 使用端口: ${port}"
            
            # 为当前实验创建临时工作目录
            experiment_temp_dir="${TEMP_DIR}/exp_${layer}_${module}_${task}_${current_experiment}"
            mkdir -p "${experiment_temp_dir}"
            
            echo "🚀 开始执行实验..."
            echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始执行: 层${layer}, 模块${module}, 任务${task}" >> "${main_log}"
            
            # 执行评估，将输出保存到专门的日志文件
            experiment_log="${CURRENT_LOG_DIR}/experiment_${layer}_${module}_${task}.log"
            
            torchrun --nproc-per-node=${NPROC_PER_NODE} --master-port=${port} run.py \
                --data "$task" \
                --model "${BASE_MODEL_NAME_OR_PATH}" \
                --verbose \
                --work-dir "${experiment_temp_dir}" \
                --cut_layer "${layer}" \
                --cut_module "${module}" \
                > "${experiment_log}" 2>&1
            
            eval_exit_code=$?
            
            # 检查执行结果
            if [ $eval_exit_code -ne 0 ]; then
                echo "❌ 实验失败"
                echo "   错误代码: ${eval_exit_code}"
                echo "   详细日志: ${experiment_log}"
                
                # 记录失败
                fail_info="$(date '+%Y-%m-%d %H:%M:%S') - 层${layer}, 模块${module}, 任务${task}, 错误代码${eval_exit_code}"
                echo "${fail_info}" >> "${failed_log}"
                echo "${fail_info}" >> "${main_log}"
                fail_count=$((fail_count + 1))
            else
                echo "✅ 实验执行完成"
                
                # 验证结果文件是否生成
                if check_result_exists "${weights}" "${layer}" "${module}" "${task}"; then
                    echo "✅ 结果文件确认生成"
                    success_count=$((success_count + 1))
                    success_info="$(date '+%Y-%m-%d %H:%M:%S') - 层${layer}, 模块${module}, 任务${task} - 成功"
                    echo "${success_info}" >> "${success_log}"
                    echo "${success_info}" >> "${main_log}"
                else
                    echo "⚠️  警告：实验完成但未找到预期的结果文件"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - 警告: 层${layer}, 模块${module}, 任务${task} - 结果文件未找到" >> "${main_log}"
                fi
            fi
            
            # 清理临时目录
            if [ -d "${experiment_temp_dir}" ]; then
                rm -rf "${experiment_temp_dir}"
            fi
            
            # 计算并显示进度
            progress=$((current_experiment * 100 / total_experiments))
            elapsed_time=$(($(date +%s) - start_time))
            
            if [ $current_experiment -gt 0 ]; then
                estimated_total_time=$((elapsed_time * total_experiments / current_experiment))
                remaining_time=$((estimated_total_time - elapsed_time))
                
                echo "📊 进度: ${progress}% (${current_experiment}/${total_experiments})"
                echo "📈 成功: ${success_count} | 跳过: ${skip_count} | 失败: ${fail_count}"
                echo "⏱️  已用时间: $(date -d@${elapsed_time} -u +%H:%M:%S)"
                echo "⏱️  预计剩余: $(date -d@${remaining_time} -u +%H:%M:%S)"
                echo "⏱️  预计完成: $(date -d@$(($(date +%s) + remaining_time)))"
                
                # 记录进度
                echo "$(date '+%Y-%m-%d %H:%M:%S') - 进度: ${progress}% (${current_experiment}/${total_experiments}) | 成功: ${success_count} | 跳过: ${skip_count} | 失败: ${fail_count} | 剩余时间: $(date -d@${remaining_time} -u +%H:%M:%S)" >> "${progress_log}"
            fi
            
            # 短暂等待，确保资源释放
            echo "⏳ 等待资源释放..."
            sleep 5
        done
    done
done

# --- 实验完成统计 ---
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "🎉 所有Cut Layer实验已完成！"
echo "==============================================="
echo "📊 实验统计："
echo "   总实验数: ${total_experiments}"
echo "   成功实验: ${success_count}"
echo "   跳过实验: ${skip_count}"
echo "   失败实验: ${fail_count}"
echo "   成功率: $(( (success_count + skip_count) * 100 / total_experiments ))%"
echo ""
echo "⏱️  时间统计："
echo "   总用时: $(date -d@${total_time} -u +%H:%M:%S)"
echo "   平均每个实验: $(( total_time / total_experiments ))秒"
echo "   开始时间: $(date -d@${start_time})"
echo "   结束时间: $(date -d@${end_time})"
echo ""

# 记录最终统计
final_stats="${CURRENT_LOG_DIR}/final_statistics.json"
cat > "${final_stats}" << EOF
{
    "experiment_completed": "$(date -Iseconds)",
    "total_experiments": ${total_experiments},
    "successful_experiments": ${success_count},
    "skipped_experiments": ${skip_count},
    "failed_experiments": ${fail_count},
    "success_rate": $(( (success_count + skip_count) * 100 / total_experiments )),
    "total_duration_seconds": ${total_time},
    "total_duration_formatted": "$(date -d@${total_time} -u +%H:%M:%S)",
    "average_time_per_experiment": $(( total_time / total_experiments )),
    "start_time": "$(date -d@${start_time} -Iseconds)",
    "end_time": "$(date -d@${end_time} -Iseconds)"
}
EOF

echo "📁 文件位置："
echo "   结果文件: ${RESULTS_DIR}"
echo "   主日志文件: ${main_log}"
echo "   详细日志目录: ${CURRENT_LOG_DIR}"
echo "   最终统计: ${final_stats}"

if [ $fail_count -gt 0 ]; then
    echo "   失败记录: ${failed_log}"
fi

echo "==============================================="

# --- 自动运行结果分析 ---
echo ""
echo "🔍 开始分析实验结果..."

if command -v python3 &> /dev/null; then
    python3 "${ANALYSIS_DIR}/analyze_cut_layer_results.py" "${RESULTS_DIR}" "${ANALYSIS_DIR}"
    analysis_exit_code=$?
    
    if [ $analysis_exit_code -eq 0 ]; then
        echo "✅ 结果分析完成"
        echo "📊 分析结果保存在: ${ANALYSIS_DIR}"
    else
        echo "⚠️  结果分析过程中出现错误，退出代码: ${analysis_exit_code}"
    fi
else
    echo "⚠️  Python3 未找到，跳过自动分析"
    echo "   您可以手动运行: python3 ${ANALYSIS_DIR}/analyze_cut_layer_results.py ${RESULTS_DIR} ${ANALYSIS_DIR}"
fi

# --- 显示最终结果概览 ---
echo ""
echo "📋 生成的文件概览："

# 显示结果文件
result_files_count=$(ls "${RESULTS_DIR}"/llava_*_eval_*.csv 2>/dev/null | wc -l)
if [ $result_files_count -gt 0 ]; then
    echo "   📊 结果文件 (${result_files_count} 个):"
    ls "${RESULTS_DIR}"/llava_*_eval_*.csv 2>/dev/null | head -5 | sed 's|.*/|     - |'
    if [ $result_files_count -gt 5 ]; then
        echo "     - ... 还有 $((result_files_count - 5)) 个文件"
    fi
else
    echo "   ⚠️  未找到结果文件"
fi

# 显示日志文件
log_files_count=$(ls "${CURRENT_LOG_DIR}"/*.log 2>/dev/null | wc -l)
if [ $log_files_count -gt 0 ]; then
    echo "   📝 日志文件 (${log_files_count} 个):"
    echo "     - 主日志: $(basename "${main_log}")"
    echo "     - 失败记录: $(basename "${failed_log}")"
    echo "     - 成功记录: $(basename "${success_log}")"
    echo "     - 进度记录: $(basename "${progress_log}")"
    if [ $log_files_count -gt 4 ]; then
        echo "     - ... 还有 $((log_files_count - 4)) 个实验日志"
    fi
fi

# 显示分析文件
analysis_files_count=$(ls "${ANALYSIS_DIR}"/* 2>/dev/null | wc -l)
if [ $analysis_files_count -gt 0 ]; then
    echo "   📈 分析文件 (${analysis_files_count} 个):"
    ls "${ANALYSIS_DIR}"/* 2>/dev/null | head -5 | sed 's|.*/|     - |'
    if [ $analysis_files_count -gt 5 ]; then
        echo "     - ... 还有 $((analysis_files_count - 5)) 个文件"
    fi
fi

echo ""
echo "🎊 Cut Layer实验全部完成！"
echo ""
echo "📖 快速查看结果:"
echo "   汇总数据: cat ${ANALYSIS_DIR}/cut_layer_summary.csv"
echo "   分析报告: cat ${ANALYSIS_DIR}/cut_layer_analysis_report.md"
echo "   实验统计: cat ${final_stats}"
echo ""
echo "📁 完整文件结构:"
echo "   ${RESULTS_BASE_DIR}/"
echo "   ├── results/          # CSV结果文件"
echo "   ├── cut_layer_logs/"  # 所有日志文件"
echo "   │   └── ${TIMESTAMP}/ # 本次实验的日志"
echo "   ├── analysis/         # 分析结果和报告"
echo "   ├── temp/             # 临时文件(已清理)"
echo "   └── cache/            # 缓存文件(已清理)"