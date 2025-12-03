
import os
import json
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def rename_weights(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    # 1. 也是最关键的一步：定义映射规则
    # 左边是你报错日志里 Provided (现有的)
    # 右边是 TRT-LLM 0.12.0 Qwen/Llama (期望的)
    key_mapping = {
        # Attention 部分
        "attention.dense.weight": "attention.o_proj.weight",
        
        # MLP 部分 (SwiGLU)
        # 注意：Qwen 的 gate/fc 和 Llama 的 gate/up 对应关系很关键
        # 通常: gate (带激活) -> gate_proj
        #       fc (不带激活) -> up_proj
        #       proj (输出)   -> down_proj
        "mlp.gate.weight": "mlp.gate_proj.weight",
        "mlp.fc.weight":   "mlp.up_proj.weight", 
        "mlp.proj.weight": "mlp.down_proj.weight",
        
        # QKV 部分
        # 如果 0.12 期望 fused qkv，且你也提供了 qkv，通常名字叫 attention.qkv.weight 即可
        # 如果报错提示找不到 q_proj，则需要拆分 (暂时假设 0.12 能处理 qkv)
        # "attention.qkv.weight": "attention.qkv.weight" # 名字通常一致，不用改
    }

    # 读取索引文件 (如果有)
    index_file = os.path.join(src_dir, "model.safetensors.index.json")
    has_index = os.path.exists(index_file)
    
    files_to_process = []
    if has_index:
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        # 复制并修改 index 数据
        new_weight_map = {}
        for old_key, filename in index_data["weight_map"].items():
            new_key = old_key
            for src_k, dst_k in key_mapping.items():
                if src_k in old_key:
                    new_key = old_key.replace(src_k, dst_k)
                    break
            new_weight_map[new_key] = filename
            if filename not in files_to_process:
                files_to_process.append(filename)
        
        index_data["weight_map"] = new_weight_map
        with open(os.path.join(dst_dir, "model.safetensors.index.json"), 'w') as f:
            json.dump(index_data, f, indent=2)
    else:
        # 如果没有 index 文件，扫描所有 safetensors
        files_to_process = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]

    print(f"检测到 {len(files_to_process)} 个权重文件，开始转换...")

    for filename in tqdm(files_to_process):
        file_path = os.path.join(src_dir, filename)
        tensors = load_file(file_path)
        new_tensors = {}
        
        for k, v in tensors.items():
            new_key = k
            for src_pattern, dst_pattern in key_mapping.items():
                # 使用字符串替换来重命名层
                if src_pattern in k:
                    new_key = k.replace(src_pattern, dst_pattern)
                    # print(f"映射: {k} -> {new_key}")
                    break
            new_tensors[new_key] = v
        
        # 保存到新目录
        save_file(new_tensors, os.path.join(dst_dir, filename))
    
    # 复制 config.json (非常重要，TRT-LLM 需要读取架构参数)
    os.system(f"cp {os.path.join(src_dir, 'config.json')} {dst_dir}")
    print("转换完成！")

if __name__ == "__main__":
    # 修改这里的路径
    SRC_DIR = "/home/xsf/fl/edge/models/tllm_ckpts/Qwen2-VL-2B-Instruct/fp16"
    DST_DIR = "/home/xsf/fl/edge/models/tllm_ckpts/Qwen2-VL-2B-Instruct/fp16_renamed"
    
    rename_weights(SRC_DIR, DST_DIR)
    