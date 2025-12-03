# scripts/convert_checkpoint.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 将补丁目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "tensorrt_llm_patch"))

from qwen2vl_model import Qwen2VLForConditionalGeneration
from tensorrt_llm.quantization import QuantMode

def main():
    # 配置路径
    hf_model_dir = "/home/xsf/fl/edge/models/Qwen2-VL-2B-Instruct"  # HF 模型路径
    mrope_table_path = "/home/xsf/fl/edge/mrope_tables.npz"  # 预计算表
    output_dir = "fl/edge/models/trt_0120_engines/llm"
    
    # 1. 创建模型实例（使用修改后的类）
    model = Qwen2VLForConditionalGeneration.from_huggingface(
        hf_model_dir,
        mrope_table_path
    )
    
    # 2. 应用量化（Orin 优化）
    quant_mode = QuantMode.use_int8_sq()  # 平滑量化
    model.quantize(quant_mode)
    
    # 3. 保存为 TRT-LLM 格式
    model.save(output_dir)
    
    # 4. 生成配置文件
    config = {
        "architecture": "Qwen2VLForConditionalGeneration",
        "dtype": "float16",
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": 512,  # Orin 优化
        "mrope_section": [16, 24, 24],
        "quantization": "int8_sq"
    }
    
    import json
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"模型转换完成！保存至: {output_dir}")

if __name__ == "__main__":
    main()