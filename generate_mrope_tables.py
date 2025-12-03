import torch
import numpy as np
import argparse
import os

# Qwen2-VL standard section
# mrope_section = [16, 24, 24] 
# head_dim = 128

def get_mrope_cos_sin(section_idx, seq_len, mrope_section, head_dim):
    base_freq = [10000.0, 5000.0, 2500.0][section_idx]
    dim = mrope_section[section_idx]
    # The formula for inv_freq usually involves dim/2 or dim depending on implementation
    # Standard RoPE: 1.0 / (base ** (arange(0, dim, 2) / dim))
    inv_freq = 1.0 / (base_freq ** (torch.arange(0, dim, 2).float() / dim))
    
    position_ids = torch.arange(seq_len).float()
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.half().numpy(), sin.half().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_path", type=str, default="mrope_tables.npz")
    args = parser.parse_args()

    mrope_section = [16, 24, 24]
    head_dim = 128
    
    print(f"Generating mrope tables with max_seq_len={args.max_seq_len}...")
    
    mrope_tables = []
    for i in range(3):
        cos, sin = get_mrope_cos_sin(i, args.max_seq_len, mrope_section, head_dim)
        mrope_tables.append({"cos": cos, "sin": sin})
        print(f"Section {i}: cos shape {cos.shape}, sin shape {sin.shape}")

    np.savez_compressed(
        args.output_path,
        text_cos=mrope_tables[0]["cos"],
        text_sin=mrope_tables[0]["sin"],
        image_cos=mrope_tables[1]["cos"],
        image_sin=mrope_tables[1]["sin"],
        video_cos=mrope_tables[2]["cos"],
        video_sin=mrope_tables[2]["sin"],
        mrope_section=np.array(mrope_section)
    )
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
