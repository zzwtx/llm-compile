import inspect
import math
import os
import shutil
import sys
import tarfile
from time import time

import yaml
import argparse

# isort: off
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from pathlib import Path
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForVision2Seq, AutoProcessor,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          FuyuForCausalLM, FuyuProcessor,
                          LlavaForConditionalGeneration, NougatProcessor,
                          Pix2StructForConditionalGeneration,
                          VisionEncoderDecoderModel, CLIPVisionModel)
# isort: on
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPImageProcessor

def export_onnx(model,
                input,
                onnx_dir,
                onnx_name='model.onnx',
                input_names=['input'],
                output_names=['encoder_output'],
                dynamic_axes={'input': {
                    0: 'batch'
                }},
                logger=trt.Logger(trt.Logger.INFO)):
    logger.log(trt.Logger.INFO, f"Exporting onnx to {onnx_dir}/{onnx_name}")
    os.makedirs(onnx_dir, exist_ok=True)

    export_kwargs = dict(
        model=model,
        args=input,
        f=f'{onnx_dir}/{onnx_name}',
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,  # 默认就会把权重当初始值写进 graph
    )

    torch.onnx.export(**export_kwargs)


def compute_rotary_pos_emb(grid_thw, hf_config, VisionRotaryEmbedding):
    head_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads
    rotary_pos_emb_func = VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(grid_thw, rotary_pos_emb_func):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
                w // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
                w // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = rotary_pos_emb_func(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    rotary_pos_emb = rot_pos_emb(grid_thw, rotary_pos_emb_func)
    return rotary_pos_emb
    
def build_qwen2_vl_engine(args):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.configuration_qwen2_vl import \
        Qwen2VLVisionConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock,
        VisionAttention, VisionRotaryEmbedding)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager")
    hf_config = AutoConfig.from_pretrained(args.model_path)
    qwen2_vl_dim = hf_config.vision_config.in_chans * hf_config.vision_config.patch_size * hf_config.vision_config.patch_size * hf_config.vision_config.temporal_patch_size
    processor = AutoProcessor.from_pretrained(args.model_path)
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type":
                "image",
                "image":
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "Describe this picture?"
            },
        ],
    }]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    for i in range(len(image_inputs)):
        image_inputs[i] = image_inputs[i].resize(
            (image_inputs[i].size[0] // 2, image_inputs[i].size[1] // 2))
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs
    image = inputs['pixel_values'].to(torch.float16)
    image_grid_thw = inputs['image_grid_thw']
    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    seq_length = image.shape[0]
    attention_mask = torch.full([1, seq_length, seq_length],
                                torch.finfo(image.dtype).min,
                                device=image.device,
                                dtype=image.dtype)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                       cu_seqlens[i - 1]:cu_seqlens[i]] = 0
    rotary_pos_emb = compute_rotary_pos_emb(image_grid_thw, hf_config,
                                            VisionRotaryEmbedding)

    class VisionAttentionOpt(VisionAttention):

        def __init__(self, config: Qwen2VLVisionConfig):
            super().__init__(config)
            self.head_dim = config.embed_dim // config.num_heads

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                      self.num_heads,
                                                      -1).permute(1, 0, 2,
                                                                  3).unbind(0)

            def rotate_half(x):
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)

            def apply_rotary_pos_emb_vision(
                    tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                orig_dtype = tensor.dtype
                tensor = tensor.float()
                cos = freqs.cos()
                sin = freqs.sin()
                cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
                sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
                output = (tensor * cos) + (rotate_half(tensor) * sin)
                output = output.to(orig_dtype)
                return output

            q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
                self.head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

    class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = VisionAttentionOpt(config)

        def forward(self, hidden_states, attention_mask,
                    rotary_pos_emb) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    class Qwen2VisionTransformerPretrainedModelOpt(
            Qwen2VisionTransformerPretrainedModel):

        def __init__(self, config) -> None:
            super().__init__(config)
            self.blocks = nn.ModuleList([
                Qwen2VLVisionBlockOpt(config, config._attn_implementation)
                for _ in range(config.depth)
            ])

        def forward(self, hidden_states: torch.Tensor,
                    rotary_pos_emb: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
            hidden_states = self.patch_embed(hidden_states)
            for blk in self.blocks:
                hidden_states = blk(hidden_states,
                                    attention_mask=attention_mask,
                                    rotary_pos_emb=rotary_pos_emb)
            res = self.merger(hidden_states)
            return res

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.visual = Qwen2VisionTransformerPretrainedModelOpt._from_config(
                model.config.vision_config,
                torch_dtype=torch.float32,
            )
            self.visual.load_state_dict(model.visual.state_dict())

        def forward(self, images, rotary_pos_emb, attention_mask):
            img_features = self.visual(images, rotary_pos_emb, attention_mask)
            return img_features

    wrapper = VisionEncoderWrapper(model)
    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        }
    }
    export_onnx(wrapper, (image, rotary_pos_emb, attention_mask),
                f'{args.output_dir}/onnx',
                input_names=['input', 'rotary_pos_emb', 'attention_mask'],
                output_names=['encoder_output'],
                dynamic_axes=dynamic_axes)

def add_multimodal_arguments(parser):
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        choices=[
                            'blip2', 'llava', 'llava_next', 'llava_onevision',
                            'llava_onevision_lmms', 'vila', 'nougat', 'cogvlm',
                            'fuyu', 'pix2struct', 'neva', 'kosmos-2',
                            'video-neva', 'phi-3-vision', 'phi-4-multimodal',
                            'mllama', 'internvl', 'qwen2_vl',
                            'internlm-xcomposer2', 'qwen2_audio', 'pixtral'
                        ],
                        help="Model type")
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help=
        "Huggingface repo, local directory with weights or path to checkpoint file"
    )
    parser.add_argument('--vila_path',
                        type=str,
                        default=None,
                        help="Path to VILA source code directory")
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help="Directory where visual TRT engines are saved")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=4,
                        help="Maximum batch size for input images")
    parser.add_argument(
        '--max_hw_dims',
        type=int,
        default=5184,
        help=
        "Maximum multiply of h and w after patching for input images for qwen2_vl"
    )
    parser.add_argument(
        '--min_hw_dims',
        type=int,
        default=128,
        help=
        "Minimum multiply of h and w after patching for input images for qwen2_vl"
    )
    parser.add_argument(
        '--num_mul_bins',
        type=int,
        default=128,
        help="Number of Mel frequency bins of input audios for qwen2_audio")
    parser.add_argument(
        '--max_mel_seq_len',
        type=int,
        default=3000,
        help=
        "Maximum Mel frequency feature lengths of input audios for qwen2_audio")
    return parser

class MultimodalEngineBuilder:

    def __init__(self, args):
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        if args.output_dir is None:
            # default path to save the engines
            model_name = args.model_path.split('/')[-1]
            args.output_dir = f'tmp/trt_engines/{model_name}/multimodal_encoder'

        os.makedirs(args.output_dir, exist_ok=True)

        self.args = args

    def build(self):
        args = self.args
        if args.model_type == 'qwen2_vl':
            build_qwen2_vl_engine(args)
        else:
            raise RuntimeError(f"Invalid model type {args.model_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_multimodal_arguments(parser)
    args = parser.parse_args()

    builder = MultimodalEngineBuilder(args)
    builder.build()