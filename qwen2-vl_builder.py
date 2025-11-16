#!/usr/bin/env python3
"""
Tiny standalone qwen2_vl engine builder.

This file contains only the code paths required to export the vision encoder
of QWen2-VL to ONNX and build a TensorRT engine. It was adapted from the
original `tensorrt_llm.tools.multimodal_builder` implementation and
keeps only the pieces needed for qwen2_vl.

Usage example:
  python3 qwen2-vl_builder.py --model_path /path/to/Qwen2-VL --output_dir /path/to/out --max_batch_size 1

Notes:
  - This script assumes the tensorrt-llm package and its Builder/Session
	utilities are available in the current Python environment.
  - It intentionally omits builders for other models.
"""

import math
import os
import shutil
import sys
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from pathlib import Path
from PIL import Image

from tensorrt_llm._utils import torch_dtype_to_str, to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session

from transformers import AutoConfig, AutoProcessor


def add_multimodal_arguments(parser):
	"""
	向 argparse.ArgumentParser 添加多模态模型构建相关的命令行参数。

	Args:
		parser (argparse.ArgumentParser): 要添加参数的解析器对象。

	Returns:
		argparse.ArgumentParser: 添加了新参数的解析器对象。
	"""
	parser.add_argument('--model_type', type=str, default='qwen2_vl')
	parser.add_argument('--model_path', type=str, default=None)
	parser.add_argument('--onnx_path', type=str, default=None)
	parser.add_argument('--output_dir', type=str, default=None)
	parser.add_argument('--max_batch_size', type=int, default=4)
	parser.add_argument('--max_hw_dims', type=int, default=5184)
	parser.add_argument('--min_hw_dims', type=int, default=128)
	return parser


def export_onnx(model,
				input,
				onnx_dir,
				onnx_name='model.onnx',
				input_names=['input'],
				output_names=['encoder_output'],
				dynamic_axes={'input': {0: 'batch'}},
				logger=trt.Logger(trt.Logger.INFO)):
	"""
	将 PyTorch 模型导出为 ONNX 格式。

	Args:
		model (torch.nn.Module): 要导出的 PyTorch 模型。
		input (tuple): 模型的示例输入，用于追踪计算图。
		onnx_dir (str): 存放 ONNX 文件的目录。
		onnx_name (str): ONNX 文件的名称。
		input_names (list): ONNX 模型输入节点的名称。
		output_names (list): ONNX 模型输出节点的名称。
		dynamic_axes (dict): 指定输入/输出张量的动态维度。
		logger (trt.Logger): TensorRT 日志记录器。
	"""
	logger.log(trt.Logger.INFO, f"Exporting onnx to {onnx_dir}/{onnx_name}")
	os.makedirs(onnx_dir, exist_ok=True)

	torch.onnx.export(model,
					  input,
					  f'{onnx_dir}/{onnx_name}',
					  opset_version=17,
					  input_names=input_names,
					  output_names=output_names,
					  dynamic_axes=dynamic_axes)


def build_trt_engine(model_type,
					 input_sizes,
					 onnx_dir,
					 engine_dir,
					 max_batch_size,
					 dtype=torch.float16,
					 model_params=None,
					 onnx_name='model.onnx',
					 engine_name='model.engine',
					 delete_onnx=False,
					 logger=trt.Logger(trt.Logger.INFO)):
	"""
	从 ONNX 文件构建 TensorRT 引擎。

	这个函数是构建流程的核心，它执行以下步骤：
	1. 初始化 TensorRT Builder、Network Definition 和 Optimization Profile。
	2. 使用 tensorrt_llm 的辅助类创建 Builder Config，配置精度等参数。
	3. 解析 ONNX 文件，将其中的计算图加载到 Network Definition 中。
	4. 针对 qwen2_vl 模型的特殊输入（图像、旋转位置编码、注意力掩码），
	   设置详细的动态尺寸范围（Min, Opt, Max），以支持不同分辨率的图像输入。
	5. 将优化配置文件应用到构建配置中。
	6. 调用 Builder 的 build_serialized_network 方法，执行构建过程，生成序列化的引擎。
	7. 将序列化引擎保存为 .engine 文件，并将构建配置（如输出形状）保存为 config.json。

	Args:
		model_type (str): 模型类型，此脚本中应为 'qwen2_vl'。
		input_sizes (list): 描述模型部分输入的尺寸信息（例如 rotary_pos_emb 的维度）。
		onnx_dir (str): ONNX 文件所在的目录。
		engine_dir (str): 引擎文件和配置文件要保存的目录。
		max_batch_size (int): 引擎支持的最大批量大小。
		dtype (torch.dtype): 引擎的计算精度，默认为 float16。
		model_params (dict, optional): 包含模型特定参数的字典，如 'qwen2_vl_dim'、'min_hw_dims'、'max_hw_dims'。
		onnx_name (str): ONNX 文件的名称。
		engine_name (str): 输出引擎文件的名称。
		delete_onnx (bool): 构建成功后是否删除 ONNX 目录。
		logger (trt.Logger): TensorRT 日志记录器。
	"""
	model_params = model_params or {}
	onnx_file = f'{onnx_dir}/{onnx_name}'
	engine_file = f'{engine_dir}/{engine_name}'
	config_file = f'{engine_dir}/config.json'
	logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_file}")

	# 1. 初始化 TensorRT 构建器、网络定义和优化配置文件
	builder = trt.Builder(logger)
	# EXPLICIT_BATCH 标志是现代 TensorRT 网络定义的标准做法
	network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
	profile = builder.create_optimization_profile()

	# 2. 使用 tensorrt_llm 的辅助类创建构建配置
	config_args = {
		"precision": torch_dtype_to_str(dtype),
		"model_type": model_type,
		"strongly_typed": False,
		"max_batch_size": max_batch_size,
		"model_name": "qwen2_vl"
	}

	if "num_frames" in model_params:
		config_args["num_frames"] = model_params["num_frames"]

	# Builder() 是 tensorrt_llm 中的辅助类，不是 trt.Builder
	config_wrapper = Builder().create_builder_config(**config_args)
	config = config_wrapper.trt_builder_config

	# 3. 创建 ONNX 解析器并解析模型文件
	parser = trt.OnnxParser(network, logger)
	with open(onnx_file, 'rb') as model:
		if not parser.parse(model.read(), os.path.abspath(onnx_file)):
			logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
			for error in range(parser.num_errors):
				logger.log(trt.Logger.ERROR, parser.get_error(error))
		logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

	# 定义批处理大小的范围
	nBS = -1
	nMinBS = 1
	nOptBS = max(nMinBS, int(max_batch_size / 2))
	nMaxBS = max_batch_size

	assert isinstance(input_sizes, list), "input_sizes must be a list"
	# 4. 针对 qwen2_vl 模型设置动态尺寸
	if model_type == "qwen2_vl":
		# 从网络定义中按索引获取输入张量
		input_images = network.get_input(0)
		inputT = network.get_input(1)  # rotary_pos_emb
		attenstion_mask = network.get_input(2)

		# 获取模型特定的维度信息
		qwen2_vl_dim = model_params.get('qwen2_vl_dim', 0)
		min_hw_dims = model_params.get('min_hw_dims', 0)
		max_hw_dims = model_params.get('max_hw_dims', 0)

		assert min_hw_dims > 0, "min_hw_dims must be positive for qwen2_vl"
		assert max_hw_dims > 0, "max_hw_dims must be positive for qwen2_vl"

		# 为动态维度 'hw' 设置最小、最优和最大尺寸
		# 'hw' 维度代表图像经过 patch embedding 后的 token 数量 (height * width)
		multi_size_min = min_hw_dims
		multi_size_max = max_hw_dims * max_batch_size
		multi_size_opt = max(multi_size_min, int(multi_size_max / 2))

		# 为 rotary_pos_emb 输入设置形状和动态范围
		inputT.shape = [-1, *input_sizes]
		profile.set_shape(inputT.name, [multi_size_min, *input_sizes], [multi_size_opt, *input_sizes], [multi_size_max, *input_sizes])

		# 为图像输入设置形状和动态范围
		input_images.shape = [-1, qwen2_vl_dim]
		profile.set_shape(input_images.name, [multi_size_min, qwen2_vl_dim], [multi_size_opt, qwen2_vl_dim], [multi_size_max, qwen2_vl_dim])

		# 为注意力掩码输入设置形状和动态范围
		attenstion_mask.shape = [1, -1, -1]
		profile.set_shape(attenstion_mask.name, [1, multi_size_min, multi_size_min], [1, multi_size_opt, multi_size_opt], [1, multi_size_max, multi_size_max])
	else:
		raise ValueError("Unsupported model_type for this builder")

	# 5. 将优化配置文件添加到构建配置中
	config.add_optimization_profile(profile)

	t0 = time()
	# 6. 执行构建，生成序列化的引擎
	engine_string = builder.build_serialized_network(network, config)
	t1 = time()
	if engine_string is None:
		raise RuntimeError("Failed building %s" % (engine_file))
	else:
		# 7. 保存引擎和配置
		logger.log(trt.Logger.INFO, "Succeeded building %s in %d s" % (engine_file, t1 - t0))
		logger.log(trt.Logger.INFO, 'Recording engine output shape in config')
		# 创建一个临时的 Session 来获取输出形状
		engine_session = Session.from_serialized_engine(engine_string)
		output_tensor_name = network.get_output(0).name
		output_shape = engine_session.engine.get_tensor_shape(output_tensor_name)
		output_shape = list(output_shape)
		config_wrapper.output_shape = output_shape

		os.makedirs(engine_dir, exist_ok=True)
		with open(engine_file, 'wb') as f:
			f.write(engine_string)

		if delete_onnx:
			shutil.rmtree(onnx_dir)

	# 保存包含元数据（如输出形状）的配置文件
	Builder.save_config(config_wrapper, config_file)


def compute_rotary_pos_emb(grid_thw, hf_config, VisionRotaryEmbedding):
	"""
	为 Qwen2-VL 的视觉变换器计算旋转位置编码 (Rotary Position Embedding)。
	这是模型在处理不同尺寸图像时保持位置感知能力的关键部分。

	Args:
		grid_thw (torch.Tensor): 描述图像网格尺寸的张量 (temporal, height, width)。
		hf_config: Hugging Face 模型的配置对象。
		VisionRotaryEmbedding: 用于生成旋转编码的类。

	Returns:
		torch.Tensor: 计算好的旋转位置编码。
	"""
	head_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads
	rotary_pos_emb_func = VisionRotaryEmbedding(head_dim // 2)

	def rot_pos_emb(grid_thw, rotary_pos_emb_func):
		pos_ids = []
		for t, h, w in grid_thw:
			hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
			hpos_ids = hpos_ids.reshape(h // hf_config.vision_config.spatial_merge_size,
										hf_config.vision_config.spatial_merge_size,
										w // hf_config.vision_config.spatial_merge_size,
										hf_config.vision_config.spatial_merge_size)
			hpos_ids = hpos_ids.permute(0, 2, 1, 3)
			hpos_ids = hpos_ids.flatten()

			wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
			wpos_ids = wpos_ids.reshape(h // hf_config.vision_config.spatial_merge_size,
										hf_config.vision_config.spatial_merge_size,
										w // hf_config.vision_config.spatial_merge_size,
										hf_config.vision_config.spatial_merge_size)
			wpos_ids = wpos_ids.permute(0, 2, 1, 3)
			wpos_ids = wpos_ids.flatten()
			pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
		pos_ids = torch.cat(pos_ids, dim=0)
		max_grid_size = grid_thw[:, 1:].max()
		rotary_pos_emb_full = rotary_pos_emb_func(max_grid_size)
		rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
		return rotary_pos_emb

	rotary_pos_emb = rot_pos_emb(grid_thw, rotary_pos_emb_func)
	return rotary_pos_emb


def build_qwen2_vl_engine(args):
	"""
	构建 Qwen2-VL 视觉编码器引擎的主函数。

	此函数执行以下操作：
	1. 从 Hugging Face Hub 加载预训练的 Qwen2-VL 模型和处理器。
	2. 创建一个示例输入，以模拟真实的使用场景。
	3. 定义并应用优化的模型组件（如 VisionAttentionOpt），这些组件被修改以适应 ONNX 导出和 TensorRT 推理。
	4. 将修改后的视觉编码器包装在 VisionEncoderWrapper 中。
	5. 调用 export_onnx 将包装好的模型导出为 ONNX 文件。
	6. 调用 build_trt_engine 从 ONNX 文件构建 TensorRT 引擎。

	Args:
		args: 从命令行解析的参数。
	"""
	from qwen_vl_utils import process_vision_info
	from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
	from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
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
		"role": "user",
		"content": [
			{"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
			{"type": "text", "text": "Describe this picture?"},
		],
	}]
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	image_inputs, video_inputs = process_vision_info(messages)
	for i in range(len(image_inputs)):
		image_inputs[i] = image_inputs[i].resize((image_inputs[i].size[0] // 2, image_inputs[i].size[1] // 2))
	inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
	image = inputs['pixel_values'].to(torch.float16)
	image_grid_thw = inputs['image_grid_thw']
	cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
	cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
	seq_length = image.shape[0]
	attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(image.dtype).min, device=image.device, dtype=image.dtype)
	for i in range(1, len(cu_seqlens)):
		attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = 0
	rotary_pos_emb = compute_rotary_pos_emb(image_grid_thw, hf_config, VisionRotaryEmbedding)

	class VisionAttentionOpt(VisionAttention):

		def __init__(self, config: Qwen2VLVisionConfig):
			super().__init__(config)
			self.head_dim = config.embed_dim // config.num_heads

		def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
			seq_length = hidden_states.shape[0]
			q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

			def rotate_half(x):
				x1 = x[..., :x.shape[-1] // 2]
				x2 = x[..., x.shape[-1] // 2:]
				return torch.cat((-x2, x1), dim=-1)

			def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
				orig_dtype = tensor.dtype
				tensor = tensor.float()
				cos = freqs.cos()
				sin = freqs.sin()
				cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
				sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
				output = (tensor * cos) + (rotate_half(tensor) * sin)
				output = output.to(orig_dtype)
				return output

			q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
			k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
			q = q.transpose(0, 1)
			k = k.transpose(0, 1)
			v = v.transpose(0, 1)
			attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
			attn_weights = attn_weights + attention_mask
			attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
			attn_output = torch.matmul(attn_weights, v)
			attn_output = attn_output.transpose(0, 1)
			attn_output = attn_output.reshape(seq_length, -1)
			attn_output = self.proj(attn_output)
			return attn_output

	class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

		def __init__(self, config, attn_implementation: str = "eager") -> None:
			super().__init__(config)
			self.attn = VisionAttentionOpt(config)

		def forward(self, hidden_states, attention_mask, rotary_pos_emb) -> torch.Tensor:
			hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
			hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
			return hidden_states

	class Qwen2VisionTransformerPretrainedModelOpt(Qwen2VisionTransformerPretrainedModel):

		def __init__(self, config) -> None:
			super().__init__(config)
			self.blocks = nn.ModuleList([Qwen2VLVisionBlockOpt(config, config._attn_implementation) for _ in range(config.depth)])

		def forward(self, hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
			hidden_states = self.patch_embed(hidden_states)
			for blk in self.blocks:
				hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
			res = self.merger(hidden_states)
			return res

	class VisionEncoderWrapper(torch.nn.Module):

		def __init__(self, model):
			super().__init__()
			self.visual = Qwen2VisionTransformerPretrainedModelOpt._from_config(model.config.vision_config, torch_dtype=torch.float32)
			self.visual.load_state_dict(model.visual.state_dict())

		def forward(self, images, rotary_pos_emb, attention_mask):
			"""
			模型的前向传播，接收图像、旋转编码和注意力掩码作为输入。
			"""
			img_features = self.visual(images, rotary_pos_emb, attention_mask)
			return img_features

	# 包装模型以准备导出
	wrapper = VisionEncoderWrapper(model)
	dynamic_axes = {
		'input': {0: 'hw'},
		'rotary_pos_emb': {0: 'hw'},
		'attention_mask': {1: 'hw', 2: 'hw'}
	}
	# 导出到 ONNX
	export_onnx(wrapper, (image, rotary_pos_emb, attention_mask), f'{args.output_dir}/onnx', input_names=['input', 'rotary_pos_emb', 'attention_mask'], output_names=['encoder_output'], dynamic_axes=dynamic_axes)
	rotary_pos_emb_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads // 2
	# 从 ONNX 构建 TRT 引擎
	build_trt_engine(args.model_type, [rotary_pos_emb_dim], f'{args.output_dir}/onnx', args.output_dir, args.max_batch_size, model_params={'qwen2_vl_dim': qwen2_vl_dim, 'min_hw_dims': args.min_hw_dims, 'max_hw_dims': args.max_hw_dims})


def main():
	"""
	程序入口点。解析命令行参数并启动构建过程。
	"""
	parser = argparse.ArgumentParser(description="Build Qwen2-VL vision TRT engine")
	parser = add_multimodal_arguments(parser)
	args = parser.parse_args()
	args.model_type = 'qwen2_vl'
	if args.output_dir is None:
		model_name = args.model_path.split('/')[-1]
		args.output_dir = f'tmp/trt_engines/{model_name}/vision'
	os.makedirs(args.output_dir, exist_ok=True)
	if args.onnx_path is None:
		build_qwen2_vl_engine(args)
	else:
		hf_config = AutoConfig.from_pretrained(args.model_path)
		rotary_pos_emb_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads // 2
		qwen2_vl_dim = hf_config.vision_config.in_chans * hf_config.vision_config.patch_size * hf_config.vision_config.patch_size * hf_config.vision_config.temporal_patch_size
		build_trt_engine(args.model_type, [rotary_pos_emb_dim], args.onnx_path, args.output_dir, args.max_batch_size, model_params={'qwen2_vl_dim': qwen2_vl_dim, 'min_hw_dims': args.min_hw_dims, 'max_hw_dims': args.max_hw_dims})

if __name__ == '__main__':
	import argparse
	main()

