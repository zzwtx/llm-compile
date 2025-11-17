
import json
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
	2. 使用 tensorrt 的原生 API 创建 Builder Config，配置精度等参数。
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
	config = builder.create_builder_config()

	# 2. 配置构建参数
	# 设置 FP16 精度
	if dtype == torch.float16:
		config.set_flag(trt.BuilderFlag.FP16)
	
	# 设置最大工作空间大小 (示例值，可以根据需要调整)
	# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 4GB

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
		
		# 使用纯 TensorRT API 获取输出形状
		runtime = trt.Runtime(logger)
		engine = runtime.deserialize_cuda_engine(engine_string)
		output_tensor_name = network.get_output(0).name
		output_shape = engine.get_tensor_shape(output_tensor_name)
		
		# 准备要保存的配置信息
		config_json = {
			"model_type": model_type,
			"max_batch_size": max_batch_size,
			"model_name": "qwen2_vl",
			"output_shape": list(output_shape)
		}
		if "num_frames" in model_params:
			config_json["num_frames"] = model_params["num_frames"]

		os.makedirs(engine_dir, exist_ok=True)
		with open(engine_file, 'wb') as f:
			f.write(engine_string)
		
		# 保存配置文件
		with open(config_file, 'w') as f:
			json.dump(config_json, f, indent=4)

		if delete_onnx:
			shutil.rmtree(onnx_dir)


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
		print("Exporting ONNX and building TRT engine...")
	else:
		hf_config = AutoConfig.from_pretrained(args.model_path)
		rotary_pos_emb_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads // 2
		qwen2_vl_dim = hf_config.vision_config.in_chans * hf_config.vision_config.patch_size * hf_config.vision_config.patch_size * hf_config.vision_config.temporal_patch_size
		build_trt_engine(args.model_type, [rotary_pos_emb_dim], args.onnx_path, args.output_dir, args.max_batch_size, model_params={'qwen2_vl_dim': qwen2_vl_dim, 'min_hw_dims': args.min_hw_dims, 'max_hw_dims': args.max_hw_dims})

if __name__ == '__main__':
	import argparse
	main()

