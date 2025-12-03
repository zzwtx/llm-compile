import argparse
import json
import os
import sys
from collections.abc import Iterator
from io import BytesIO

from utils import add_common_args, compute_str_match_rate

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer)
from _utils import (str_dtype_to_torch, str_dtype_to_trt,
                      supports_inflight_batching, torch_dtype_to_trt,
                      trt_dtype_to_torch)
from tensorrt_llm._utils import mpi_rank

from typing import Optional, Tuple
import torch.nn.functional as F
from tensorrt_llm.runtime.session import Session, TensorInfo
from model_runner import ModelRunner
from PIL import Image, UnidentifiedImageError
try:
    from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp
except Exception:
    ModelRunnerCpp = None

from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from functional import RopeEmbeddingUtils, RotaryScalingType
from attention import MropeParams
import torch
import os
import json
import numpy as np
import requests
import time
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.runtime.enc_dec_model_runner import EncDecModelRunner
import asyncio

try:
    import tensorrt_llm.bindings  # NOQA
    PYTHON_BINDINGS = True
except ImportError:
    PYTHON_BINDINGS = False

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp

# from tensorrt_llm.runtime import MultimodalModelRunner

def compute_rotary_pos_emb(grid_thw, hf_config, VisionRotaryEmbedding):
    head_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads
    rotary_pos_emb_func = VisionRotaryEmbedding(head_dim // 2)
    hf_config.vision_config.spatial_merge_size

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

class MultimodalModelRunner:

    def __init__(self, args):
        self.args = args
        self.use_trtllm_vision_engine = False

        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        if self.args.mm_embedding_offloading is None:
            self.args.mm_embedding_offloading = self.args.enable_chunked_context
        elif self.args.mm_embedding_offloading and not self.args.enable_chunked_context:
            logger.warning(
                "mm_embedding_offloading requires enable_chunked_context to be True. Setting mm_embedding_offloading to None."
            )
            self.args.mm_embedding_offloading = None

        # parse model type from visual engine config
        with open(os.path.join(self.visual_engine_dir, "config.json"),
                  "r") as f:
            config = json.load(f)
        if 'pretrained_config' in config:
            if config['pretrained_config'][
                    'architecture'] == 'LlavaNextForConditionalGeneration':
                self.model_type = 'llava_next'
                self.vision_precision = config['pretrained_config']['dtype']
                self.use_trtllm_vision_engine = True
            else:
                logger.error(
                    "Currently only Llava-NeXT supports TRT-LLM vision engines."
                )
        else:
            self.model_type = config['builder_config']['model_type']
            self.vision_precision = config['builder_config']['precision']
            
        # qwen2_vl uses a decoder-style LLM in this runner
        self.decoder_llm = True

        hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id
        self.vision_token_id = hf_config.vision_token_id
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.spatial_merge_size = hf_config.vision_config.spatial_merge_size
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.hidden_size = hf_config.hidden_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.rope_theta = hf_config.rope_theta
        
        self.audio_input_names = self.audio_output_names = None
        
        self.vision_input_names = ["input"]
        self.vision_output_names = ["encoder_output"]

        self.session = args.session
        if self.cpp_e2e:
            self.visual_output_shape = config['builder_config'].get(
                'output_shape', None)
        if self.decoder_llm:
            if not supports_inflight_batching(self.llm_engine_dir):
                logger.warning(
                    "The given engine does not support in-flight batching, both visual engine and LLM fallback to python session"
                )
                self.session = 'python'

            if not PYTHON_BINDINGS and 'cpp' in args.session:
                logger.warning(
                    "Python bindings of C++ session is unavailable, both visual engine and LLM fallback to Python session."
                )
                self.session = 'python'

            args.debug_mode = False
            if args.debug_mode and 'cpp' in args.session:
                logger.warning(
                    "Debug mode is not supported in C++ session for now, both visual engine and LLM fallback to Python session."
                )
                self.session = 'python'

            # qwen2_vl requires cpp_llm_only session by design for this runner
            # if self.args.session != "cpp_llm_only":
            #     logger.warning(
            #         "Qwen2-vl only support C++ session for now, fallback to C++ session."
            #     )
            #     self.args.session = "cpp_llm_only"

            if args.session == 'cpp':
                logger.warning(
                    f'C++ end-to-end mode does not support {self.model_type}. Visual engine fallbacks to Python session. See support matrix in README.'
                )
                args.session = 'cpp_llm_only'
            self.session = args.session

        else:
            self.session = 'cpp_llm_only'

        self.init_tokenizer()
        self.init_processor()
        self.init_image_encoder()
        self.init_llm()

        self.last_streaming_metrics = None

        self.audio_encoder_session = self.audio_precision = None

    @property
    def cpp_e2e(self):
        return self.session == 'cpp'

    @property
    def cpp_llm_only(self):
        return self.session == 'cpp_llm_only'

    @property
    def python_e2e(self):
        return self.session == 'python'

    @property
    def visual_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'vision')

    @property
    def audio_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'audio')

    @property
    def llm_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'llm')

    def init_tokenizer(self):
        use_fast = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.hf_model_dir,
            use_fast=use_fast,
            use_legacy=False,
            trust_remote_code=True)

        self.tokenizer.padding_side = "right"

    def init_processor(self):
        from torchvision import transforms
        self.processor = AutoProcessor.from_pretrained(
            self.args.hf_model_dir, trust_remote_code=True, num_crops=16)

    def init_image_encoder(self):
        # Phi-4-multimodal uses pytorch engine due to issues with creating TRT engine.
        if self.cpp_e2e:
            logger.info(
                "Using C++ runtime for both visual engine and LLM decoder, skip loading visual engine in Python runtime."
            )
        else:
            vision_encoder_path = os.path.join(self.visual_engine_dir,
                                               self.args.visual_engine_name)
            logger.info(f'Loading engine from {vision_encoder_path}')
            with open(vision_encoder_path, 'rb') as f:
                engine_buffer = f.read()
            logger.info(f'Creating session from engine {vision_encoder_path}')
            self.visual_encoder_session = Session.from_serialized_engine(
                engine_buffer)

    def init_llm(self):
        if self.decoder_llm:
            cross_kv_cache_fraction = None
            if self.python_e2e:
                logger.info(f'Running LLM with Python runner')
                self.model = ModelRunner.from_dir(
                    self.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    stream=self.stream,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc,
                )
                self.model_config = self.model.session._model_config
            elif self.cpp_e2e:
                logger.info(
                    f'Running both visual engine and LLM with Python runner')
                self.model = ModelRunnerCpp.from_dir(
                    self.args.engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    is_enc_dec=True,  # TODO: add a separate model variant here?
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc)
                self.model_config = self.model.model_config
            else:
                logger.info(f'Running LLM with C++ runner')
                self.model = ModelRunnerCpp.from_dir(
                    self.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    enable_chunked_context=self.args.enable_chunked_context,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc,
                    kv_cache_free_gpu_memory_fraction=self.args.
                    kv_cache_free_gpu_memory_fraction,
                    # cross_kv_cache_fraction=cross_kv_cache_fraction,
                    multi_block_mode=self.args.multi_block_mode,
                    # mm_embedding_offloading=self.args.mm_embedding_offloading,
                )
                self.model_config = self.model.model_config
            self.runtime_mapping = self.model.mapping
        else:
            self.model = EncDecModelRunner.from_engine(
                os.path.basename(self.args.hf_model_dir),
                self.llm_engine_dir,
                skip_encoder=self.model_type in ['nougat', 'pix2struct'],
                debug_mode=False,
                stream=self.stream,
                enable_context_fmha_fp32_acc=self.args.
                enable_context_fmha_fp32_acc)
            self.model_config = self.model.encoder_model_config
            self.runtime_mapping = self.model.encoder_runtime_mapping

    def preprocess(self, pre_prompt, post_prompt, image, other_vision_inputs,
                   other_audio_inputs):
        audio = None
        # same prompt for single/multiple image(s)
        n_prompts_n_images = False
        if isinstance(post_prompt,
                      list) and len(post_prompt) > 1 and image is not None:
            if hasattr(image, "pixel_values"):
                if len(post_prompt) == image["pixel_values"].shape[0]:
                    n_prompts_n_images = True
                    # n prompts and n images
            else:
                if isinstance(
                        image,
                        torch.Tensor) and len(post_prompt) == image.shape[0]:
                    n_prompts_n_images = True
                    # n prompts and n images

        # qwen2_vl preprocessing (always executed in this simplified runner)
        input = image
        image = input['image']
        input_ids = input['input_ids']
        other_vision_inputs['image_grid_thw'].shape[0]
        attention_mask = other_vision_inputs['attention_mask_llm']
        other_vision_inputs.pop('attention_mask_llm')
        image_grid_thw = other_vision_inputs['image_grid_thw']
        other_vision_inputs.pop('image_grid_thw')

        profiler.start("Vision encoder")
        visual_features, visual_atts, model_runner_input = None, None, None
        if image is not None:
            model_runner_input = torch.stack(
                image['image_patches'],
                dim=0) if self.model_type == 'fuyu' else image

            if self.cpp_e2e:
                # If using E2E C++ runtime, visual_features will not be computed here in Python runtime.
                # Instead, it only contains a shape read from the engine config, and is used for generating
                # decoder prompt later
                logger.info(
                    'Skip running visual engine, get visual output shape from engine config.'
                )
                model_runner_input = model_runner_input.to(
                    str_dtype_to_torch(self.vision_precision))
                batch_size = model_runner_input.shape[0]
                output_shape = list(self.visual_output_shape)
                output_shape[0] = batch_size
                if self.model_type == 'fuyu':
                    output_shape[1] = model_runner_input.shape[
                        2]  # fuyu's output patch number is not fixed, same as input patch number
                visual_features = TensorInfo(
                    'encoder_output',
                    str_dtype_to_trt(self.vision_precision),
                    tuple(output_shape))
                atts_shape = visual_features.shape[:-1]
                visual_atts = TensorInfo('image_atts', None,
                                            tuple(atts_shape))
                model_runner_input = torch.vsplit(
                    model_runner_input, model_runner_input.shape[0])
            else:
                visual_features, visual_atts = self.get_visual_features(
                    model_runner_input, other_vision_inputs)
                model_runner_input = None
        profiler.stop("Vision encoder")

        profiler.start("Audio encoder")
        audio_features = None
        profiler.stop("Audio encoder")

        # qwen2_vl path (always taken)
        length = input_ids.shape[1]
        input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
            torch.int32)
        input_ids, ptuning_args, mrope_args, mrope_position_ids = self.setup_fake_prompts_qwen2vl(
            visual_features, input_ids, image_grid_thw, attention_mask,
            input_lengths)
        
        # Flatten position_ids if needed (for remove_input_padding)
        # mrope_position_ids is [batch, 3, seq] (transposed in setup_fake_prompts)
        # We need [3, num_tokens]
        # attention_mask is [batch, seq]
        # Note: input_ids was modified to include fake prompts, but attention_mask might not have been updated?
        # Wait, setup_fake_prompts modifies input_ids in place or returns new one?
        # It returns new input_ids.
        # Does attention_mask match the new input_ids?
        # In setup_fake_prompts, input_ids is modified by inserting fake prompts.
        # But attention_mask is passed in.
        # If input_ids length changed, attention_mask is invalid!
        # Let's check setup_fake_prompts.
        # It replaces tokens in input_ids with fake prompt IDs. It does NOT change length.
        # "input_ids[masks] = values[masks]" -> In-place modification of values, shape preserved.
        # So attention_mask is still valid.
        
        # Permute to [3, batch, seq]
        mrope_position_ids = mrope_position_ids.permute(1, 0, 2)
        # Flatten using attention_mask
        # We need to broadcast attention_mask to [3, batch, seq]
        # attention_mask is [batch, seq]
        # mask = attention_mask.unsqueeze(0).expand(3, -1, -1)
        # flat_position_ids = mrope_position_ids[mask.bool()].view(3, -1)
        # Wait, boolean indexing flattens.
        # mrope_position_ids[:, attention_mask.bool()] should work and return [3, num_tokens]
        
        position_ids = mrope_position_ids[:, attention_mask.bool().to(mrope_position_ids.device)]
        
        return input_ids, input_lengths, ptuning_args, visual_features, mrope_args, position_ids

    @staticmethod
    def tokenizer_image_token(batch_size,
                              pre_prompt,
                              post_prompt,
                              tokenizer,
                              image_token_index=-200):
        if isinstance(post_prompt, list):
            prompts = [pre_prompt + item for item in post_prompt]
        else:
            prompts = [pre_prompt + post_prompt]

        def insert_separator(X, sep):
            return [
                ele for sublist in zip(X, [sep] * len(X)) for ele in sublist
            ][:-1]

        result = []
        for prompt in prompts:
            prompt_chunks = [
                tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
            ]
            input_ids = []
            offset = 0
            if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0
                    and prompt_chunks[0][0] == tokenizer.bos_token_id):
                offset = 1
                input_ids.append(prompt_chunks[0][0])

            for x in insert_separator(prompt_chunks,
                                      [image_token_index] * (offset + 1)):
                input_ids.extend(x[offset:])

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids[input_ids == image_token_index] = 0
            result.append(input_ids)

        if not isinstance(post_prompt, list):
            result = result[0].unsqueeze(0).expand(batch_size, -1)
        return result

    def split_prompt_by_images(self, tensor):
        batch_splits = []
        for batch in tensor:
            # Find indices where value is zero (<image>)
            zero_indices = (batch == 0).nonzero(as_tuple=False).squeeze(0)
            # Add starting point for slicing
            start_idx = 0
            splits = []
            for idx in zero_indices:
                if start_idx != idx:  # Ensure not slicing zero-length tensors
                    splits.append(batch[start_idx:idx].unsqueeze(0))
                start_idx = idx + 1  # Move start index past the zero
            if start_idx < len(
                    batch):  # Handle last segment if it's not zero-ending
                splits.append(batch[start_idx:].unsqueeze(0))
            # Remove empty tensors resulting from consecutive zeros
            splits = [split for split in splits if split.numel() > 0]
            batch_splits.append(splits)

        return batch_splits

    def prepare_position_ids_for_cogvlm(self, input_ids):
        batch_size = len(input_ids)
        position_ids = torch.arange(input_ids.shape[1])
        position_ids[2:1227] = 2
        position_ids[1227:] = torch.arange(3, input_ids.shape[1] + 1 - 1225)

        position_ids = position_ids.to(torch.int32).to('cuda')
        input_position_ids = []
        for i in range(batch_size):
            input_position_ids.append(position_ids)

        return input_position_ids

    def generate(self,
                 pre_prompt,
                 post_prompt,
                 image,
                 decoder_input_ids,
                 max_new_tokens,
                 other_vision_inputs={},
                 other_audio_inputs={},
                 other_decoder_inputs={}):
        profiler.start("Generate")
        profiler.start("Preprocess")
        # qwen2_vl preprocessing (always executed)
        input_ids, input_lengths, ptuning_args, visual_features, mrope_args, position_ids = self.preprocess(
            pre_prompt, post_prompt, image, other_vision_inputs,
            other_audio_inputs)
        mrope_params = MropeParams(
            mrope_rotary_cos_sin=mrope_args[0],
            mrope_position_deltas=mrope_args[1],
        )
        profiler.stop("Preprocess")

        print(f"DEBUG: position_ids type: {type(position_ids)}")
        if isinstance(position_ids, torch.Tensor):
            print(f"DEBUG: position_ids shape: {position_ids.shape}")
        elif isinstance(position_ids, list):
            print(f"DEBUG: position_ids list len: {len(position_ids)}")
            if len(position_ids) > 0:
                print(f"DEBUG: position_ids[0] shape: {position_ids[0].shape}")
        
        print(f"DEBUG: input_ids type: {type(input_ids)}")
        if isinstance(input_ids, torch.Tensor):
            print(f"DEBUG: input_ids shape: {input_ids.shape}")
            # Print sample position_ids to verify they are not all zeros and have 3D structure
            if isinstance(position_ids, torch.Tensor):
                print(f"DEBUG: position_ids sample (first 5 cols):\n{position_ids[:, :5]}")
                print(f"DEBUG: position_ids max values: {position_ids.max(dim=1)[0]}")

        # use prompt tuning to pass multimodal features
        # model.generate() expects the following params (see layers/embedding.py):
        # args[0]: prompt embedding table, [batch_size, multimodal_len, hidden_size], later flattened to [batch_size * multimodal_len, hidden_size]
        # args[1]: prompt task ids, [batch_size]. in multimodal case, arange(batch_size), i.e. in VILA batching mode 2, each image is treated separately in the batch instead of concated together (although the prompt embedding table has to be concated)
        # args[2]: prompt task vocab size, [1]. assuming all table has the same length, which in multimodal case equals to multimodal_len
        profiler.start("LLM")
        llm_start_time = time.perf_counter()
        streaming_result = None
        if self.decoder_llm:
            end_id = self.tokenizer.eos_token_id

            prompt_tasks = None
            prompt_table = None
            if not self.cpp_e2e:
                batch_size = len(input_ids)
                prompt_tasks = ",".join(
                    np.arange(batch_size, dtype=np.int32).astype(str))
                prompt_table = torch.stack([ptuning_args[0]])
                prompt_table = prompt_table.view(batch_size, -1,
                                                 prompt_table.shape[-1])

            streaming_result = self.model.generate(
                input_ids,
                position_ids=position_ids,
                mrope_params=mrope_params,
                sampling_config=None,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.all_special_ids[0],
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                num_beams=self.args.num_beams,
                lora_uids=self.args.lora_task_uids,
                output_sequence_lengths=False,
                return_dict=False,
                mm_embedding_offloading=self.args.mm_embedding_offloading,
                streaming=True,
            )

        streaming_metrics = {
            "llm_start_time": llm_start_time,
            "llm_end_time": None,
            "first_token_time": None,
            "chunk_count": 0,
        }
        final_output_ids = None
        if isinstance(streaming_result, Iterator):
            last_chunk = None
            for chunk in streaming_result:
                last_chunk = chunk
                streaming_metrics["chunk_count"] += 1
                if streaming_metrics["first_token_time"] is None:
                    streaming_metrics["first_token_time"] = time.perf_counter()
            streaming_metrics["llm_end_time"] = time.perf_counter()
            final_output_ids = last_chunk
        else:
            streaming_metrics["llm_end_time"] = time.perf_counter()
            final_output_ids = streaming_result

        if final_output_ids is None:
            final_output_ids = streaming_result

        self.last_streaming_metrics = streaming_metrics
        profiler.stop("LLM")

        if mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            profiler.start("Tokenizer decode")
            output_beams_list = [
                self.tokenizer.batch_decode(
                    final_output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True) for batch_idx in range(
                        min(self.args.batch_size, input_lengths.shape[0]))
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(
                min(self.args.batch_size, input_lengths.shape[0]))]
            profiler.stop("Tokenizer decode")
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image, other_vision_inputs):
        visual_features = {
            self.vision_input_names[0]:
            image.to(str_dtype_to_torch(self.vision_precision)),
        }
        # qwen2_vl: ensure attention_mask uses engine dtype
        other_vision_inputs['attention_mask'] = other_vision_inputs[
            'attention_mask'].to(str_dtype_to_torch(self.vision_precision))
        for key, tensor in other_vision_inputs.items():
            visual_features.update({key: tensor})

        tensor_info = [
            TensorInfo(self.vision_input_names[0],
                       str_dtype_to_trt(self.vision_precision), image.shape),
        ]
        for key, tensor in other_vision_inputs.items():
            tensor_info.append(
                TensorInfo(key, torch_dtype_to_trt(tensor.dtype), tensor.shape))

        visual_output_info = self.visual_encoder_session.infer_shapes(
            tensor_info)
        self.visual_encoder_session.set_shapes(visual_features)
        visual_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs[self.vision_output_names[0]]

        if self.args.mm_embedding_offloading:
            # CUDA Stream Overlapping Requirements:
            # 1. Both memory copy stream and kernel execution stream must be non-default streams
            # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
            pinned_embeds = torch.empty_like(image_embeds,
                                             device='cpu',
                                             pin_memory=True)
            pinned_embeds.copy_(image_embeds, non_blocking=True)
            image_embeds = pinned_embeds

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        if hasattr(self, 'num_frames') and (visual_features.shape[1]
                                            == self.num_frames):
            visual_features = visual_features.view(visual_features.shape[0], -1,
                                                   visual_features.shape[-1])

        if visual_features is not None:
            if self.python_e2e:
                # Non-IFB Mode(used in python session): All requests in a batch have their prompt_table concatenated in
                # a shape of (bs*vision_embedding_len, vision_hidden). So only one fake_prompt_id is needed for the
                # entire batch, with values from 0 to bs * vision_embedding_len-1.
                fake_prompt_id = torch.arange(
                    self.model_config.vocab_size, self.model_config.vocab_size +
                    visual_features.shape[0] * visual_features.shape[1])
                fake_prompt_id = fake_prompt_id.reshape(
                    visual_features.shape[0], visual_features.shape[1])
            else:
                # IFB Mode(used in c++ session): Each request's prompt_table is independent and requires a fake_prompt_id
                # for each request, with values ranging from 0 to vision_embedding_len-1.
                fake_prompt_id = torch.arange(
                    self.model_config.vocab_size,
                    self.model_config.vocab_size + visual_features.shape[1])
                fake_prompt_id = fake_prompt_id.repeat(visual_features.shape[0],
                                                       1)

        if post_input_ids is not None:
            if isinstance(post_input_ids, list):
                pre_input_fake_prompt_ids = [
                    pre_input_ids[:len(fake_prompt_id)], fake_prompt_id
                ]
                pre_input_fake_prompt_ids = torch.cat(
                    pre_input_fake_prompt_ids,
                    dim=1).contiguous().to(torch.int32)
                input_ids = [
                    torch.cat((pre_input_fake_prompt_id,
                                post_input_id)).contiguous().to(torch.int32)
                    for pre_input_fake_prompt_id, post_input_id in zip(
                        pre_input_fake_prompt_ids, post_input_ids)
                ]
            else:
                input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
                input_ids = torch.cat(input_ids,
                                        dim=1).contiguous().to(torch.int32)
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
            input_ids = torch.cat(input_ids,
                                    dim=1).contiguous().to(torch.int32)

        if (self.decoder_llm or self.runtime_mapping.is_first_pp_rank()
            ) and self.model_type != "mllama" and isinstance(
                visual_features, torch.Tensor):
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def get_rope_index(
        self,
        input_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.IntTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(3,
                                      input_ids.shape[0],
                                      input_ids.shape[1],
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len +
                        st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                llm_positions = torch.cat(llm_pos_ids_list,
                                          dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 -
                                             len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def setup_fake_prompts_qwen2vl(self, visual_features, input_ids,
                                   vision_grid_thws, attention_mask,
                                   input_lengths):

        visual_features = torch.unsqueeze(visual_features, 0)

        # Get the rope index
        # From HF's preprocess code
        mrope_position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=vision_grid_thws,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        # This is where we convert input_ids of image features into fake_prompt_ids mapping for TRT-LLM engine.
        masks = (input_ids == self.image_token_id) | (
            input_ids == self.vision_token_id) | (input_ids
                                                  == self.video_token_id)
        cumulative_counts = masks.cumsum(dim=1)
        values = (self.model_config.vocab_size - 1) + cumulative_counts
        input_ids[masks] = values[masks]

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        # This does not have dependency on input.
        # Switch to attributes to use across iterations.
        if not hasattr(self, 'rotary_cos_sin'):
            inv_freq, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                num_pos=self.max_position_embeddings,
                dim=int(self.hidden_size / self.num_attention_heads),
                theta=float(self.rope_theta),
                scale_type=RotaryScalingType.mrope)
            self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(
                visual_features.device)
            self.rotary_cos_sin = self.rotary_cos_sin.reshape(
                self.max_position_embeddings,
                int(self.hidden_size / self.num_attention_heads / 2), 2)
            self.cos_ori = self.rotary_cos_sin[:, :, 0]
            self.sin_ori = self.rotary_cos_sin[:, :, 1]

        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        mrope_position_ids_padding = torch.zeros(
            mrope_position_ids.shape[:-1] + (self.max_position_embeddings, ),
            dtype=torch.int32,
            device=visual_features.device)
        mrope_position_ids_padding[:, :, :mrope_position_ids.
                                   shape[-1]] = mrope_position_ids
        cos = self.cos_ori[mrope_position_ids_padding]
        sin = self.sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        cos = torch.cat([
            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        sin = torch.cat([
            m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)

        # Pass the base table (self.rotary_cos_sin) instead of the processed one (concat_cos_sin)
        # because attention.py logic expects the base table to perform lookup with position_ids.
        # mrope_args = [concat_cos_sin, mrope_position_deltas]
        
        # Flatten to [1, max_pos * head_dim] to match engine expectation [1, 4194304]
        rotary_cos_sin_flat = self.rotary_cos_sin.view(1, -1)
        mrope_args = [rotary_cos_sin_flat, mrope_position_deltas]
        # mrope_position_ids was transposed to [batch, 3, seq] at line 938?
        # Line 938: mrope_position_ids = mrope_position_ids.transpose(1, 0)
        # Original get_rope_index returns [3, batch, seq].
        # So now it is [batch, 3, seq].
        # We want to return it.
        return input_ids, ptuning_args, mrope_args, mrope_position_ids

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))

            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            if hasattr(self.model_config, 'dtype'):
                prompt_table = prompt_table.cuda().to(
                    dtype=str_dtype_to_torch(self.model_config.dtype))
            else:
                if self.args.mm_embedding_offloading:
                    # CUDA Stream Overlapping Requirements:
                    # 1. Both memory copy stream and kernel execution stream must be non-default streams
                    # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
                    prompt_table = prompt_table.pin_memory().to(
                        dtype=self.model.dtype)
                else:
                    prompt_table = prompt_table.cuda().to(
                        dtype=self.model.dtype)
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        remove_input_padding = self.model_config.remove_input_padding if hasattr(
            self.model_config,
            'remove_input_padding') else self.model_config.use_packed_input
        if remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if self.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            if not isinstance(input_ids, list):
                tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()
            else:
                max_length = max(input_id.size(-1) for input_id in input_ids)
                tasks = torch.zeros((len(input_ids), max_length),
                                    dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def load_test_data(self, image_path=None, video_path=None):

        def load_images(image_paths):
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            images = []
            for image_path in image_paths:
                if image_path.startswith("http") or image_path.startswith(
                        "https"):
                    logger.info(f"downloading image from url {image_path}")
                    try:
                        response = requests.get(image_path, timeout=5)
                        response.raise_for_status()
                        if 'image' not in response.headers.get(
                                'Content-Type', ''):
                            raise Exception(
                                f"URL does not point to an image: {image_path}."
                            )
                        image = Image.open(BytesIO(
                            response.content)).convert("RGB")
                    except (UnidentifiedImageError, IOError):
                        raise Exception(
                            f"Cannot identify image file at URL: {image_path}.")
                    except Exception as e:
                        raise Exception(
                            f"Failed to download image from url {image_path}: {e}"
                        )
                else:
                    image = Image.open(image_path).convert("RGB")
                images.append(image)
            return images if len(images) > 1 else images[0]

        # If running profiling, prefer local images directory (/home/xsf/fl/edge/images).
        # If not found or empty, fall back to streaming from HF mirror (best-effort),
        # then to the bundled demo image.
        if self.args.run_profiling:
            images = []
            local_dir = '/home/xsf/fl/edge/images'
            if os.path.isdir(local_dir):
                files = sorted([f for f in os.listdir(local_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                for fn in files[:10]:
                    try:
                        path = os.path.join(local_dir, fn)
                        img = Image.open(path).convert('RGB')
                        img = img.resize((504, 504))
                        images.append(img)
                    except Exception:
                        logger.warning(f'Failed to load image {fn} from {local_dir}, skipping')

            if len(images) > 0:
                logger.info(f'Using {len(images)} local profiling images from {local_dir}')

            else:
                # Fallback to the original demo image if no downloads succeeded
                logger.warning('No profiling images available; falling back to bundled demo image')
                demo_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
                try:
                    image = Image.open(requests.get(demo_url, stream=True, timeout=5).raw).convert('RGB')
                    image = image.resize((504, 504))
                    images = [image]
                except Exception:
                    images = []

            return images

        # qwen2_vl demo/test image loading (non-profiling path)
        images = []
        if self.args.image_path is None:
            img_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
            image = Image.open(
                requests.get(img_url, stream=True,
                             timeout=5).raw).convert('RGB')
            image = image.resize((504, 504))
            images.append(image)
        else:
            images = []
            for image_path in self.args.image_path:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((504, 504))
                images.append(image)
        return images

    def setup_inputs(self, input_text, raw_image, raw_audio=None):
        # from tensorrt_llm.tools.multimodal_builder import compute_rotary_pos_emb
        other_vision_inputs = {}
        other_audio_inputs = {}
        other_decoder_inputs = {}
        if 'qwen2_vl' in self.model_type:
            from qwen_vl_utils import process_vision_info
            from transformers.models.qwen2_vl.modeling_qwen2_vl import \
                VisionRotaryEmbedding
            hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            if input_text is None:
                input_text = ["Question: Describe this image. Answer:"
                              ] * self.args.batch_size
            messages = [[{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": raw_image[idx],
                    },
                    {
                        "type": "text",
                        "text": input_text[idx],
                    },
                ],
            }] for idx in range(self.args.batch_size)]

            texts = [
                self.processor.apply_chat_template(msg,
                                                   tokenize=False,
                                                   add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            image = inputs['pixel_values']
            image_grid_thw = inputs['image_grid_thw']
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2],
                image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            seq_length = image.shape[0]
            # Create block indices using bucketing
            block_indices = torch.bucketize(torch.arange(seq_length,
                                                         device=image.device),
                                            cu_seqlens,
                                            right=True) - 1

            # Generate block diagonal mask using matrix expansion
            attention_mask_vit = torch.where(
                block_indices.view(-1, 1) == block_indices.view(1, -1),
                torch.zeros((), device=image.device, dtype=image.dtype),
                torch.full((),
                           torch.finfo(torch.float16).min,
                           device=image.device,
                           dtype=image.dtype)).unsqueeze(0)

            decoder_input_ids = None
            post_prompt = None
            pre_prompt = None
            images_qwenvl = {
                "image": image,
                "input_ids": input_ids,
            }
            rotary_pos_emb = compute_rotary_pos_emb(
                image_grid_thw, hf_config, VisionRotaryEmbedding).to("cuda")
            other_vision_inputs['attention_mask_llm'] = attention_mask
            other_vision_inputs['image_grid_thw'] = image_grid_thw
            other_vision_inputs['attention_mask'] = attention_mask_vit
            other_vision_inputs['rotary_pos_emb'] = rotary_pos_emb
            return input_text, pre_prompt, post_prompt, images_qwenvl, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.args.batch_size
        if not isinstance(input_text, list):
            post_prompt = [post_prompt] * self.args.batch_size
        if image is not None:
            if image.dim() == 5:
                image = image.expand(self.args.batch_size, -1, -1, -1,
                                        -1).contiguous()
            elif image.dim() == 6:
                image = image.expand(self.args.batch_size, -1, -1, -1, -1,
                                        -1).contiguous()
            else:
                if not isinstance(input_text, list):
                    image = image.expand(self.args.batch_size, -1, -1,
                                            -1).contiguous()
                else:
                    image = image.expand(
                        min(self.args.batch_size, len(input_text)), -1, -1,
                        -1).contiguous()
        # Note: For pixtral model, image is a dict with each value being a list of tensors.
        # Moving to device is handled above. So, it's safe to skip this for pixtral.
        if image is not None and 'pixtral' not in self.model_type:
            image = image.to(self.device)
        # Generate decoder_input_ids for enc-dec models
        # Custom prompts can be added as:
        # decoder_input_ids = model.tokenizer(decoder_prompt).input_ids
        if self.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            if "blip2" in self.model_type:
                decoder_start_id = config.text_config.decoder_start_token_id  # T5
            elif "nougat" in self.model_type:
                decoder_start_id = config.decoder.bos_token_id  # Nougat
            else:
                decoder_start_id = config.decoder_start_token_id

            decoder_input_ids = torch.IntTensor([[decoder_start_id]])
            decoder_input_ids = decoder_input_ids.repeat(
                (self.args.batch_size, 1))

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs

    def run(self, input_text, input_image, input_audio, max_new_tokens):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs = self.setup_inputs(
            input_text, input_image, input_audio)
        output_text = self.generate(pre_prompt,
                                    post_prompt,
                                    processed_image,
                                    decoder_input_ids,
                                    max_new_tokens,
                                    other_vision_inputs=other_vision_inputs,
                                    other_audio_inputs=other_audio_inputs,
                                    other_decoder_inputs=other_decoder_inputs)
        return input_text, output_text


def print_result(model, input_text, output_text, args):
    logger.info("---------------------------------------------------------")
    logger.info(f"\n[Q] {input_text}")
    for i in range(len(output_text)):
        logger.info(f"\n[A]: {output_text[i]}")

    if args.num_beams == 1:
        output_ids = model.tokenizer(output_text[0][0],
                                     add_special_tokens=False)['input_ids']
        logger.info(f"Generated {len(output_ids)} tokens")

    if args.check_accuracy:
        # simplified: this runner targets qwen2_vl only
        assert 'dog' in output_text[0][0].lower()

    if args.run_profiling:
        msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(
            name) / args.profiling_iterations
        logger.info('Latencies per batch (msec)')
        logger.info('e2e generation: %.1f' % (msec_per_batch('Generate')))
        logger.info(' ' * 2 + 'Preprocessing: %.1f' %
                    (msec_per_batch('Preprocess')))
        logger.info(' ' * 4 + 'Vision encoder: %.1f' %
                    (msec_per_batch('Vision encoder')))
        if profiler.elapsed_time_in_sec('Feature transform') is not None:
            logger.info(' ' * 4 + 'Feature transform: %.1f' %
                        (msec_per_batch('Feature transform')))
        logger.info(' ' * 2 + 'LLM generate: %.1f' % (msec_per_batch('LLM')))
        logger.info(' ' * 2 + 'Tokenizer decode: %.1f' %
                    (msec_per_batch('Tokenizer decode')))

    logger.info("---------------------------------------------------------")
    
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()
    logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)
    visual_data = model.load_test_data(args.image_path, args.video_path)
    audio_data = None

    if args.run_profiling:
        # Enforce profiling test parameters as requested:
        # warmup: 3, total requests: 10, concurrency: 1 (sequential), batch_size:1,
        # expected output length: 256 tokens. Use downloaded visual_data (list of images).
        logger.info('Running in profiling mode: overriding some args for a controlled perf test')
        args.profiling_iterations = 10
        args.batch_size = 1
        args.max_new_tokens = 256

        num_warmup_iters = 3
        # visual_data is expected to be a list of images for profiling download
        n_images = len(visual_data) if isinstance(visual_data, (list, tuple)) else 1
        for i in range(num_warmup_iters):
            idx = i % max(1, n_images)
            # model.run expects a list of images for qwen2_vl, so pass a length-1 list
            img_for_run = visual_data[idx:idx+1] if isinstance(visual_data, (list, tuple)) else visual_data
            input_text, output_text = model.run(args.input_text, img_for_run,
                                                audio_data, args.max_new_tokens)
        profiler.reset()

    num_iters = args.profiling_iterations if args.run_profiling else 1

    # Collect per-iteration metrics
    vision_secs = []
    llm_secs = []
    tokenizer_secs = []
    total_secs = []
    generated_tokens = []
    ttft_list = []
    throughput_list = []

    # Run num_iters requests sequentially (concurrency=1). If profiling data is a list,
    # cycle through the downloaded images so each request uses one image.
    n_images = len(visual_data) if isinstance(visual_data, (list, tuple)) else 1
    for it in range(num_iters):
        if isinstance(visual_data, (list, tuple)) and n_images > 0:
            idx = it % n_images
            img_for_run = visual_data[idx:idx+1]
        else:
            img_for_run = visual_data

        # Reset profiler to capture per-iteration section timings
        try:
            profiler.reset()
        except Exception:
            pass

        t0 = time.perf_counter()
        input_text, output_text = model.run(args.input_text, img_for_run,
                                            audio_data, args.max_new_tokens)
        t1 = time.perf_counter()

        # Try to read section times from profiler (seconds). If not available, leave as None.
        try:
            v = profiler.elapsed_time_in_sec('Vision encoder')
        except Exception:
            v = None
        try:
            l = profiler.elapsed_time_in_sec('LLM')
        except Exception:
            l = None
        try:
            tok = profiler.elapsed_time_in_sec('Tokenizer decode')
        except Exception:
            tok = None

        total = t1 - t0

        # Compute generated token count for rank 0 only
        tok_count = 0
        if tensorrt_llm.mpi_rank() == 0 and output_text is not None:
            try:
                # output_text is a list of lists: batch x beams
                text0 = output_text[0][0]
                token_ids = model.tokenizer(text0, add_special_tokens=False)['input_ids']
                tok_count = len(token_ids)
            except Exception:
                tok_count = 0

        vision_secs.append(v if v is not None else 0.0)
        tokenizer_secs.append(tok if tok is not None else 0.0)
        total_secs.append(total)
        generated_tokens.append(tok_count)

        streaming_metrics = getattr(model, "last_streaming_metrics", None)
        llm_elapsed = None
        ttft_value = None
        throughput_value = 0.0
        if streaming_metrics:
            llm_start = streaming_metrics.get("llm_start_time")
            llm_end = streaming_metrics.get("llm_end_time")
            if llm_start is not None and llm_end is not None:
                llm_elapsed = llm_end - llm_start
            first_token_time = streaming_metrics.get("first_token_time")
            if first_token_time is not None and llm_start is not None:
                ttft_value = v + max(first_token_time - llm_start, 0.0)
                if llm_end is not None and llm_end > first_token_time and tok_count > 0:
                    throughput_value = tok_count / max(llm_end - first_token_time, 1e-9)

        if llm_elapsed is None:
            llm_elapsed = l if l is not None else total

        if ttft_value is None:
            if tok_count <= 0 or llm_elapsed <= 0:
                ttft_value = v
            else:
                ttft_value = v + (llm_elapsed * (1.0 / max(1, tok_count)))
            throughput_value = tok_count / llm_elapsed if llm_elapsed > 0 else 0.0

        llm_secs.append(llm_elapsed)
        ttft_list.append(ttft_value)
        throughput_list.append(throughput_value)
        
        print_result(model, input_text, output_text, args)

    # After runs, compute statistics and print summarized metrics (only on rank 0)
    if tensorrt_llm.mpi_rank() == 0:
        def stats(arr):
            a = np.array(arr, dtype=float)
            return float(np.mean(a)), float(np.median(a)), float(np.var(a))

        total_mean, total_med, total_var = stats(total_secs)
        ttft_mean, ttft_med, ttft_var = stats(ttft_list)
        thr_mean, thr_med, thr_var = stats(throughput_list)

        logger.info('=== Profiling summary across %d iterations ===' % (num_iters,))
        logger.info('TTFT (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (ttft_mean, ttft_med, ttft_var))
        logger.info('Output Throughput (tokens/s) - \nmean: %.2f, median: %.2f, var: %.4f' % (thr_mean, thr_med, thr_var))
        logger.info('Total Latency (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (total_mean, total_med, total_var))

        # Also log per-section means for visibility
        v_mean, v_med, v_var = stats(vision_secs)
        l_mean, l_med, l_var = stats(llm_secs)
        tok_mean, tok_med, tok_var = stats(tokenizer_secs)
        logger.info('Vision encoder (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (v_mean, v_med, v_var))
        logger.info('LLM generate (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (l_mean, l_med, l_var))
        logger.info('Tokenizer decode (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (tok_mean, tok_med, tok_var))

        # Keep original per-run print of the last result for consistency
        # print_result(model, input_text, output_text, args)

        # TODO: raise error if VILA mode 1 with C++ runtime