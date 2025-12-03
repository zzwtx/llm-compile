# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional

import numpy as np
import tensorrt as trt
import torch

from tensorrt_llm._common import default_net, precision
from _utils import (fp32_array, int32_array, is_same_dtype, trt_dtype_to_np,
                      trt_dtype_to_str)
from functional import (ACT2FN, AllReduceParams, AllReduceFusionParams, AttentionMaskType,
                          Conditional, LayerNormType, PositionEmbeddingType,
                          RopeEmbeddingUtils, RotaryScalingType, Tensor,
                          allgather, arange, bert_attention, cast, clip, concat,
                          constant, embedding, expand, expand_dims, expand_mask,
                          generate_alibi_biases, generate_alibi_slopes,
                          gpt_attention, matmul, generate_logn_scaling, chunk, gather)
from functional import max as fmax
from functional import (minimum, repeat_interleave, shape, slice, softmax,
                          split, unsqueeze, where, squeeze)
# from module import Module
from tensorrt_llm.module import Module
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.quantization.functional import dequantize, quantize
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers.lora import LoraRuntimeParams
from tensorrt_llm.layers.normalization import GroupNorm, LayerNorm, RmsNorm

from functional import maximum  # isort:skip

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}


def make_causal_mask(bsz, tgt_len, past_key_values_length, dtype):
    _range = arange(start=constant(int32_array(0)),
                    end=tgt_len,
                    dtype=trt_dtype_to_str(dtype))
    mask = repeat_interleave(_range, tgt_len, 0).view(concat([tgt_len,
                                                              tgt_len]))
    mask = where(mask < mask.transpose(-1, -2), 1.0, 0.0)

    zero = constant(fp32_array(0))
    zero = expand_dims(zero, [0, 1])
    zero = expand(zero, concat([tgt_len, past_key_values_length]))
    mask = concat([zero, mask], dim=1)
    mask *= np.finfo(trt_dtype_to_np(dtype)).min.item()
    mask = mask.view(concat([1, 1, tgt_len, tgt_len + past_key_values_length]))
    mask = expand(mask,
                  concat([bsz, 1, tgt_len, tgt_len + past_key_values_length]))
    return mask


def compute_relative_bias(query_length,
                          key_length,
                          num_buckets,
                          max_distance,
                          bidirectional,
                          rel_attn_table,
                          tp_size=1,
                          tp_group=None,
                          tp_rank=None):

    def make_relative_position_bucket(relative_position, bidirectional,
                                      num_buckets, max_distance):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += where(relative_position > 0, num_buckets, 0)
            relative_position = relative_position.abs()
        else:
            relative_position = 0 - minimum(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        max_exact_fp = constant(fp32_array(max_exact))
        tmp = cast(relative_position, "float32") / max_exact_fp
        tmp = tmp.log()
        const1 = math.log(max_distance / max_exact)
        const2 = constant(fp32_array(num_buckets - max_exact))
        relative_position_if_large = tmp / const1 * const2
        relative_position_if_large = cast(relative_position_if_large, "int32")
        relative_position_if_large = max_exact + relative_position_if_large
        relative_position_if_large = minimum(relative_position_if_large,
                                             num_buckets - 1)

        relative_buckets += where(is_small, relative_position,
                                  relative_position_if_large)
        return relative_buckets

    context_position = arange(start=constant(int32_array(0)),
                              end=query_length,
                              dtype=trt_dtype_to_str(trt.int32))
    context_position = unsqueeze(context_position, -1)
    memory_position = arange(start=constant(int32_array(0)),
                             end=key_length,
                             dtype=trt_dtype_to_str(trt.int32))
    memory_position = unsqueeze(memory_position, 0)
    relative_position = memory_position - context_position
    relative_position_bucket = make_relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional,
        num_buckets,
        max_distance,
    )
    # shape (query_length, key_length, num_heads)
    values = embedding(relative_position_bucket,
                       rel_attn_table,
                       tp_size=tp_size,
                       tp_group=tp_group,
                       tp_rank=tp_rank)
    # shape (1, num_heads, query_length, key_length)
    values = unsqueeze(values.permute([2, 0, 1]), 0)
    return values


class AttentionParams(object):

    def __init__(self,
                 sequence_length: Tensor = None,
                 context_lengths: Tensor = None,
                 host_context_lengths: Tensor = None,
                 max_context_length: int = None,
                 host_request_types: Tensor = None,
                 encoder_input_lengths: Tensor = None,
                 encoder_max_input_length: Tensor = None,
                 host_runtime_perf_knobs: Tensor = None,
                 host_context_progress: Tensor = None):
        self.sequence_length = sequence_length
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        # max allowed context length. Required to
        # compute scratch memory size.
        self.max_context_length = max_context_length
        self.host_request_types = host_request_types

        self.encoder_input_lengths = encoder_input_lengths
        self.encoder_max_input_length = encoder_max_input_length

        self.host_runtime_perf_knobs = host_runtime_perf_knobs

        self.host_context_progress = host_context_progress

        # const parameters that will be reused by all layers.
        self.embed_positions = None
        self.rotary_inv_freq = None
        self.embed_positions_for_gpt_attention = None

        # auxiliary params to support models with non-homegeneous attn layers requiring
        # a different set of rope params. e.g. Gemma3.
        self.embed_positions_local = None
        self.rotary_inv_freq_local = None
        self.embed_positions_for_gpt_attention_local = None

        # long rope const parameters
        self.long_rope_embed_positions = None
        self.long_rope_rotary_inv_freq = None
        self.long_rope_embed_positions_for_gpt_attention = None
        self.short_mscale = 1.0
        self.long_mscale = 1.0

    def fill_attention_const_params_for_rope(
            self,
            embed_positions: Tensor = None,
            rotary_inv_freq: Tensor = None,
            embed_positions_for_gpt_attention: Tensor = None,
            embed_positions_local: Tensor = None,
            rotary_inv_freq_local: Tensor = None,
            embed_positions_for_gpt_attention_local: Tensor = None):
        self.embed_positions = embed_positions
        self.rotary_inv_freq = rotary_inv_freq
        self.embed_positions_for_gpt_attention = embed_positions_for_gpt_attention
        self.embed_positions_local = embed_positions_local
        self.rotary_inv_freq_local = rotary_inv_freq_local
        self.embed_positions_for_gpt_attention_local = embed_positions_for_gpt_attention_local
        return self

    def fill_attention_const_params_for_long_rope(
            self, embed_positions, long_rope_embed_positions, rotary_inv_freq,
            long_rope_rotary_inv_freq, embed_positions_for_gpt_attention,
            long_rope_embed_positions_for_gpt_attention, short_mscale,
            long_mscale):
        self.embed_positions = embed_positions
        self.long_rope_embed_positions = long_rope_embed_positions
        self.rotary_inv_freq = rotary_inv_freq
        self.long_rope_rotary_inv_freq = long_rope_rotary_inv_freq
        self.embed_positions_for_gpt_attention = embed_positions_for_gpt_attention
        self.long_rope_embed_positions_for_gpt_attention = long_rope_embed_positions_for_gpt_attention
        self.short_mscale = short_mscale
        self.long_mscale = long_mscale
        return self

    def is_valid_cross_attn(self, do_cross_attention):
        if do_cross_attention:
            if self.encoder_input_lengths is None:
                return False
            if self.encoder_max_input_length is None:
                return False
        return True

    def is_valid(self, gpt_attention_plugin, remove_input_padding,
                 use_kv_cache):
        if gpt_attention_plugin:
            if use_kv_cache and self.sequence_length is None:
                return False
            if self.context_lengths is None:
                return False
            if self.host_request_types is None:
                return False
            if self.max_context_length is None:
                return False
            if self.host_runtime_perf_knobs is None:
                return False
            # if self.host_context_progress is None:
            #     return False

        if remove_input_padding:
            if self.host_context_lengths is None:
                return False
            if not gpt_attention_plugin:
                return False

        return True

class SpecDecodingParams:

    def __init__(self,
                 spec_decoding_is_generation_length_variable: bool = False,
                 spec_decoding_max_generation_length: int = 1,
                 spec_decoding_generation_lengths: Tensor = None,
                 spec_decoding_position_offsets: Tensor = None,
                 spec_decoding_packed_mask: Tensor = None):

        self.spec_decoding_is_generation_length_variable = spec_decoding_is_generation_length_variable
        self.spec_decoding_max_generation_length = spec_decoding_max_generation_length
        self.spec_decoding_generation_lengths = spec_decoding_generation_lengths
        self.spec_decoding_position_offsets = spec_decoding_position_offsets
        self.spec_decoding_packed_mask = spec_decoding_packed_mask


class KeyValueCacheParams:

    def __init__(self,
                 past_key_value: List[Tensor] = None,
                 host_past_key_value_lengths: Tensor = None,
                 host_max_attention_window_sizes: Tensor = None,
                 host_sink_token_length: Tensor = None,
                 kv_cache_block_offsets: Tensor = None,
                 host_kv_cache_block_offsets: Tensor = None,
                 host_kv_cache_pool_pointers: Tensor = None,
                 cache_indirection: Tensor = None,
                 past_key_value_length: Tensor = None,
                 cross_kv_cache_block_offsets: Tensor = None,
                 host_cross_kv_cache_block_offsets: Tensor = None,
                 host_cross_kv_cache_pool_pointers: Tensor = None):
        self.past_key_value = past_key_value
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.host_max_attention_window_sizes = host_max_attention_window_sizes
        self.host_sink_token_length = host_sink_token_length
        self.kv_cache_block_offsets = kv_cache_block_offsets
        self.host_kv_cache_block_offsets = host_kv_cache_block_offsets
        self.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        self.cross_kv_cache_block_offsets = cross_kv_cache_block_offsets
        self.host_cross_kv_cache_block_offsets = host_cross_kv_cache_block_offsets
        self.host_cross_kv_cache_pool_pointers = host_cross_kv_cache_pool_pointers
        self.cache_indirection = cache_indirection
        # self.past_key_value_length = past_key_value_length

    def get_first_past_key_value(self):
        if self.past_key_value is None:
            return None
        return self.past_key_value[0]

    def fill_none_tensor_list(self, list_size):
        if self.past_key_value is None:
            self.past_key_value = tuple([None] * list_size)

    def is_valid(self, gpt_attention_plugin):
        if gpt_attention_plugin:
            if self.host_past_key_value_lengths is None:
                return False
            if self.host_max_attention_window_sizes is None:
                return False
            if self.host_sink_token_length is None:
                return False
            if self.cache_indirection is None:
                return False

        return True


class BlockSparseAttnParams:

    def __init__(self,
                 block_size: int = 64,
                 homo_head_pattern: bool = False,
                 num_local_blocks: int = 16,
                 vertical_stride: int = 8):
        self.block_size = block_size
        self.homo_head_pattern = homo_head_pattern
        self.num_local_blocks = num_local_blocks
        self.vertical_stride = vertical_stride


class MropeParams:
    def __init__(
        self,
        mrope_rotary_cos_sin: Tensor = None,
        mrope_position_deltas: Tensor = None,
    ):
        self.mrope_rotary_cos_sin = mrope_rotary_cos_sin
        self.mrope_position_deltas = mrope_position_deltas


class Attention(Module):

    def __init__(self,
                 *,
                 local_layer_idx,
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads=None,
                 max_position_embeddings=1024,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_head_size=None,
                 qk_layernorm=False,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_share=True,
                 inner_layernorm=False,
                 eps=1e-05,
                 attention_mask_type=AttentionMaskType.padding,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_base=10000.0,
                 rotary_embedding_base_local=1.0,
                 rotary_embedding_scaling=None,
                 rotary_embedding_percentage=1.0,
                 rope_scaling_short_factors=None,
                 rope_scaling_long_factors=None,
                 rope_scaling_short_mscale=None,
                 rope_scaling_long_mscale=None,
                 original_max_position_embeddings=1024,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 quant_mode: QuantMode = QuantMode(0),
                 q_scaling=1.0,
                 cross_attention=False,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 dense_bias=None,
                 clip_qkv=None,
                 alibi_bias_max=8,
                 skip_cross_kv=False,
                 max_attn_value=0.0,
                 block_sparse_params=None,
                 use_implicit_relative_attention=False,
                 reorder=False,
                 enable_qkv=True,
                 cp_group=[0],
                 cp_size=1,
                 cp_rank=0,
                 max_seqlen_for_logn_scaling=8192,
                 use_logn_scaling=False,
                 is_local=False):
        super().__init__()

        self.local_layer_idx = local_layer_idx
        self.cross_attention = cross_attention
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        assert num_attention_heads % tp_size == 0, \
        "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.bias = bias
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.is_local = is_local

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = position_embedding_type == PositionEmbeddingType.alibi_with_scale
        self.alibi_bias_max = alibi_bias_max
        self.position_embedding_type = position_embedding_type

        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.num_buckets = num_buckets
        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_base_local = rotary_embedding_base_local
        self.rotary_embedding_scaling = rotary_embedding_scaling
        self.rotary_embedding_scale_type = RotaryScalingType.none
        self.rotary_embedding_scale = 1.0
        self.short_mscale = 1.0
        self.long_mscale = 1.0
        self.rotary_embedding_percentage = rotary_embedding_percentage
        self.use_implicit_relative_attention = self.relative_attention and use_implicit_relative_attention
        self.max_seqlen_for_logn_scaling = max_seqlen_for_logn_scaling
        self.use_logn_scaling = use_logn_scaling
        if rotary_embedding_scaling is not None:
            rotary_scaling_type = rotary_embedding_scaling.get(
                "type", rotary_embedding_scaling.get("rope_type"))
            self.rotary_embedding_scale_type = RotaryScalingType.from_string(
                rotary_scaling_type)

            self.rotary_embedding_scale = rotary_embedding_scaling.get(
                "factor", 1.0)

        self.rotary_embedding_dim = 0
        if self.position_embedding_type.is_rope():
            # print("Rope Wrong!!!")
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
        elif self.position_embedding_type.is_alibi():
            alibi_scale = 1. / self.norm_factor if self.scale_alibi_bias else 1.
            alibi_slopes = generate_alibi_slopes(
                self.num_attention_heads * self.tp_size,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                alibi_scale=alibi_scale,
                alibi_bias_max=self.alibi_bias_max)
            self.register_parameter(
                'alibi_slopes',
                Parameter(alibi_slopes, dtype='float32', is_buffer=True))

        if self.use_logn_scaling:
            logn_scaling = generate_logn_scaling(
                self.max_seqlen_for_logn_scaling, self.max_position_embeddings)
            self.register_parameter(
                'logn_scaling',
                Parameter(logn_scaling, dtype='float32', is_buffer=True))

        self.quant_mode = quant_mode
        self.max_attn_value = max_attn_value
        self.register_parameter('kv_cache_scaling_factor', None)
        self.register_parameter('attention_output_orig_quant_scale', None)
        self.register_parameter('attention_output_sf_scale', None)

        self.block_sparse_params = block_sparse_params if block_sparse_params is not None else BlockSparseAttnParams(
        )

        # The output feature size is therefore (h/tp + 2*kvh/tp) * d, where h is num_heads,
        # d is head_size, kvh is the num_kv_heads and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*kvh) * d / tp,
        # which matches the desired output size (h/tp + 2*kvh/tp) * d after splitting

        # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
        # example: d_model != num_heads * head_size in Flan-T5/ByT5/Gemma
        if enable_qkv:
            self.qkv = ColumnLinear(
                hidden_size,
                tp_size * self.num_attention_heads * self.attention_head_size +
                (2 * tp_size * self.num_attention_kv_heads *
                 self.attention_head_size),
                bias=bias,
                dtype=dtype,
                tp_group=tp_group,
                tp_size=tp_size,
                gather_output=False,
                is_qkv=True)
        self.dense = RowLinear(tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               hidden_size,
                               bias=self.dense_bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

        # see optimize_model's add_lora for LoRA initialization
        self.qkv_lora = None
        self.qkv_dora = None

        # per-layer relative attention table
        if self.use_implicit_relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

        # qk layernorm
        self.qk_layernorm = qk_layernorm
        self.layernorm_type = layernorm_type
        self.layernorm_share = layernorm_share
        ln_type = layernorm_map[layernorm_type]
        if self.qk_layernorm:
            # layernorm_share indicates whether all the QK head in one layer shares the same norm parameters or not
            if layernorm_share:
                self.q_layernorm = ln_type(self.attention_head_size,
                                           eps=eps,
                                           dtype=dtype)
                self.k_layernorm = ln_type(self.attention_head_size,
                                           eps=eps,
                                           dtype=dtype)
            else:
                assert ln_type == LayerNorm
                self.q_layernorm = ln_type(
                    (self.num_attention_heads, self.attention_head_size),
                    eps=eps,
                    dtype=dtype,
                    bias=False,
                    tp_size=tp_size,
                    tp_dim=0)
                self.k_layernorm = ln_type(
                    (self.num_attention_kv_heads, self.attention_head_size),
                    eps=eps,
                    dtype=dtype,
                    bias=False,
                    tp_size=tp_size,
                    tp_dim=0)

        self.inner_layernorm = ln_type(self.hidden_size, dtype=dtype,
                                       eps=eps) if inner_layernorm else None
        if clip_qkv is not None:
            self.clip_qkv = fp32_array([clip_qkv])
        else:
            self.clip_qkv = None

        self.skip_cross_kv = skip_cross_kv

        if self.position_embedding_type.is_mrope():
            inv_freq, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                num_pos=self.max_position_embeddings,
                dim=self.attention_head_size,
                theta=float(self.rotary_embedding_base),
                scale_type=RotaryScalingType.mrope)
            
            self.register_parameter(
                'mrope_rotary_cos_sin',
                Parameter(rotary_cos_sin, dtype='float32', is_buffer=True))

    @staticmethod
    def create_attention_const_params(model_cls, config):
        # get rotary parameters.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_head_size = config.head_size
        max_position_embeddings = config.max_position_embeddings
        position_embedding_type = config.position_embedding_type
        rotary_embedding_base = getattr(config, 'rotary_base', 10000.0)
        rotary_embedding_scaling = getattr(config, 'rotary_scaling', None)
        rotary_embedding_percentage = getattr(config, 'rotary_pct', 1.0)
        # only rope need the const parameters.
        if not position_embedding_type.is_rope():
            return
        # attention head size
        attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        # rotary embedding dim.
        rotary_embedding_dim = getattr(
            config, 'rotary_dim',
            int(attention_head_size * rotary_embedding_percentage))
        # rotary scaling.
        rotary_embedding_scale_type = RotaryScalingType.none
        rotary_embedding_scale = 1.0
        if rotary_embedding_scaling is not None:
            rotary_scaling_type = rotary_embedding_scaling.get(
                "type", rotary_embedding_scaling.get("rope_type"))
            rotary_embedding_scale_type = RotaryScalingType.from_string(
                rotary_scaling_type)
            rotary_embedding_scale = rotary_embedding_scaling.get("factor", 1.0)

        if position_embedding_type == PositionEmbeddingType.long_rope:
            rope_scaling_short_factors, rope_scaling_long_factors = None, None
            rope_scaling_short_mscale, rope_scaling_long_mscale = None, None
            original_max_position_embeddings = max_position_embeddings

            if hasattr(config, "longrope_scaling_short_factors"):
                rope_scaling_short_factors = np.asarray(
                    config.longrope_scaling_short_factors).astype(np.float32)
                rope_scaling_long_factors = np.asarray(
                    config.longrope_scaling_long_factors).astype(np.float32)

                original_max_position_embeddings = config.original_max_position_embeddings

                if config.architecture == "Phi3SmallForCausalLM" or config.architecture == "PhiMoEForCausalLM":
                    rope_scaling_short_mscale = config.longrope_short_mscale
                    rope_scaling_long_mscale = config.longrope_long_mscale

                embed_positions, long_rope_embed_positions, \
                (rotary_inv_freq, embed_positions_for_gpt_attention), \
                (long_rope_rotary_inv_freq, long_rope_embed_positions_for_gpt_attention), mscale \
                    = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope_for_attention_plugin(
                    max_position_embeddings,
                    original_max_position_embeddings, rotary_embedding_dim,
                    rotary_embedding_base, rope_scaling_short_factors,
                    rope_scaling_long_factors, rope_scaling_short_mscale, rope_scaling_long_mscale)

                if rope_scaling_short_mscale is not None:
                    assert rope_scaling_long_mscale is not None
                    short_mscale = rope_scaling_short_mscale
                    long_mscale = rope_scaling_long_mscale
                else:
                    short_mscale = long_mscale = mscale

                model_cls.register_parameter(
                    'embed_positions',
                    Parameter(embed_positions, dtype='float32', is_buffer=True))
                model_cls.register_parameter(
                    'long_rope_embed_positions',
                    Parameter(long_rope_embed_positions,
                              dtype='float32',
                              is_buffer=True))
                model_cls.register_parameter(
                    'rotary_inv_freq',
                    Parameter(rotary_inv_freq, dtype='float32', is_buffer=True))
                model_cls.register_parameter(
                    'long_rope_rotary_inv_freq',
                    Parameter(long_rope_rotary_inv_freq,
                              dtype='float32',
                              is_buffer=True))
                model_cls.register_parameter(
                    'embed_positions_for_gpt_attention',
                    Parameter(embed_positions_for_gpt_attention,
                              dtype='float32',
                              is_buffer=True))
                model_cls.register_parameter(
                    'long_rope_embed_positions_for_gpt_attention',
                    Parameter(long_rope_embed_positions_for_gpt_attention,
                              dtype='float32',
                              is_buffer=True))
                model_cls.short_mscale = short_mscale
                model_cls.long_mscale = long_mscale
        elif rotary_embedding_scale_type == RotaryScalingType.yarn:
            beta_fast = rotary_embedding_scaling.get("beta_fast", 32.0)
            beta_slow = rotary_embedding_scaling.get("beta_slow", 1.0)
            mscale = rotary_embedding_scaling.get("mscale", 1.0)
            mscale_all_dim = rotary_embedding_scaling.get("mscale_all_dim", 0.0)
            original_max_position_embeddings = rotary_embedding_scaling.get(
                "original_max_position_embeddings", 4096)
            rotary_inv_freq, embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                max_position_embeddings, rotary_embedding_dim,
                rotary_embedding_base, rotary_embedding_scale,
                original_max_position_embeddings, beta_fast, beta_slow, mscale,
                mscale_all_dim, False)

            embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions(
                max_position_embeddings,
                rotary_embedding_dim,
            )
            model_cls.register_parameter(
                'embed_positions',
                Parameter(embed_positions, dtype='float32', is_buffer=True))
            model_cls.register_parameter(
                'rotary_inv_freq',
                Parameter(rotary_inv_freq, dtype='float32', is_buffer=True))
            model_cls.register_parameter(
                'embed_positions_for_gpt_attention',
                Parameter(embed_positions_for_gpt_attention,
                          dtype='float32',
                          is_buffer=True))
        else:

            def register_rope_params(rotary_base, names_to_register):
                # Rotary const weights.
                embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions(
                    max_position_embeddings,
                    rotary_embedding_dim,
                )
                rotary_inv_freq, embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                    max_position_embeddings, rotary_embedding_dim, rotary_base,
                    rotary_embedding_scale, rotary_embedding_scale_type,
                    rotary_embedding_scaling)
                model_cls.register_parameter(
                    names_to_register[0],
                    Parameter(embed_positions, dtype='float32', is_buffer=True))
                model_cls.register_parameter(
                    names_to_register[1],
                    Parameter(rotary_inv_freq, dtype='float32', is_buffer=True))
                model_cls.register_parameter(
                    names_to_register[2],
                    Parameter(embed_positions_for_gpt_attention,
                              dtype='float32',
                              is_buffer=True))

            register_rope_params(rotary_base=rotary_embedding_base,
                                 names_to_register=[
                                     'embed_positions', 'rotary_inv_freq',
                                     'embed_positions_for_gpt_attention'
                                 ])

            # For models with non-homegeneous attention layers requiring a second set of rope params. e.g. Gemma3.
            rotary_embedding_base_local = getattr(config,
                                                  'rope_local_base_freq', None)
            if rotary_embedding_base_local is not None:
                register_rope_params(
                    rotary_base=rotary_embedding_base_local,
                    names_to_register=[
                        'embed_positions_local', 'rotary_inv_freq_local',
                        'embed_positions_for_gpt_attention_local'
                    ])

    @staticmethod
    def fill_attention_params(model_cls, attention_params):
        if model_cls.position_embedding_type.is_rope():
            if attention_params is None:
                attention_params = AttentionParams()
            else:
                print("Warning: attention_params is not None. Overwriting the rope parameters.")
            if model_cls.position_embedding_type == PositionEmbeddingType.long_rope:
                return attention_params.fill_attention_const_params_for_long_rope(
                    model_cls.embed_positions.value,
                    model_cls.long_rope_embed_positions.value,
                    model_cls.rotary_inv_freq.value,
                    model_cls.long_rope_rotary_inv_freq.value,
                    model_cls.embed_positions_for_gpt_attention.value,
                    model_cls.long_rope_embed_positions_for_gpt_attention.value,
                    model_cls.short_mscale, model_cls.long_mscale)
            else:
                return attention_params.fill_attention_const_params_for_rope(
                    model_cls.embed_positions.value,
                    model_cls.rotary_inv_freq.value,
                    model_cls.embed_positions_for_gpt_attention.value,
                    model_cls.embed_positions_local.value if hasattr(
                        model_cls, "embed_positions_local") else None,
                    model_cls.rotary_inv_freq_local.value if hasattr(
                        model_cls, "rotary_inv_freq_local") else None,
                    model_cls.embed_positions_for_gpt_attention_local.value
                    if hasattr(
                        model_cls,
                        "embed_positions_for_gpt_attention_local") else None)
        # Fill nothing.
        return attention_params

    def _get_output_orig_quant_scale(self):
        attention_output_orig_quant_scale = self.attention_output_orig_quant_scale.value if self.attention_output_orig_quant_scale is not None else None
        if attention_output_orig_quant_scale is not None and (
                default_net().plugin_config.gemm_plugin == 'nvfp4'
                or self.quant_mode.has_nvfp4()):
            # The scale was intended for nvfp4 quantization: max_value * scale = fp4_max * fp8_max
            # So if we want to quantize the output to fp8, the scale should be divided by fp4_max
            attention_output_orig_quant_scale = attention_output_orig_quant_scale / 6.0
        return attention_output_orig_quant_scale

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        attention_packed_mask=None,
        use_cache=False,
        spec_decoding_params=None,
        mrope_params=None,
        kv_cache_params=None,
        attention_params=None,
        encoder_output: Optional[Tensor] = None,
        position_embedding=None,
        norm_before_bmm1=False,
        lora_layer_params=None,
        position_ids=None,
        cross_kv_cache_gen: Optional[Tensor] = None,
        cross_kv_reuse: Optional[Tensor] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        skip_attn=None,
    ):
        attention_input = hidden_states

        assert isinstance(hidden_states, (Tensor, tuple))

        spec_decoding_params = SpecDecodingParams(
        ) if spec_decoding_params is None else spec_decoding_params

        print(mrope_params is not None)
        mrope_params = MropeParams() if mrope_params is None else mrope_params
        logn_scaling = None
        if self.use_logn_scaling:
            logn_scaling = self.logn_scaling.value

        alibi_slopes = None
        if self.position_embedding_type.is_alibi():
            alibi_slopes = self.alibi_slopes.value
            if default_net().plugin_config.gpt_attention_plugin:
                alibi_slopes = cast(alibi_slopes, hidden_states.dtype)

        qkv_lora_params = None
        if lora_layer_params is not None:
            if not self.cross_attention:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_qkv")
            else:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_qkv")

        unfuse_qkv_gemm = self.qkv is None
        if unfuse_qkv_gemm:
            qkv_gemm = [self.q, self.k, self.v]
            qkv = [gemm(hidden_states) for gemm in qkv_gemm]
            if default_net(
            ).plugin_config.lora_plugin and qkv_lora_params is not None:
                lora = self.qkv.lora(hidden_states, qkv_lora_params)
                kv_size = self.attention_head_size * self.num_attention_kv_heads
                qkv_lora = split(lora,
                                 [self.attention_hidden_size, kv_size, kv_size],
                                 dim=1)
                qkv = [tensor + lora for tensor, lora in zip(qkv, qkv_lora)]
        else:
            qkv = self.qkv(hidden_states, qkv_lora_params)

        if self.clip_qkv is not None:
            qkv = clip(qkv, -self.clip_qkv, self.clip_qkv)

        if default_net().plugin_config.remove_input_padding:
            if unfuse_qkv_gemm:
                for tensor in qkv:
                    assert tensor.ndim() == 2
            else:
                print(f"DEBUG: Line 903 check: qkv.ndim={qkv.ndim()}, shape={qkv.shape}")
                # assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            if not self.cross_attention:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_v")
            else:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_runtime_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths,
                    max_encoder_context_length=q_lora_params.
                    max_encoder_context_length,
                    host_encoder_input_lengths=q_lora_params.
                    host_encoder_input_lengths,
                    partial_lora_mask=lora_layer_params.partial_lora_mask,
                )

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_runtime_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora
                if self.qkv_dora is not None:
                    qkv = self.qkv_dora(qkv, qkv_lora_runtime_params)
        if self.qk_layernorm:
            base_shape = shape(qkv, 0) if qkv.ndim() == 2 else concat(
                [shape(qkv, 0), shape(qkv, 1)])
            qkv_sections = [
                self.num_attention_heads, self.num_attention_kv_heads,
                self.num_attention_kv_heads
            ]
            total_heads = sum(qkv_sections)
            if self.num_attention_heads != self.num_attention_kv_heads:
                qkv = qkv.view(
                    concat([base_shape, total_heads, self.attention_head_size]))
                query, key, value = split(qkv, qkv_sections, dim=qkv.ndim() - 2)
            else:
                qkv = qkv.view(
                    concat([
                        base_shape, self.num_attention_heads, 3,
                        self.attention_head_size
                    ]))
                query, key, value = split(qkv, 1, dim=qkv.ndim() - 2)
                q_shape = concat([
                    base_shape, self.num_attention_heads,
                    self.attention_head_size
                ])
                query = query.view(q_shape)
                key = key.view(q_shape)
                value = value.view(q_shape)

            normalized_shape = None
            if not self.layernorm_share:
                normalized_shape = self.attention_head_size
            query = self.q_layernorm(query, normalized_shape=normalized_shape)
            key = self.k_layernorm(key, normalized_shape=normalized_shape)
            qkv = concat([query, key, value], dim=query.ndim() - 2)
            qkv = qkv.view(
                concat([base_shape, total_heads * self.attention_head_size]))
        if self.position_embedding_type == PositionEmbeddingType.chatglm:
            qkv = RopeEmbeddingUtils.apply_rotary_pos_emb_chatglm(
                qkv,
                position_embedding,
                self.num_attention_heads,
                self.attention_head_size,
                self.max_position_embeddings,
                self.rotary_embedding_scale,
                default_net().plugin_config.remove_input_padding,
            )
            self.rotary_embedding_scale_type = RotaryScalingType.none
            self.rotary_embedding_scale = 1.0

        paged_kv_cache = default_net().plugin_config.paged_kv_cache

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding, use_cache)

        if use_cache:
            assert kv_cache_params is None or kv_cache_params.is_valid(
                default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )

        # if cross attention, cross QKV only needs to be calculated once in the
        # 1st decoding step --> write to cross KV cache --> remains constant
        # during the entire decoding steps.
        # 1st and >1st steps are distinguished by a boolean tensor `cross_kv_cache_gen` passed at runtime
        # also, cross KV cache max length is set from encoder output seqlen,
        # this maps to the max context length concept in decoder-only models
        cross_kv = None
        if self.cross_attention and encoder_output:
            assert isinstance(encoder_output, Tensor)

            def compute_cross_kv(encoder_output):
                if hasattr(self, 'kv'):
                    # We optimize the graph by adding kv in the cross attention layer, preventing computing the
                    # query of encoder_output.
                    assert qkv_lora_params is None, "Not support LoRA when we only compute key/value in cross atteniton"
                    # see optimization_model's optimize_cross_qkv
                    cross_kv = self.kv(encoder_output, qkv_lora_params)
                    base_shape = shape(
                        cross_kv, 0) if cross_kv.ndim() == 2 else concat(
                            [shape(cross_kv, 0),
                             shape(cross_kv, 1)])
                    if self.qk_layernorm:
                        cross_kv = cross_kv.view(
                            concat([
                                base_shape, 2 * self.num_attention_kv_heads,
                                self.attention_head_size
                            ]))

                        key, value = split(cross_kv, [
                            self.num_attention_kv_heads,
                            self.num_attention_kv_heads
                        ],
                                           dim=cross_kv.ndim() - 2)

                        key = self.k_layernorm(key)
                        cross_kv = concat([key, value], dim=key.ndim() - 2)
                else:
                    cross_qkv = self.qkv(encoder_output, qkv_lora_params)
                    base_shape = shape(
                        cross_qkv, 0) if cross_qkv.ndim() == 2 else concat(
                            [shape(cross_qkv, 0),
                             shape(cross_qkv, 1)])

                    cross_qkv = cross_qkv.view(
                        concat([
                            base_shape, self.num_attention_heads +
                            2 * self.num_attention_kv_heads,
                            self.attention_head_size
                        ]))

                    if self.qk_layernorm:
                        _, key, value = split(cross_qkv, [
                            self.num_attention_heads,
                            self.num_attention_kv_heads,
                            self.num_attention_kv_heads
                        ],
                                              dim=cross_qkv.ndim() - 2)

                        key = self.k_layernorm(key)
                        cross_kv = concat([key, value], dim=key.ndim() - 2)
                    else:
                        _, cross_kv = split(cross_qkv, [
                            self.num_attention_heads,
                            self.num_attention_kv_heads * 2
                        ],
                                            dim=cross_qkv.ndim() - 2)
                cross_kv = cross_kv.view(
                    concat([
                        base_shape, 2 * self.num_attention_kv_heads *
                        self.attention_head_size
                    ]))

                if default_net(
                ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
                    _, cross_k_lora, cross_v_lora = self.qkv_lora(
                        encoder_output,
                        qkv_lora_runtime_params,
                        is_cross_attention=True)
                    cross_kv_lora = concat([cross_k_lora, cross_v_lora],
                                           dim=cross_k_lora.rank() - 1)
                    cross_kv = cross_kv + cross_kv_lora
                    if self.qkv_dora is not None:
                        cross_kv = self.qkv_dora(cross_kv,
                                                 qkv_lora_runtime_params,
                                                 is_cross_attention=True)

                return cross_kv

            if self.skip_cross_kv:
                conditional = Conditional(cross_kv_cache_gen)
                cond_in1 = conditional.add_input(encoder_output)
                cond_in2 = conditional.add_input(cross_kv_reuse)

                ## True branch: context phase, compute cross qkv
                cross_kv_true = compute_cross_kv(cond_in1)

                ## False branch: generation phase, no compute but need to obey shape constraints
                # because TRT's IfConditional requires the output shape of two subgraphs to be identical
                # our 1st attempt was to stack encoder_output [B, S, H] or [N, H] --> cross qkv [B, S, 3*H] or [N, 3*H],
                # but it still introduces unnecessary concat. A better solution is to create a dummy torch tensor `cross_kv_resue`
                # with the correct shape and reuse it in every generation step
                cross_kv_false = cond_in2
                cross_kv = conditional.add_output(cross_kv_true, cross_kv_false)
            else:
                cross_kv = compute_cross_kv(encoder_output)

        if default_net().plugin_config.gpt_attention_plugin:
            if self.cross_attention and (past_key_value is not None):
                past_key_value = kv_cache_params.past_key_value[1]
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm,
                AttentionMaskType.blocksparse
            ], 'Plugin only support masked MHA.'

            # KV cache scales.
            if self.kv_cache_scaling_factor is not None:
                kv_orig_quant_scale = self.kv_cache_rcp_scaling_factor.value
                kv_quant_orig_scale = self.kv_cache_scaling_factor.value
            else:
                kv_orig_quant_scale = None
                kv_quant_orig_scale = None

            # The output SF scale, needed when fuse_fp4_quant is enabled.
            attention_output_sf_scale = self.attention_output_sf_scale.value if self.attention_output_sf_scale is not None else None

            # The rotary inv freq can be pre-computed.
            rotary_inv_freq = getattr(attention_params, "rotary_inv_freq", None)
            # Rotary cos/sin cache.
            rotary_cos_sin = getattr(attention_params,
                                     "embed_positions_for_gpt_attention", None)
            rotary_inv_freq_local = getattr(attention_params,
                                            "rotary_inv_freq_local", None)
            rotary_cos_sin_local = getattr(
                attention_params, "embed_positions_for_gpt_attention_local",
                None)

            long_rope_rotary_inv_freq = getattr(attention_params,
                                                "long_rope_rotary_inv_freq",
                                                None)
            long_rope_rotary_cos_sin = getattr(
                attention_params, "long_rope_embed_positions_for_gpt_attention",
                None)

            if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
                rotary_inv_freq = None
                rotary_cos_sin = None

            # check if the cache is provided.
            if self.position_embedding_type.is_rope():
                assert (rotary_inv_freq is not None) and (
                    rotary_cos_sin is not None
                ), "rotary_inv_freq and embed_positions_for_gpt_attention must be provided."
            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                assert long_rope_rotary_inv_freq is not None
                assert long_rope_rotary_cos_sin is not None
                
            rotary_embedding_dim = self.rotary_embedding_dim
            position_embedding_type = self.position_embedding_type
            mrope_rotary_cos_sin_arg = mrope_params.mrope_rotary_cos_sin
            mrope_position_deltas_arg = mrope_params.mrope_position_deltas

            if self.position_embedding_type.is_mrope():
                if not isinstance(qkv, list):
                    # ==========================================================================================
                    # 手动实现mRoPE（详细说明）
                    # ==========================================================================================
                    # TRT-LLM 0.12.0中的GPTAttention插件可能无法完全支持Qwen2-VL所需的特定mRoPE模式， 
                    #（多模态旋转位置嵌入）模式，或我们绕过该插件以在Python中手动处理复杂旋转逻辑。
                    #
                    # 该过程包括：
                    # 1. 将融合的QKV张量拆分为查询(Q)、键(K)和值(V)。
                    # 2. 重塑 Q 和 K 以分离 ‘heads’ 维度，因 RoPE 需按每个头部单独应用。
                    # 3. 根据位置 ID 检索预计算的 Cos/Sin 值。
                    # 4. 执行旋转变换：x_rotated = x * cos + rotate_half(x) * sin。
                    # 5. 将Q、K、V重新拼接为单一QKV张量。
                    # 6. 配置下游GPTAttention插件跳过其内部RoPE处理，
                    #    因我们已完成该操作。
                    # ==========================================================================================

                    head_size = self.attention_head_size
                    num_heads = self.num_attention_heads
                    num_kv_heads = self.num_attention_kv_heads
                    
                    q_size = num_heads * head_size
                    k_size = num_kv_heads * head_size
                    v_size = num_kv_heads * head_size
                    
                    # 1. Split QKV
                    # qkv shape: [num_tokens, (num_heads + 2*num_kv_heads) * head_size] (if remove_input_padding)
                    q, k, v = split(qkv, [q_size, k_size, v_size], dim=-1)
                    
                    # Save original dimensions (shape(...) returns a Tensor; don't index it)
                    batch_or_tokens = shape(q, 0)
                    # When remove_input_padding is enabled, q may be 2D: [num_tokens, hidden]
                    # Otherwise q is 3D: [batch, seq_len, hidden]
                    ndim = q.ndim()

                    # 2. Reshape for RoPE
                    # We need to view the tensor as [..., num_heads, head_size] to apply rotation to the head_size dim.
                    if ndim == 2:
                        # Case: remove_input_padding=True (Packed)
                        # Shape: [num_tokens, num_heads, head_size]
                        q = q.view(concat([batch_or_tokens, num_heads, head_size]))
                        k = k.view(concat([batch_or_tokens, num_kv_heads, head_size]))
                    else:
                        # Case: Padded batch
                        # Shape: [batch, seq_len, num_heads, head_size]
                        seq_len = shape(q, 1)
                        q = q.view(concat([batch_or_tokens, seq_len, num_heads, head_size]))
                        k = k.view(concat([batch_or_tokens, seq_len, num_kv_heads, head_size]))
                    
                    # 3. Prepare Cos/Sin
                    # Use the static constant
                    if mrope_rotary_cos_sin_arg is not None:
                        cos_sin_table = mrope_rotary_cos_sin_arg
                    else:
                        cos_sin_table = self.mrope_rotary_cos_sin.value
                    
                    head_size = self.attention_head_size
                    
                    # cos_sin_table is [max_pos, head_size].
                    # We need to separate cos and sin.
                    cos_sin_table = cos_sin_table.view([self.max_position_embeddings, head_size // 2, 2])
                    
                    # Extract cos and sin
                    cos_table = slice(cos_sin_table, [0, 0, 0], [self.max_position_embeddings, head_size // 2, 1])
                    cos_table = cos_table.view([self.max_position_embeddings, head_size // 2])
                    
                    sin_table = slice(cos_sin_table, [0, 0, 1], [self.max_position_embeddings, head_size // 2, 1])
                    sin_table = sin_table.view([self.max_position_embeddings, head_size // 2])
                    
                    # Gather
                    pos_ids = cast(position_ids, 'int32')
                    
                    # position_ids is [3, batch, seq_len] or [3, num_tokens]
                    # cos_table is [max_pos, head_size/2]
                    # embedding output: [3, batch, seq_len, head_size/2] or [3, num_tokens, head_size/2]
                    cos = embedding(pos_ids, cos_table) 
                    sin = embedding(pos_ids, sin_table)
                    
                    # Transpose to put '3' (mrope dim) before head_dim
                    # If ndim=4 [3, batch, seq_len, D] -> [batch, seq_len, 3, D]
                    # If ndim=3 [3, num_tokens, D] -> [num_tokens, 3, D]
                    if cos.ndim() == 4:
                        cos = cos.permute([1, 2, 0, 3])
                        sin = sin.permute([1, 2, 0, 3])
                    else:
                        cos = cos.permute([1, 0, 2])
                        sin = sin.permute([1, 0, 2])
                    
                    # Apply mRoPE section logic
                    # Now shape is [..., 3, head_size/2]
                    c0, c1, c2 = split(cos, [1, 1, 1], dim=-2)
                    c0 = squeeze(c0, -2)
                    c1 = squeeze(c1, -2)
                    c2 = squeeze(c2, -2)
                    
                    # Split head_dim (64) into [16, 24, 24]
                    c0_parts = split(c0, [16, 24, 24], dim=-1)
                    c1_parts = split(c1, [16, 24, 24], dim=-1)
                    c2_parts = split(c2, [16, 24, 24], dim=-1)
                    
                    # cos_final = concat([c0_parts[0], c1_parts[1], c2_parts[2]], dim=-1)
                    
                    s0, s1, s2 = split(sin, [1, 1, 1], dim=-2)
                    s0 = squeeze(s0, -2)
                    s1 = squeeze(s1, -2)
                    s2 = squeeze(s2, -2)
                    
                    s0_parts = split(s0, [16, 24, 24], dim=-1)
                    s1_parts = split(s1, [16, 24, 24], dim=-1)
                    s2_parts = split(s2, [16, 24, 24], dim=-1)
                    
                    # sin_final = concat([s0_parts[0], s1_parts[1], s2_parts[2]], dim=-1)
                    
                    # Expand to match head_size (128) by repeating for pairs
                    # Standard RoPE rotate_half assumes [-x2, x1] layout (half-half), so cos/sin should be duplicated (concat), not interleaved.
                    # However, mrope_rotate_half splits the tensor into sections [32, 48, 48] and rotates each section.
                    # So we must construct cos/sin to match this structure: [sec0, sec0, sec1, sec1, sec2, sec2].
                    
                    cos = concat([
                        c0_parts[0], c0_parts[0],
                        c1_parts[1], c1_parts[1],
                        c2_parts[2], c2_parts[2]
                    ], dim=-1)
                    
                    sin = concat([
                        s0_parts[0], s0_parts[0],
                        s1_parts[1], s1_parts[1],
                        s2_parts[2], s2_parts[2]
                    ], dim=-1)
                    
                    # 4. Broadcast Cos/Sin to match Q/K heads
                    # Expand for heads
                    if ndim == 2:
                        cos = expand_dims(cos, 1)
                        sin = expand_dims(sin, 1)
                    else:
                        cos = expand_dims(cos, 2)
                        sin = expand_dims(sin, 2)
                    
                    target_dtype = q.dtype 

                    # For packed (remove_input_padding) path, q/k are 3D: [tokens, heads, head_dim].
                    # RopeEmbeddingUtils.rotate_half requires a 4D tensor, so temporarily
                    # unsqueeze a batch dimension at axis 0, apply rotate_half, then squeeze back.
                    cos = cast(cos, target_dtype)
                    sin = cast(sin, target_dtype)

                    # 5. Apply Rotation
                    def mrope_rotate_half(x):
                        # Qwen2-VL mRoPE sections: 16, 24, 24 pairs -> 32, 48, 48 elements
                        # Total 128 elements.
                        # Note: This hardcoding is specific to Qwen2-VL-2B/7B.
                        x_sections = split(x, [32, 48, 48], dim=-1)
                        rx_sections = [RopeEmbeddingUtils.rotate_half(s) for s in x_sections]
                        return concat(rx_sections, dim=-1)

                    if ndim == 2:
                        q4 = unsqueeze(q, 0)
                        k4 = unsqueeze(k, 0)
                        cos4 = unsqueeze(cos, 0)
                        sin4 = unsqueeze(sin, 0)

                        q4 = (q4 * cos4) + (mrope_rotate_half(q4) * sin4)
                        k4 = (k4 * cos4) + (mrope_rotate_half(k4) * sin4)

                        q = squeeze(q4, 0)
                        k = squeeze(k4, 0)
                    else:
                        q = (q * cos) + (mrope_rotate_half(q) * sin)
                        k = (k * cos) + (mrope_rotate_half(k) * sin)
                    
                    v = cast(v, target_dtype)
                    if ndim == 2:
                        q = q.view(concat([batch_or_tokens, q_size]))
                        k = k.view(concat([batch_or_tokens, k_size]))
                        v = v.view(concat([batch_or_tokens, v_size]))
                    else:
                        q = q.view(concat([batch_or_tokens, seq_len, q_size]))
                        k = k.view(concat([batch_or_tokens, seq_len, k_size]))
                        v = v.view(concat([batch_or_tokens, seq_len, v_size]))
                    
                    q = cast(q, target_dtype)
                    k = cast(k, target_dtype)
                    v = cast(v, target_dtype)
                    
                    # 6. Concatenate and Finalize
                    qkv = concat([q, k, v], dim=-1)
                    
                    # Force a layout reset / copy to ensure contiguous memory
                    from tensorrt_llm.functional import identity
                    qkv = identity(qkv)

                    print(f"DEBUG: Manual RoPE applied. qkv.shape={qkv.shape}, qkv.dtype={qkv.dtype}")
                    
                    # 7. Disable Plugin RoPE
                    # We have already rotated Q and K. We must instruct the plugin NOT to apply RoPE again.
                    # We do this by setting rotary_embedding_dim to 0 and using a non-RoPE position embedding type.
                    rotary_embedding_dim = 0
                    position_embedding_type = PositionEmbeddingType.learned_absolute
                    rotary_cos_sin = None
                    rotary_inv_freq = None
                    mrope_rotary_cos_sin_arg = None
                    mrope_position_deltas_arg = None

            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                # attention_mask=attention_mask,
                # attention_packed_mask=attention_packed_mask,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                # num_kv_heads_origin=self.num_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base
                if not self.is_local else self.rotary_embedding_base_local,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_short_m_scale=attention_params.short_mscale,
                rotary_embedding_long_m_scale=attention_params.long_mscale,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                rotary_embedding_original_max_positions=self.
                original_max_position_embeddings,
                position_embedding_type=position_embedding_type,
                rotary_inv_freq=rotary_inv_freq
                if not self.is_local else rotary_inv_freq_local,
                rotary_cos_sin=rotary_cos_sin
                if not self.is_local else rotary_cos_sin_local,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                attention_output_orig_quant_scale=self.
                _get_output_orig_quant_scale(),
                # attention_output_sf_scale=attention_output_sf_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                block_sparse_block_size=self.block_sparse_params.block_size,
                block_sparse_homo_head_pattern=self.block_sparse_params.
                homo_head_pattern,
                block_sparse_num_local_blocks=self.block_sparse_params.
                num_local_blocks,
                block_sparse_vertical_stride=self.block_sparse_params.
                vertical_stride,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets
                if not self.cross_attention else
                kv_cache_params.cross_kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_pool_pointers,
                # host_kv_cache_pool_mapping=kv_cache_params.
                # host_kv_cache_pool_mapping if not self.cross_attention else
                # kv_cache_params.host_cross_kv_cache_pool_mapping,
                do_cross_attention=self.cross_attention,
                # cross_kv=cross_kv,
                # cross_kv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                # logn_scaling=logn_scaling,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
                spec_decoding_is_generation_length_variable=spec_decoding_params
                .spec_decoding_is_generation_length_variable,
                spec_decoding_max_generation_length=spec_decoding_params.
                spec_decoding_max_generation_length,
                spec_decoding_generation_lengths=spec_decoding_params.
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_params.
                spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_params.
                spec_decoding_packed_mask,
                # spec_decoding_use=spec_decoding_params.spec_decoding_use,
                # long_rope_rotary_inv_freq=long_rope_rotary_inv_freq,
                # long_rope_rotary_cos_sin=long_rope_rotary_cos_sin,
                mrope_rotary_cos_sin=mrope_rotary_cos_sin_arg,
                mrope_position_deltas=mrope_position_deltas_arg,
                qk_tanh_scale=self.max_attn_value,
                # attn_logit_softcapping_scale=self.max_attn_value,
                host_runtime_perf_knobs=attention_params.
                host_runtime_perf_knobs,
                # host_context_progress=attention_params.host_context_progress,
                # skip_attn=skip_attn,
                # cp_size=self.cp_size,
                # cp_rank=self.cp_rank,
                # cp_group=self.cp_group
            )

        else:
            # plain TensorRT mode
            assert paged_kv_cache == False

            assert logn_scaling is None, "plan TensorRT mode does not support logn scaling now"

            def transpose_for_scores(x,
                                     rotary: bool = False,
                                     is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])
                if rotary:
                    return x.view(new_x_shape)
                else:
                    return x.view(new_x_shape).permute([0, 2, 1, 3])

            # qkv after projection is of shape
            #   [bs, seqlen, (num_attention_heads + 2 * num_attention_kv_heads), attention_head_size].
            # The projected and split qkv after transpose_for_scores():
            #   Q[bs, num_attention_heads, seqlen, attention_head_size]
            #   K[bs, num_attention_kv_heads, seqlen, attention_head_size]
            #   V[bs, num_attention_kv_heads, seqlen, attention_head_size]
            kv_size = self.attention_head_size * self.num_attention_kv_heads
            if unfuse_qkv_gemm:
                query, key, value = qkv[0], qkv[1], qkv[2]
            else:
                query, key, value = split(
                    qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)

            # in cross attention mode, replace kv by encoder_output
            if self.cross_attention and encoder_output is not None:
                key, value = split(cross_kv, [kv_size, kv_size], dim=2)

            query = transpose_for_scores(
                query, rotary=self.position_embedding_type.is_rope())
            key = transpose_for_scores(
                key, is_kv=True, rotary=self.position_embedding_type.is_rope())
            value = transpose_for_scores(value, is_kv=True)

            if self.position_embedding_type.is_rope():
                if self.position_embedding_type == PositionEmbeddingType.long_rope:
                    sequence_length = shape(hidden_states, 1)
                    floor_seq_length = maximum(
                        sequence_length, self.original_max_position_embeddings)

                    starts = concat([0, 0, 0, 0])
                    shapes = concat(
                        [1, floor_seq_length, self.rotary_embedding_dim])
                    short = slice(attention_params.embed_positions, starts,
                                  shapes)
                    long = slice(attention_params.long_rope_embed_positions,
                                 starts, shapes)

                    embed_positions = concat([short, long], dim=0)
                    select = where(
                        sequence_length
                        <= self.original_max_position_embeddings, 0, 1)
                    embed_positions = slice(embed_positions,
                                            concat([select, 0, 0]),
                                            sizes=shape(short))
                    embed_positions = cast(embed_positions, self.dtype)
                elif is_same_dtype(self.dtype, trt.bfloat16):
                    embed_positions = cast(attention_params.embed_positions,
                                           trt.bfloat16)
                else:
                    embed_positions = cast(attention_params.embed_positions,
                                           query.dtype)

                if self.rotary_embedding_dim is not None:
                    # When shape(hidden_states, 1) > 1(Context phase), the embedding start from 0,
                    # otherwise (Generation phase) move start to position
                    if not use_cache:
                        # Only context phase is involved when kv cache is disabled.
                        start = 0
                    else:
                        start = where(
                            shape(hidden_states, 1) > 1, 0,
                            shape(past_key_value, 3))
                    size = where(
                        shape(hidden_states, 1) > 1, shape(hidden_states, 1), 1)
                    sincos = slice(embed_positions, concat([0, start, 0]),
                                   concat([1, size, self.rotary_embedding_dim]))
                    sin, cos = split(sincos,
                                     self.rotary_embedding_dim // 2,
                                     dim=-1)

                    key_rot_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), self.rotary_embedding_dim
                    ])
                    query_rot_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), self.rotary_embedding_dim
                    ])
                    remaining = shape(key, 3) - self.rotary_embedding_dim
                    key_pass_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), remaining
                    ])
                    query_pass_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), remaining
                    ])
                    k_rot = slice(key, [0, 0, 0, 0], key_rot_size)
                    k_pass = slice(key, [0, 0, 0, self.rotary_embedding_dim],
                                   key_pass_size)

                    q_rot = slice(query, [0, 0, 0, 0], query_rot_size)
                    q_pass = slice(query, [0, 0, 0, self.rotary_embedding_dim],
                                   query_pass_size)

                    k_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        k_rot, [cos, sin], self.position_embedding_type)
                    q_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        q_rot, [cos, sin], self.position_embedding_type)

                    key = concat([k_rot, k_pass], dim=3)
                    query = concat([q_rot, q_pass], dim=3)
                else:
                    key = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        key, [cos, sin], self.position_embedding_type)
                    query = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        query, [cos, sin], self.position_embedding_type)

                key = key.permute([0, 2, 1, 3])
                query = query.permute([0, 2, 1, 3])

            if past_key_value is not None and not self.cross_attention:
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = dequantize(
                        past_key_value,
                        self.kv_cache_scaling_factor.value,
                        output_type=self.dtype)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)

                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            if use_cache:
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)

                # TRT quantizes the tensor value by doing `cast(clip(fp_value / scale))` while
                # the plugin quantizes it by doing `cast(clip(fp_value * scale))`.
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = quantize(
                        past_key_value,
                        self.kv_cache_scaling_factor.value,
                        dtype='fp8'
                        if self.quant_mode.has_fp8_kv_cache() else 'int8')

            # MQA broadcast
            if self.num_attention_heads // self.num_attention_kv_heads > 1:
                key = repeat_interleave(
                    key,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)
                value = repeat_interleave(
                    value,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)

            key_length = shape(key, 2)

            # The following code creates a 2D tensor with 0s in the lower triangular (including the diagonal) and
            # +INF in the upper triangular parts. This bias tensor will be added to the output of the Q*K^T matrix
            # multiplication (BMM1). The +INF elements will be transformed to 0s by the Softmax operator that
            # follows. The elements that corresponds to 0s in the bias are unaffected by the bias tensor.
            #
            # Note that when we added to another bias tensor B (for example, with AliBi), the values in the lower-
            # triangular part of the B tensor are not affected and the upper-triangular ones are set to +INF.
            if self.attention_mask_type == AttentionMaskType.causal and not self.cross_attention:
                if self.position_embedding_type.is_alibi():
                    query_length = shape(query, 2)
                    # bsz, tatget_length, past_key_value_length
                    buffer = make_causal_mask(shape(query, 0), query_length,
                                              key_length - query_length,
                                              trt.float32)
                    starts = concat([0, 0, 0, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    generated_mask = slice(buffer, starts, sizes)

                else:
                    query_length = shape(query, 2)
                    starts = concat([0, 0, key_length - query_length, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    if self.position_embedding_type == PositionEmbeddingType.long_rope:
                        buf_shape = (self.original_max_position_embeddings,
                                     self.original_max_position_embeddings)
                    else:
                        buf_shape = (self.max_position_embeddings,
                                     self.max_position_embeddings)
                    select_buf = np.expand_dims(
                        np.tril(np.ones(buf_shape)).astype(bool), (0, 1))

                    select_buf = np.logical_not(select_buf)
                    mask_buf = np.zeros_like(select_buf, np.float32)
                    mask_buf[select_buf] = float('-inf')
                    buffer = constant(mask_buf)
                    generated_mask = slice(buffer, starts, sizes)

            elif self.attention_mask_type == AttentionMaskType.bidirectional and not self.cross_attention:
                query_length = shape(query, 2)
                zero_buf = np.expand_dims(
                    np.zeros((self.max_position_embeddings,
                              self.max_position_embeddings),
                             dtype=np.float32), (0, 1))

                zero_buf[:, :, :-1, -1] = 1
                zero_buf *= -10000

                mask = constant(zero_buf)

                # context phase, query_length
                mask_size = where(query_length > 1, query_length, 1)
                mask_start = where(query_length > 1,
                                   self.max_position_embeddings - mask_size, 1)
                start = concat([0, 0, mask_start, mask_start])
                size = concat([1, 1, mask_size, mask_size])
                generated_mask = slice(mask, start, size)

            if attention_mask is not None:
                if self.cross_attention:
                    batch_size = shape(attention_mask, 0)
                    query_len = shape(attention_mask, 1)
                    encoder_input_len = shape(attention_mask, 2)
                    attention_mask = attention_mask.view(
                        concat([batch_size, 1, query_len, encoder_input_len]))
                    attention_mask = where(attention_mask == 0, float('-inf'),
                                           0.0)
                else:
                    attention_mask = expand_mask(attention_mask,
                                                 shape(query, 2))
            bias = attention_mask
            if self.position_embedding_type.is_alibi():
                alibi_biases = generate_alibi_biases(alibi_slopes, key_length)
                bias = alibi_biases if bias is None else bias + alibi_biases

            if self.relative_attention:
                query_length = shape(query, 2)
                if self.use_implicit_relative_attention:
                    relative_bias = compute_relative_bias(
                        query_length + key_length - 1,
                        key_length,
                        self.num_buckets,
                        self.max_distance,
                        False,  # bidirectional
                        self.rel_attn_table.value.transpose(1, 0),
                        tp_size=self.tp_size,
                        tp_group=self.tp_group,
                        tp_rank=self.tp_rank)
                else:
                    relative_bias = unsqueeze(self.rel_attn_table.value, 0)
                start = concat([0, 0, query_length + key_length - 2,  0])
                size = concat([
                    shape(relative_bias, 0),
                    shape(relative_bias, 1), 1, key_length
                ])
                relative_bias = slice(relative_bias, start, size)

            key = key.permute([0, 1, 3, 2])
            model_type = query.dtype
            with precision('float32'):
                # FIXME the "with precision('float32') does not really work and lead to nan"
                # in some cases
                query = cast(query, 'float32')
                key = cast(key, 'float32')
                if norm_before_bmm1:
                    # Apply norm on query earlier to prevent matmul fp16 overflow.
                    query /= (self.q_scaling * self.norm_factor)
                attention_scores = matmul(query, key)
                if not norm_before_bmm1:
                    attention_scores = attention_scores / (self.q_scaling *
                                                           self.norm_factor)
                if self.max_attn_value > 0:
                    attention_scores = self.max_attn_value * ACT2FN['tanh'](
                        attention_scores / self.max_attn_value)

                if self.attention_mask_type in [
                        AttentionMaskType.causal,
                        AttentionMaskType.bidirectional
                ] and not self.cross_attention:

                    bias = generated_mask if bias is None else bias + generated_mask

                if bias is not None:
                    bias = cast(bias, attention_scores.dtype)
                    attention_scores = attention_scores + bias

                if self.relative_attention:
                    attention_scores = attention_scores + relative_bias

                attention_probs = softmax(attention_scores, dim=-1)
                attention_probs = cast(attention_probs, model_type)

            # A dummy reshape WAR for mha fusion
            attention_probs = attention_probs.view(
                concat([
                    shape(attention_probs, 0),
                    shape(attention_probs, 1),
                    shape(attention_probs, 2),
                    shape(value, 2)
                ]))
            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")

        if skip_attn is not None and not default_net(
        ).plugin_config.use_fp8_context_fmha:
            # This case is used when we can skip this attention layer directly.
            # The output would be undefined and not used if skip_attn is not None
            # and set skip_attn as True during runtime
            # But when use_fp8_context_fmha is enabled, the output data type of
            # attention_plugin is fp8. Since TRT's conditional layer does not support
            # FP8 data type yet, we cannot use it to skip the computation in such case.

            dense_conditional = Conditional(skip_attn)
            skip_case = dense_conditional.add_input(attention_input)
            context = dense_conditional.add_input(context)

        if self.inner_layernorm is not None:
            context = self.inner_layernorm(context)

        context = self.dense(context,
                             lora_runtime_params=dense_lora_params,
                             all_reduce_params=all_reduce_params)

        if skip_attn is not None and not default_net(
        ).plugin_config.use_fp8_context_fmha:
            context = dense_conditional.add_output(skip_case, context)

        if use_cache:
            return (context, past_key_value)
        else:
            return context

    def set_rel_attn_table(self, max_seq_len, precomputed_relative_attention):
        self.rel_attn_table = Parameter(shape=(self.num_attention_heads,
                                               max_seq_len + 1,
                                               max_seq_len + 1),
                                        dtype=self.dtype)
        self.rel_attn_table.value = precomputed_relative_attention

    def postprocess(self, tllm_key, weights, **kwargs):

        if tllm_key.endswith("kv_cache_scaling_factor"):
            if weights is None:
                return {tllm_key: torch.ones(1, ).float()}
            elif isinstance(weights, torch.Tensor):
                return {tllm_key: weights.float()}
            elif None in weights:
                return {tllm_key: torch.ones(1, ).float()}
            else:
                return {tllm_key: max(weights).float()}
        elif tllm_key.endswith("kv_cache_rcp_scaling_factor"):
            if weights is None:
                return {tllm_key: torch.ones(1, ).float()}
            elif isinstance(weights, torch.Tensor):
                return {tllm_key: torch.reciprocal(weights.float())}
            elif None in weights:
                return {tllm_key: torch.ones(1, ).float()}
            else:
                return {tllm_key: torch.reciprocal(max(weights).float())}
        else:
            return {tllm_key: weights}


class BertAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings=1024,
                 num_layers=1,
                 attention_head_size=None,
                 num_kv_heads=None,
                 q_scaling=1.0,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 cp_group=None,
                 cp_size=1,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.cp_group = cp_group
        self.cp_size = cp_size

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = q_scaling
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.dtype = dtype

        self.relative_attention = relative_attention
        self.max_distance = max_distance
        self.num_buckets = num_buckets

        # out dim is not necessarily hidden_size + kv specific size (in MQA/GQA), but num_heads * heads_size
        # example: d_model != num_heads * head_size in Flan-T5
        self.qkv = ColumnLinear(hidden_size,
                                tp_size * self.attention_hidden_size +
                                (2 * tp_size * self.num_attention_kv_heads *
                                 self.attention_head_size),
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False,
                                is_qkv=True)
        self.dense = RowLinear(tp_size * self.num_attention_heads *
                               self.attention_head_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

        # see optimize_model's add_lora for LoRA initialization
        self.qkv_lora = None

        # per-layer relative attention table
        if relative_attention:
            self.rel_attn_table = Parameter(shape=(num_attention_heads //
                                                   tp_size, num_buckets),
                                            dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None,
                lora_layer_params=None):
        assert isinstance(hidden_states, Tensor)

        qkv_lora_params = None
        if lora_layer_params is not None:
            qkv_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_qkv")

        qkv = self.qkv(hidden_states, qkv_lora_params)

        if default_net().plugin_config.remove_input_padding:
            assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            q_lora_params = lora_layer_params.get_runtime_params(0, "attn_q")
            k_lora_params = lora_layer_params.get_runtime_params(0, "attn_k")
            v_lora_params = lora_layer_params.get_runtime_params(0, "attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths,
                    max_num_tokens=q_lora_params.max_num_tokens)

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora

        if default_net().plugin_config.bert_attention_plugin:
            # TRT plugin mode
            assert input_lengths is not None
            assert self.cp_size == 1
            context = bert_attention(
                qkv,
                input_lengths,
                self.num_attention_heads,
                self.attention_head_size,
                q_scaling=self.q_scaling,
                relative_attention=self.relative_attention,
                max_distance=self.max_distance,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_input_length=max_input_length)
        else:
            # plain TRT mode
            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            kv_size = self.attention_head_size * self.num_attention_kv_heads
            query, key, value = split(
                qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)
            if self.cp_size > 1 and self.cp_group is not None:
                key = allgather(key, self.cp_group, gather_dim=1)
                value = allgather(value, self.cp_group, gather_dim=1)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key, use_fp32_acc=False)
            attention_scores = attention_scores / (self.q_scaling *
                                                   self.norm_factor)

            if self.relative_attention:
                query_len = shape(attention_scores, 2)
                key_len = shape(attention_scores, 3)
                bias = compute_relative_bias(
                    query_len,
                    key_len,
                    self.num_buckets,
                    self.max_distance,
                    True,  # bidirectional
                    self.rel_attn_table.value.transpose(1, 0),
                    tp_size=self.tp_size,
                    tp_group=self.tp_group,
                    tp_rank=self.tp_rank)
                attention_scores = attention_scores + bias

            if attention_mask is not None:
                attention_mask = expand_mask(attention_mask, shape(query, 2))
                attention_mask = cast(attention_mask, attention_scores.dtype)
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")
        context = self.dense(context, lora_runtime_params=dense_lora_params)

        return context


class CogVLMAttention(Attention):

    def __init__(
            self,
            *,
            local_layer_idx,
            hidden_size,
            num_attention_heads,
            num_kv_heads=None,
            max_position_embeddings=1024,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            dtype=None,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=10000.0,
            rotary_embedding_scaling=None,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            vision_start=1,
            vision_length=1225,
            quant_mode: QuantMode = QuantMode(0),
            dense_bias=None,
    ):
        super().__init__(local_layer_idx=local_layer_idx,
                         hidden_size=hidden_size,
                         num_attention_heads=num_attention_heads,
                         num_kv_heads=num_kv_heads,
                         max_position_embeddings=max_position_embeddings,
                         dtype=dtype,
                         attention_mask_type=attention_mask_type,
                         bias=bias,
                         position_embedding_type=position_embedding_type,
                         rotary_embedding_base=rotary_embedding_base,
                         rotary_embedding_scaling=rotary_embedding_scaling,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         quant_mode=quant_mode)

        self.vision_length = vision_length
        self.vision_start = vision_start

        self.vis_qkv = ColumnLinear(
            hidden_size,
            tp_size * self.num_attention_heads * self.attention_head_size +
            (2 * tp_size * self.num_attention_kv_heads *
             self.attention_head_size),
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            is_qkv=True)
        self.vis_dense = RowLinear(tp_size * self.num_attention_heads *
                                   self.attention_head_size,
                                   hidden_size,
                                   bias=self.dense_bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        assert isinstance(hidden_states, Tensor)
        assert (not default_net().plugin_config.remove_input_padding)
        assert (default_net().plugin_config.gpt_attention_plugin)

        bs = shape(hidden_states, 0)
        seq_length = shape(hidden_states, 1)
        bos = slice(hidden_states, [0, 0, 0],
                    concat([bs, self.vision_start, self.hidden_size]))
        vis_seq_length = minimum(self.vision_length + 1, seq_length - 1)
        vision_hidden_states = slice(
            hidden_states, [0, self.vision_start, 0],
            concat([bs, vis_seq_length, self.hidden_size]))
        text_seq_length = maximum(
            0, seq_length - (self.vision_length + 1 + self.vision_start))
        language_hidden_states = slice(
            hidden_states, [0, self.vision_length + 1 + self.vision_start, 0],
            concat([bs, text_seq_length, self.hidden_size]))
        bos_qkv = self.qkv(bos)
        language_qkv = self.qkv(language_hidden_states)
        vision_qkv = self.vis_qkv(vision_hidden_states)
        qkv = concat([bos_qkv, vision_qkv, language_qkv], dim=1)

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params is None or kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )

        if default_net().plugin_config.gpt_attention_plugin:
            if self.cross_attention and (past_key_value is not None):
                past_key_value = kv_cache_params.past_key_value[1]
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm
            ], 'Plugin only support masked MHA.'

            # KV cache scales.
            kv_orig_quant_scale = constant(
                fp32_array([1.0])
            ) / self.kv_cache_scaling_factor.value if self.quant_mode.has_kv_cache_quant(
            ) else None
            kv_quant_orig_scale = self.kv_cache_scaling_factor.value if self.quant_mode.has_kv_cache_quant(
            ) else None

            # Attention output scales
            assert (
                not default_net().plugin_config.use_fp8_context_fmha
            ) or self.quant_mode.has_fp8_qdq(
            ), "FP8 Context FMHA must be used together with the fp8 quantization workflow."

            rotary_inv_freq = attention_params.rotary_inv_freq
            rotary_cos_sin = attention_params.embed_positions_for_gpt_attention
            attention_output_orig_quant_scale = self.attention_output_orig_quant_scale.value if self.attention_output_orig_quant_scale is not None else None
            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                rotary_inv_freq=rotary_inv_freq,
                rotary_cos_sin=rotary_cos_sin,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                attention_output_orig_quant_scale=
                attention_output_orig_quant_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                alibi_slopes=None,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                vision_start=self.vision_start,
                vision_length=self.vision_length,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers,
                do_cross_attention=self.cross_attention,
                cross_qkv=None,
                cross_qkv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
                host_runtime_perf_knobs=attention_params.host_runtime_perf_knobs
            )

        bs = shape(context, 0)
        seq_length = shape(context, 1)
        bos = slice(context, [0, 0, 0],
                    concat([bs, self.vision_start, self.hidden_size]))
        vis_seq_length = minimum(self.vision_length + 1, seq_length - 1)
        vision_hidden_states = slice(
            context, [0, self.vision_start, 0],
            concat([bs, vis_seq_length, self.hidden_size]))
        text_seq_length = maximum(
            0, seq_length - (self.vision_length + 1 + self.vision_start))
        language_hidden_states = slice(
            context, [0, self.vision_length + 1 + self.vision_start, 0],
            concat([bs, text_seq_length, self.hidden_size]))

        bos_dense = self.dense(bos)
        language_dense = self.dense(language_hidden_states)
        vision_dense = self.vis_dense(vision_hidden_states)
        context = concat([bos_dense, vision_dense, language_dense], dim=1)

        if use_cache:
            return (context, past_key_value)
        else:
            return context
