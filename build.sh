python3 /home/xsf/fl/edge/tensorrt-llm-examples/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir=/home/xsf/fl/edge/models/Qwen2-VL-2B-Instruct \
    --output_dir=/home/xsf/fl/edge/models/tllm_ckpts/Qwen2-VL-2B-Instruct/fp16 \
    --dtype float16

trtllm-build --checkpoint_dir=/home/xsf/fl/edge/models/tllm_ckpts/Qwen2-VL-2B-Instruct/fp16 \
    --output_dir=/home/xsf/fl/edge/models/trt_engines/qwen2vl/fp16/vl/llm \
    --gemm_plugin=float16 \
    --gpt_attention_plugin=float16 \
    --max_batch_size=4 \
    --max_input_len=2048 \
    --max_seq_len=3072 \
    --max_multimodal_len=1296 #(max_batch_size) * 324 (num_visual_features), this's for image_shape=[504,504]

python /home/xsf/fl/edge/tensorrt-llm-examples/examples/models/core/multimodal/build_multimodal_engine.py \
    --model_type qwen2_vl --model_path /home/xsf/fl/edge/models/Qwen2-VL-2B-Instruct \
    --output_dir /home/xsf/fl/edge/models/trt_engines/qwen2vl/fp16/vl/vision



python /home/xsf/fl/edge/trtllm0120/trtllm-build.py --checkpoint_dir=/home/xsf/fl/edge/models/tllm_ckpts/Qwen2-VL-2B-Instruct/fp16     --output_dir=/home/xsf/fl/edge/models/trt_0120_engines/llm     --gemm_plugin=float16     --gpt_attention_plugin=float16     --max_batch_size=1     --max_input_len=2048     --max_seq_len=3072     --max_multimodal_len=1296

