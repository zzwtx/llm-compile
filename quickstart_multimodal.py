import argparse
import json
import os
import time

import numpy as np
from tensorrt_llm import logger
from tensorrt_llm import mpi_rank

from quickstart_advanced import add_llm_args, setup_llm

from tensorrt_llm.inputs import (ALL_SUPPORTED_MULTIMODAL_MODELS,
                                 default_multimodal_input_loader)

example_medias_and_prompts = {
    "image": {
        "media": [
            "/home/xsf/fl/edge/images/ILSVRC2012_val_00000001.JPEG",
            "/home/xsf/fl/edge/images/ILSVRC2012_val_00000003.JPEG",
            "/home/xsf/fl/edge/images/ILSVRC2012_val_00000002.JPEG",
        ],
        "prompt": [
            "Describe the natural environment in the image.",
            "Describe the object and the weather condition in the image.",
            "Describe the action in the picture.",
        ]
    },
    "video": {
        "media": [
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
        ],
        "prompt": [
            "Tell me what you see in the video briefly.",
            "Describe the scene in the video briefly.",
        ]
    },
    "audio": {
        "media": [
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_the_traffic_sign_in_the_image.wav",
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav",
        ],
        "prompt": [
            "Transcribe the audio clip into text, please don't add other text.",
            "Transcribe the audio clip into text, please don't add other text.",
        ]
    },
    "image_audio": {
        "media": [
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
        ],
        "prompt": [
            "Describe the scene in the image briefly.",
            "",
        ]
    },
    "multiple_image": {
        "media": [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
            "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
        ],
        "prompt": ["Describe the difference between the two images."],
    },
    "mixture_text_image": {
        "media": [
            [],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
            ],
        ],
        "prompt": [
            "Who invented the internet?",
            "Describe the scene in the image briefly.",
        ],
    },
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
EXPERIMENT_IMAGE_FILES = [
    os.path.join(EXPERIMENT_IMAGE_DIR,
                 f"ILSVRC2012_val_{i:08d}.JPEG")
    for i in range(1, 11)
]
EXPERIMENT_PROMPTS = [
    f"Question: Describe this image. Answer:" for i in range(len(EXPERIMENT_IMAGE_FILES))
]


def add_multimodal_args(parser):
    parser.add_argument("--model_type",
                        type=str,
                        choices=ALL_SUPPORTED_MULTIMODAL_MODELS,
                        help="Model type.")
    parser.add_argument("--modality",
                        type=str,
                        choices=[
                            "image", "video", "audio", "image_audio",
                            "multiple_image", "mixture_text_image"
                        ],
                        default="image",
                        help="Media type.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    parser.add_argument("--image_format",
                        type=str,
                        choices=["pt", "pil"],
                        default="pt",
                        help="The format of the image.")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="The device to have the input on.")
    # Add multiturn conversation related parameters
    parser.add_argument("--multiturn",
                        action="store_true",
                        help="Enable multi-turn conversation mode.")
    parser.add_argument(
        "--conversation_turns",
        type=int,
        default=2,
        help="Number of conversation turns for automated testing.")
    return parser


def add_lora_args(parser):
    parser.add_argument("--load_lora",
                        default=False,
                        action='store_true',
                        help="Whether to load the LoRA model.")
    parser.add_argument("--auto_model_name",
                        type=str,
                        default=None,
                        help="The auto model name in TRTLLM repo.")
    return parser


def _make_lora_request(model_class, args, llm, input_count):
    if not args.load_lora:
        return None
    if model_class is None:
        raise RuntimeError("LoRA was requested but the model class could not be loaded.")
    return model_class.lora_request(input_count, args.modality, llm._hf_model_dir)


def _build_multimodal_inputs(llm, args, model_type, prompt, media):
    return default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=llm._hf_model_dir,
        model_type=model_type,
        modality=args.modality,
        prompts=[prompt],
        media=[media],
        image_data_format=args.image_format,
        num_frames=args.num_frames,
        device=args.device,
    )


def _run_iteration(llm, sampling_params, args, model_type, prompt, media, model_class):
    start_time = time.perf_counter()
    inputs = _build_multimodal_inputs(llm, args, model_type, prompt, media)
    lora_request = _make_lora_request(model_class, args, llm, len(inputs))

    llm_start_time = time.perf_counter()
    streaming_result = llm.generate_async(inputs[0],
                                          sampling_params,
                                          lora_request=lora_request,
                                          streaming=True)

    chunk_count = 0
    first_token_time = None
    last_chunk = None

    for _ in streaming_result:
        chunk_count += 1
        chunk = streaming_result.outputs[0]
        if first_token_time is None and chunk.token_ids_diff:
            first_token_time = time.perf_counter()
        last_chunk = chunk

    llm_end_time = time.perf_counter()
    total_time = llm_end_time - start_time

    if last_chunk is None and streaming_result.outputs:
        last_chunk = streaming_result.outputs[0]

    text = last_chunk.text if last_chunk is not None else ""
    token_ids = list(last_chunk.token_ids) if last_chunk is not None else []
    tok_count = len(token_ids)

    streaming_metrics = {
        "llm_start_time": llm_start_time,
        "llm_end_time": llm_end_time,
        "first_token_time": first_token_time,
        "chunk_count": chunk_count,
    }
    setattr(llm, "last_streaming_metrics", streaming_metrics)

    llm_elapsed = max(llm_end_time - llm_start_time, 0.0)
    # ttft_value = first_token_time - start_time
    if first_token_time is not None:
        ttft_value = max(first_token_time - start_time, 0.0)

    throughput = tok_count / max(llm_elapsed, 1e-9) if tok_count > 0 else 0.0
    metrics = {
        "total": total_time,
        "llm": llm_elapsed,
        "ttft": ttft_value,
        "tokens": tok_count,
        "throughput": throughput,
        "prompt": prompt,
        "media": media,
        "text": text,
        "streaming_metrics": streaming_metrics,
    }
    outputs = [streaming_result]
    return outputs, metrics


def _log_experiment_summary(records):
    if mpi_rank() != 0:
        return

    if not records:
        logger.info("No experiment records to summarize.")
        return

    def stats(values):
        arr = np.array(values, dtype=float)
        return float(np.mean(arr)), float(np.median(arr)), float(np.var(arr))

    total_secs = [r["total"] for r in records]
    llm_secs = [r["llm"] for r in records]
    ttft_vals = [r["ttft"] for r in records]
    throughput_vals = [r["throughput"] for r in records]

    total_mean, total_med, total_var = stats(total_secs)
    ttft_mean, ttft_med, ttft_var = stats(ttft_vals)
    thr_mean, thr_med, thr_var = stats(throughput_vals)
    llm_mean, llm_med, llm_var = stats(llm_secs)

    logger.info('=== Experiment summary across %d iterations ===' % len(records))
    logger.info('TTFT (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (ttft_mean, ttft_med, ttft_var))
    logger.info('Output Throughput (tokens/s) - \nmean: %.2f, median: %.2f, var: %.4f' % (thr_mean, thr_med, thr_var))
    logger.info('Total Latency (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (total_mean, total_med, total_var))
    logger.info('LLM generate (s) - \nmean: %.4fs, median: %.4fs, var: %.6f' % (llm_mean, llm_med, llm_var))
    logger.info('Vision encoder (s) - not instrumented in this script')


def run_experiment(llm, sampling_params, args, model_class, model_type):
    if mpi_rank() == 0:
        if args.modality != "image":
            logger.warning("Experiment mode is best suited for the image modality; overriding to 'image'.")
            args.modality = "image"

        if not EXPERIMENT_IMAGE_FILES:
            raise RuntimeError("No experiment images were found.")

        num_warmup = args.experiment_warmup_iters
        num_iters = args.experiment_iterations
        logger.info(
            f"Running experiment: warmup={num_warmup}, iterations={num_iters}, image count={len(EXPERIMENT_IMAGE_FILES)}"
        )

    for warmup_idx in range(num_warmup):
        image_idx = warmup_idx % len(EXPERIMENT_IMAGE_FILES)
        _, warmup_metrics = _run_iteration(
            llm,
            sampling_params,
            args,
            model_type,
            EXPERIMENT_PROMPTS[image_idx],
            EXPERIMENT_IMAGE_FILES[image_idx],
            model_class,
        )
        if mpi_rank() == 0:
            logger.info(
                f"[Warmup {warmup_idx + 1}/{num_warmup}] prompt={warmup_metrics['prompt']!r},"
                f" tokens={warmup_metrics['tokens']}, total={warmup_metrics['total']:.3f}s"
            )

    records = []
    for test_idx in range(num_iters):
        image_idx = test_idx % len(EXPERIMENT_IMAGE_FILES)
        outputs, metrics = _run_iteration(
            llm,
            sampling_params,
            args,
            model_type,
            EXPERIMENT_PROMPTS[image_idx],
            EXPERIMENT_IMAGE_FILES[image_idx],
            model_class,
        )
        if mpi_rank() == 0:
            records.append(metrics)
            logger.info(
                f"[Test {test_idx + 1}/{num_iters}] prompt={metrics['prompt']!r},"
                f" tokens={metrics['tokens']}, total={metrics['total']:.3f}s,"
                f" throughput={metrics['throughput']:.2f} tok/s"
            )
            if metrics['text']:
                logger.info(f"Generated text: {metrics['text']!r}")

    _log_experiment_summary(records)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser = add_multimodal_args(parser)
    parser = add_lora_args(parser)
    args = parser.parse_args()

    args.disable_kv_cache_reuse = True  # kv cache reuse does not work for multimodal, force overwrite
    if args.kv_cache_fraction is None:
        args.kv_cache_fraction = 0.6  # lower the default kv cache fraction for multimodal

    return args


def main():
    if mpi_rank() != 0:
        # All subsequent logic is only for rank 0
        # The LLM engine will be initialized in each rank, but the main loop
        # should only be in rank 0.
        # The other ranks will be in a waiting state.
        return

    args = parse_arguments()
    logger.set_level('info')

    model_class = None
    lora_config = None
    if args.load_lora:
        assert args.auto_model_name is not None, "Please provide the auto model name to load LoRA config."
        import importlib
        models_module = importlib.import_module('tensorrt_llm._torch.models')
        model_class = getattr(models_module, args.auto_model_name)
        lora_config = model_class.lora_config(args.model_dir)
        # For stability - explicitly set the LoRA GPU cache & CPU cache to have space for 2 adapters
        lora_config.max_loras = 2
        lora_config.max_cpu_loras = 2

    llm, sampling_params = setup_llm(args, lora_config=lora_config)

    image_format = args.image_format
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = json.load(
            open(os.path.join(llm._hf_model_dir, 'config.json')))['model_type']
    assert model_type in ALL_SUPPORTED_MULTIMODAL_MODELS, f"Unsupported model_type: {model_type}"

    if args.experiment_mode:
        run_experiment(llm, sampling_params, args, model_class, model_type)
        return

    # If multiturn mode is enabled
    if args.multiturn:
        # Run predefined multiturn conversation examples
        assert args.prompt is not None, "Please provide a prompt for multiturn conversation."
        assert args.media is not None, "Please provide media for multiturn conversation."
        # Determine how many turns to run
        max_turns = min(args.conversation_turns, len(args.prompt))
        generated_outputs = []  # Store generated outputs for return

        # Initialize conversation history with the first prompt
        conversation_history = args.prompt[0] if args.prompt else ""

        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")

            try:
                # Use multimodal input loader to process input with conversation context
                # Use accumulated conversation history instead of just the current prompt
                cur_prompt = conversation_history
                inputs = default_multimodal_input_loader(
                    tokenizer=llm.tokenizer,
                    model_dir=llm._hf_model_dir,
                    model_type=model_type,
                    modality=args.modality,
                    prompts=[cur_prompt],
                    media=args.media,
                    image_data_format="pt",
                    num_frames=8,
                    device="cpu")

                lora_request = _make_lora_request(model_class, args, llm, len(inputs))

                # Generate response
                outputs = llm.generate(inputs,
                                       sampling_params,
                                       lora_request=lora_request)
                assert outputs and len(
                    outputs) > 0 and outputs[0].outputs and len(
                        outputs[0].outputs) > 0
                response = outputs[0].outputs[0].text.strip()

                # Store generated output
                generated_outputs.append({
                    "turn": i + 1,
                    "user_input": cur_prompt,
                    "assistant_response": response,
                    "media": args.media
                })

                conversation_history = conversation_history + "\n" + response
                if i + 1 < len(args.prompt):
                    conversation_history = conversation_history + "\n" + args.prompt[
                        i + 1]

            except Exception as e:
                print(f"Error in turn {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        for i, output in enumerate(generated_outputs):
            print(
                f"[{i}] Prompt: {output['user_input']!r}, Generated text: {output['assistant_response']!r}"
            )
        return

    # Original single-turn processing logic
    # set prompts and media to example prompts and images if they are not provided
    if args.prompt is None:
        args.prompt = example_medias_and_prompts[args.modality]["prompt"]
    if args.media is None:
        args.media = example_medias_and_prompts[args.modality]["media"]
    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=llm._hf_model_dir,
                                             model_type=model_type,
                                             modality=args.modality,
                                             prompts=args.prompt,
                                             media=args.media,
                                             image_data_format=image_format,
                                             num_frames=args.num_frames,
                                             device=args.device)

    lora_request = None
    if args.load_lora:
        lora_request = _make_lora_request(model_class, args, llm, len(inputs))

    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=lora_request,
    )

    for i, output in enumerate(outputs):
        prompt = args.prompt[i]
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
