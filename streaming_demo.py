"""
streaming_demo.py

示例：优先使用异步流（LLM.generate_async(..., streaming=True)），若不可用则回退到同步 generate(..., streaming=True) 的 generator。
测量并记录：start、first token 到达时间（TTFT）、结束时间、token count、吞吐（tokens/s）。

用法示例：
    python3 streaming_demo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --concurrency 1

注意：运行此脚本需要可用的 tensorrt_llm 运行时与已编译 engine 或可通过 HF model id 下载的模型。
"""

import argparse
import asyncio
import time
import statistics
from typing import List, Optional

try:
    from tensorrt_llm import LLM, SamplingParams
    HAVE_TRTLLM = True
except Exception:
    HAVE_TRTLLM = False

try:
    from transformers import AutoTokenizer
    HAVE_TOKENIZER = True
except Exception:
    HAVE_TOKENIZER = False


async def _stream_async(llm: LLM, prompt: str, sampling: SamplingParams, tokenizer=None):
    start = time.monotonic()
    first_token_time = None
    chunks: List[str] = []

    # generate_async(..., streaming=True) 应返回一个 async iterator
    async for out in llm.generate_async(prompt, sampling, streaming=True):
        # out.outputs[0].text 是常见字段
        txt = ""
        try:
            txt = out.outputs[0].text
        except Exception:
            txt = getattr(out.outputs[0], "text", "") or str(out.outputs[0])
        now = time.monotonic()
        if first_token_time is None and txt.strip() != "":
            first_token_time = now
        chunks.append(txt)
        # 即时打印流式片段
        print(txt, end="", flush=True)

    end = time.monotonic()
    full = "".join(chunks)
    token_count = None
    if tokenizer is not None:
        try:
            token_count = len(tokenizer.encode(full, add_special_tokens=False))
        except Exception:
            token_count = None

    ttft = (first_token_time - start) if first_token_time is not None else None
    throughput = None
    if token_count is not None and first_token_time is not None and end > first_token_time:
        throughput = token_count / (end - first_token_time)

    return {
        "prompt": prompt,
        "full_text": full,
        "token_count": token_count,
        "start": start,
        "first_token_time": first_token_time,
        "end": end,
        "ttft": ttft,
        "total_latency": end - start,
        "throughput_tokens_per_s": throughput,
    }


def _stream_sync(llm, prompt: str, sampling, tokenizer=None):
    """
    回退路径：如果没有 async 接口，则尝试同步 generate(..., streaming=True) -> 返回一个 generator（见 model_runner_cpp._stream）
    """
    start = time.monotonic()
    first_token_time = None
    chunks: List[str] = []

    # llm.generate 可能抛出或不接受 streaming=True；调用方应捕获异常
    gen = llm.generate(prompt, sampling, streaming=True)
    # gen 预期为 generator，逐次 yield 出部分输出（dict 或自定义对象）
    try:
        for out in gen:
            # out 可能是 dict 或对象；尝试提取文本片段
            txt = ""
            # common shapes: out['output_text'] or out.outputs[0].text or out['outputs'][0]['text']
            if isinstance(out, dict):
                # 优先找常见键
                if 'output_text' in out:
                    txt = out['output_text']
                else:
                    # inspect nested
                    try:
                        txt = out.get('outputs', [])[0].get('text', '')
                    except Exception:
                        txt = str(out)
            else:
                try:
                    txt = out.outputs[0].text
                except Exception:
                    txt = getattr(out.outputs[0], 'text', '') or str(out)

            now = time.monotonic()
            if first_token_time is None and txt.strip() != "":
                first_token_time = now
            chunks.append(txt)
            print(txt, end="", flush=True)
    except TypeError:
        # 如果不是 generator，则表示该 API 不支持 streaming=True 的回退
        raise

    end = time.monotonic()
    full = "".join(chunks)
    token_count = None
    if tokenizer is not None:
        try:
            token_count = len(tokenizer.encode(full, add_special_tokens=False))
        except Exception:
            token_count = None

    ttft = (first_token_time - start) if first_token_time is not None else None
    throughput = None
    if token_count is not None and first_token_time is not None and end > first_token_time:
        throughput = token_count / (end - first_token_time)

    return {
        "prompt": prompt,
        "full_text": full,
        "token_count": token_count,
        "start": start,
        "first_token_time": first_token_time,
        "end": end,
        "ttft": ttft,
        "total_latency": end - start,
        "throughput_tokens_per_s": throughput,
    }


async def run_prompts_async(model_name: str, prompts: List[str], concurrency: int, tokenizer_name: Optional[str], max_new_tokens: int):
    if not HAVE_TRTLLM:
        raise RuntimeError('tensorrt_llm is not importable in this environment')

    llm = LLM(model=model_name)
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_new_tokens=max_new_tokens)

    tokenizer = None
    if tokenizer_name and HAVE_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception:
            tokenizer = None

    sem = asyncio.Semaphore(concurrency)

    async def guarded(prompt):
        async with sem:
            # 优先使用 async 接口
            if hasattr(llm, 'generate_async'):
                return await _stream_async(llm, prompt, sampling, tokenizer)
            else:
                # 回退到线程池中运行同步生成（因为 _stream_sync 可能阻塞）
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, _stream_sync, llm, prompt, sampling, tokenizer)

    tasks = [guarded(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results


def print_aggregate(results):
    ttfts = [r['ttft'] for r in results if r['ttft'] is not None]
    throughputs = [r['throughput_tokens_per_s'] for r in results if r['throughput_tokens_per_s'] is not None]
    latencies = [r['total_latency'] for r in results if r['total_latency'] is not None]

    def stats(arr):
        if not arr:
            return None
        return {
            'mean': statistics.mean(arr),
            'median': statistics.median(arr),
            'pstd': statistics.pstdev(arr),
            'n': len(arr),
        }

    print('\n\n=== per-request summary ===')
    for r in results:
        print(f"Prompt: {r['prompt']!r}")
        print(f" TTFT: {r['ttft']}")
        print(f" Total latency: {r['total_latency']:.4f}s")
        print(f" Token count: {r['token_count']}")
        print(f" Throughput (tokens/s): {r['throughput_tokens_per_s']}")
        print('----')

    print('\n=== aggregated metrics ===')
    print('TTFT:', stats(ttfts))
    print('Throughput tokens/s:', stats(throughputs))
    print('Total latency:', stats(latencies))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--prompts', type=str, nargs='*', default=[
        'Hello, my name is',
        'The capital of France is',
        'The future of AI is',
    ])
    parser.add_argument('--tokenizer', type=str, default=None, help='tokenizer name for accurate token counting')
    parser.add_argument('--max_new_tokens', type=int, default=64)

    args = parser.parse_args()

    # 运行 async 主体
    results = asyncio.run(run_prompts_async(args.model, args.prompts, args.concurrency, args.tokenizer, args.max_new_tokens))
    print_aggregate(results)


if __name__ == '__main__':
    main()
