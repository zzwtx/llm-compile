#!/usr/bin/env python3
"""
Download the first N images from the ILSVRC/imagenet-1k validation split on Hugging Face
using streaming (datasets.load_dataset streaming=True). This dataset is gated; the script
expects a valid HF token in the environment variable HF_TOKEN or HFG_TOKEN (or provided
via --hf_token). Images are saved as JPEGs into the output directory.

Usage:
  HF_TOKEN=<your_token> python3 scripts/download_imagenet_first10.py --output_dir /home/xsf/fl/edge/images --num 10

If the dataset is gated and you don't provide a token, the script will exit with an error.
"""

import os
import sys
import argparse
from itertools import islice
from pathlib import Path

try:
    from datasets import load_dataset
except Exception as e:
    print('Missing dependency "datasets". Install with: pip install datasets[image]', file=sys.stderr)
    raise

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/home/xsf/fl/edge/images')
    parser.add_argument('--num', type=int, default=10, help='Number of images to download')
    parser.add_argument('--dataset', type=str, default='ILSVRC/imagenet-1k', help='Dataset id to stream')
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading dataset {args.dataset} in streaming mode... (this requires a valid HF token)')
    try:
        dataset = load_dataset(args.dataset, split='validation', streaming=True)
    except Exception as e:
        print(f'Failed to load dataset streaming: {e}', file=sys.stderr)
        sys.exit(3)

    print(f'Streaming first {args.num} samples and saving to {out_dir}')
    saved = 0
    for i, sample in enumerate(islice(dataset, args.num)):
        try:
            img = sample.get('image')
            if img is None:
                print(f'Sample #{i} has no image field, skipping', file=sys.stderr)
                continue
            # Ensure RGB
            img = img.convert('RGB')
            fname = f'ILSVRC2012_val_{saved+1:08d}.JPEG'
            path = out_dir / fname
            img.save(path, format='JPEG')
            print(f'Saved {path}')
            saved += 1
        except Exception as e:
            print(f'Failed to save sample #{i}: {e}', file=sys.stderr)

    print(f'Done. Saved {saved} images to {out_dir}')


if __name__ == '__main__':
    main()
