#!/usr/bin/env python3
import os
import shutil
import argparse
import torch

from transformers import AutoConfig, AutoModelForImageTextToText

from modeling.qwen3hf_ada import register


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="directory contain steer")
    parser.add_argument("--scale", type=float, required=True, help="scaling values")
    parser.add_argument("--layer_idx", type=int, default=31, help="default layer 31")
    parser.add_argument("--save_path", type=str, default=None, help="save path")
    parser.add_argument("--extra_files_path", type=str, default=None,
                        help="Optional path to copy extra tokenizer/config files from")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    args = parse_args()

    src_path = args.src_path
    scale = args.scale
    layer_idx = args.layer_idx
    torch_dtype = get_torch_dtype(args.torch_dtype)

    if args.save_path is None:
        scale_str = str(scale).replace(".", "p")
        save_path = f"{src_path}-layer{layer_idx}-scale{scale_str}"
    else:
        save_path = args.save_path

    print("[INFO] Register custom config/model ...")
    register()

    print(f"[INFO] Load config from: {src_path}")
    cfg = AutoConfig.from_pretrained(src_path)

    cfg.adapter_layers = "all"

    print(f"[INFO] Load model from: {src_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        src_path,
        config=cfg,
        torch_dtype=torch_dtype,
        device_map="auto",  
    )

    layer = model.model.language_model.layers[layer_idx]
    if not hasattr(layer, "adapter") or layer.adapter is None:
        raise RuntimeError(f"layer {layer_idx} has no adapter")
    if not hasattr(layer.adapter, "steer"):
        raise RuntimeError(f"layer {layer_idx}.adapter has no steer")

    steer = layer.adapter.steer
    before_norm = steer.data.float().norm().item()
    before_max = steer.data.float().abs().max().item()

    print(f"[INFO] Before scaling:")
    print(f"       name = model.language_model.layers.{layer_idx}.adapter.steer")
    print(f"       shape = {tuple(steer.shape)}")
    print(f"       dtype = {steer.dtype}")
    print(f"       norm = {before_norm:.6f}")
    print(f"       absmax = {before_max:.6f}")

    steer.data.mul_(scale)

    after_norm = steer.data.float().norm().item()
    after_max = steer.data.float().abs().max().item()

    print(f"[INFO] After scaling by {scale}:")
    print(f"       norm = {after_norm:.6f}")
    print(f"       absmax = {after_max:.6f}")

    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] Saving to: {save_path}")
    model.save_pretrained(
        save_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    if args.extra_files_path is not None:
        extra_files = [
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor.json",
        ]
        print(f"[INFO] Copying extra files from {args.extra_files_path} ...")
        for fname in extra_files:
            src_file = os.path.join(args.extra_files_path, fname)
            dst_file = os.path.join(save_path, fname)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)

    print("[INFO] Done.")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()