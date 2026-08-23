"""Extract the vision tower from a full Qwen2.5-VL checkpoint.

The training scripts operate on the ViT slice alone, so this produces the
`Qwen2.5-VL-ViT-Only` directory they expect as `--base_model`:

    python extract_ViT.py --model_path Qwen/Qwen2.5-VL-7B-Instruct \
                          --output_path ./Qwen2.5-VL-ViT-Only

The processor is saved alongside the weights because the evaluation wrapper and
the renderer load their image processor from this directory.
"""
import argparse
import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def print_model_stats(model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_bytes = param_size + buffer_size

    print(f"\n======== {model_name} Statistics ========")
    print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Memory Footprint     : {total_size_bytes / 1024 ** 2:.2f} MB "
          f"({total_size_bytes / 1024 ** 3:.4f} GB)")
    print(f"Dtype                : {next(model.parameters()).dtype}")
    print("========================================\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True,
                    help="Full Qwen2.5-VL checkpoint (hub id or local directory)")
    ap.add_argument("--output_path", default="./Qwen2.5-VL-ViT-Only")
    ap.add_argument("--stats_only", action="store_true",
                    help="Only report parameter counts, write nothing")
    args = ap.parse_args()

    print(f"Loading full model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )

    print("Extracting Visual module...")
    visual_model = model.visual
    print_model_stats(visual_model, "Vision tower")
    if args.stats_only:
        return

    vision_config = model.config.vision_config

    print(f"Saving to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(visual_model.state_dict(),
               os.path.join(args.output_path, "pytorch_model.bin"))
    vision_config.save_pretrained(args.output_path)
    AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    ).save_pretrained(args.output_path)

    print("Verifying load...")
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    loaded = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
    loaded.load_state_dict(
        torch.load(os.path.join(args.output_path, "pytorch_model.bin")), strict=True
    )
    print(f"Done. Use '{args.output_path}' as --base_model.")


if __name__ == "__main__":
    main()
