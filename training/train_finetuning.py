import os
import sys
import glob
import random
import json
import textwrap
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from tqdm import tqdm 
import re

import fire
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset

from PIL import Image, ImageDraw, ImageFont, ImageFile, ImageOps, ImageFilter, ImageEnhance
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoProcessor, 
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    AutoConfig,
    set_seed
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
import types

from datasets import load_from_disk, load_dataset
from safetensors.torch import load_file


try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def is_rank_0():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def rank0_print(*args, **kwargs):
    """Print only on rank 0."""
    if is_rank_0():
        print(*args, **kwargs)

# ==============================================================================
# FontLibrary (Kept Same)
# ==============================================================================
class FontLibrary:
    def __init__(self, fonts_root: str):
        self.fonts_root = fonts_root
        self.font_map = {
            "Arabic": [], "Chinese": [], "Japanese": [], "Korean": [],
            "English": [], "Code": [], "Universal": []
        }
        self.chinese_safe_map = [] 
        self.universal_specific_map = {} 
        self._scan_fonts()
        self._index_universal_fonts()

    def _scan_fonts(self):
        if not os.path.exists(self.fonts_root):
            rank0_print(f"Warning: Fonts directory {self.fonts_root} not found.")
            return

        structure = {
            "Arabic": ["Arabic"],
            "Chinese": ["Chinese_Simplified"],
            "Japanese": ["Japanese"],
            "Korean": ["Korean"],
            "English": ["English_Handwriting", "English_Serif_Sans"],
            "Code": ["Code_Monospace"],
            "Universal": ["Universal_Coverage"]
        }

        for key, folders in structure.items():
            for folder in folders:
                path = os.path.join(self.fonts_root, folder, "*.ttf")
                found = glob.glob(path)
                self.font_map[key].extend(found)
        
        for f_path in self.font_map["Chinese"]:
            f_name = os.path.basename(f_path).lower()
            if "noto" in f_name and "sc" in f_name:
                self.chinese_safe_map.append(f_path)
        
        total = sum(len(v) for v in self.font_map.values())
        rank0_print(f"[FontLib] Loaded total: {total} fonts.")

    def _index_universal_fonts(self):
        for font_path in self.font_map["Universal"]:
            filename = os.path.basename(font_path).lower()
            keywords = [
                "armenian", "bengali", "canadian", "cherokee", "devanagari", 
                "ethiopic", "georgian", "gujarati", "gurmukhi", "hebrew", 
                "kannada", "khmer", "lao", "malayalam", "mongolian", "myanmar", 
                "oriya", "sinhala", "syriac", "tamil", "telugu", "thaana", "thai",
                "math", "symbols", "nko"
            ]
            for kw in keywords:
                if kw in filename:
                    if kw not in self.universal_specific_map:
                        self.universal_specific_map[kw] = []
                    self.universal_specific_map[kw].append(font_path)

    def _is_simple_simplified_chinese(self, text: str) -> bool:
        try:
            text.encode('gb2312')
            return True
        except UnicodeEncodeError:
            return False

    def detect_best_font(self, text: str) -> tuple:
        # print(text)
        if not text or text.strip() == "":
            candidates = self.font_map["Code"]
            if candidates:
                return random.choice(candidates), "Empty_Fallback"
            return self._get_random_fallback(), "Empty_Fallback"

        if re.search(r'[\u0600-\u06ff]', text) and self.font_map["Arabic"]:
            return random.choice(self.font_map["Arabic"]), "Arabic"
        if re.search(r'[\uac00-\ud7af]', text) and self.font_map["Korean"]:
            return random.choice(self.font_map["Korean"]), "Korean"
        if re.search(r'[\u3040-\u30ff]', text) and self.font_map["Japanese"]:
            return random.choice(self.font_map["Japanese"]), "Japanese"
        if re.search(r'[\u4e00-\u9fff]', text):
            if self._is_simple_simplified_chinese(text) and self.font_map["Chinese"]:
                return random.choice(self.font_map["Chinese"]), "Chinese_Art"
            else:
                if self.chinese_safe_map:
                    return random.choice(self.chinese_safe_map), "Chinese_Trad_Safe"
                elif self.font_map["Chinese"]:
                    return random.choice(self.font_map["Chinese"]), "Chinese_Fallback"

        uni_rules = [
            (r'[\u0E00-\u0E7F]', "thai"), (r'[\u0590-\u05FF]', "hebrew"),          
            (r'[\u0900-\u097F]', "devanagari"), (r'[\u0980-\u09FF]', "bengali"),        
            (r'[\u1000-\u109F]', "myanmar"), (r'[\u1780-\u17FF]', "khmer"),           
            (r'[\u0E80-\u0EFF]', "lao"), (r'[\u1200-\u137F]', "ethiopic"),        
            (r'[\u0530-\u058F]', "armenian"), (r'[\u0D80-\u0DFF]', "sinhala"),
        ]

        for pattern, keyword in uni_rules:
            if re.search(pattern, text):
                candidates = self.universal_specific_map.get(keyword, [])
                if candidates:
                    return random.choice(candidates), f"Universal_{keyword.title()}"
        
        fallback_pool = self.font_map["English"] + self.font_map["Code"]
        if fallback_pool:
            return random.choice(fallback_pool), "Latin_Code"
            
        if self.font_map["Universal"]:
            return self.font_map["Universal"][0], "Final_Fallback"
            
        return self._get_random_fallback(), "System_Fallback"

    def _get_random_fallback(self):
        all_fonts = [f for cats in self.font_map.values() for f in cats]
        if all_fonts:
            return random.choice(all_fonts)
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ==============================================================================
# New Dataset Class for Triplet Text (AllNLI style)
# ==============================================================================
class TripletTextDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = []
        
        rank0_print(f"Loading Triplet dataset from: {data_path}")
        
        # Support loading from disk (JSON/JSONL) or HF Hub logic if needed
        if os.path.isdir(data_path):
            # If it's a saved HF dataset
            if os.path.exists(os.path.join(data_path, "dataset_info.json")):
                 ds = load_from_disk(data_path)
                 if "train" in ds: ds = ds["train"]
                 self.dataset = ds
            else:
                # Load JSON files
                files = glob.glob(os.path.join(data_path, "*.json")) + glob.glob(os.path.join(data_path, "*.jsonl"))
                for f_path in files:
                    with open(f_path, 'r', encoding='utf-8') as f:
                        if f_path.endswith('.jsonl'):
                            for line in f:
                                self.dataset.append(json.loads(line))
                        else:
                            data = json.load(f)
                            if isinstance(data, list): self.dataset.extend(data)
        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
             with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    for line in f:
                        self.dataset.append(json.loads(line))
                else:
                    self.dataset = json.load(f)
        
        # Validation to ensure triplets exist
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if not all(k in sample for k in ["anchor", "positive", "negative"]):
                # Fallback for standard NLI columns if names differ
                if "sentence1" in sample and "sentence2" in sample and "sentence3" in sample:
                    rank0_print("Detected 'sentence1/2/3' format. Mapping to anchor/pos/neg.")
                    self.mapping = True
                else:
                    raise ValueError("Dataset must contain keys: 'anchor', 'positive', 'negative' (or sentence1/2/3)")
            else:
                self.mapping = False

        if is_rank_0():
            rank0_print(f"Triplet Dataset Ready. Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if getattr(self, "mapping", False):
            anchor = str(item.get('sentence1', ''))
            positive = str(item.get('sentence2', ''))
            negative = str(item.get('sentence3', ''))
        else:
            anchor = str(item.get('anchor', ''))
            positive = str(item.get('positive', ''))
            negative = str(item.get('negative', ''))
        
        return {
            "anchor": anchor,       
            "positive": positive,
            "negative": negative
        }



@dataclass
class DataCollatorForTripletWithRendering:
    processor: AutoProcessor
    max_length: int
    
    dtd_bg_path: str = None
    bg_prob: float = 0.5
    fonts_dir: str = "./font_lib"
    debug_output_dir: str = "./debug_samples"

    def __post_init__(self):
        # 1. Initialize Backgrounds
        self.bg_images = []
        if self.dtd_bg_path and os.path.exists(self.dtd_bg_path):
            patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
            for pat in patterns:
                self.bg_images.extend(glob.glob(os.path.join(self.dtd_bg_path, pat), recursive=True))
            rank0_print(f"Loaded {len(self.bg_images)} background images.")

        # 2. Initialize Font Lib
        self.font_lib = FontLibrary(self.fonts_dir)
        
        # 3. Debug Setup
        if is_rank_0() and not SWANLAB_AVAILABLE and self.debug_output_dir and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir, exist_ok=True)
        self.debug_counter = 0

    def _get_contrasting_color(self, bg_img: Image.Image) -> Tuple[int, int, int]:
        try:
            mean_color = bg_img.resize((1, 1)).getpixel((0, 0))
            if isinstance(mean_color, int): # Grayscale
                brightness = mean_color
            else:
                brightness = 0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]
            
            if brightness > 128:
                base_val = 0 
            else:
                base_val = 255
            
            jitter = 60
            r = min(255, max(0, base_val + random.randint(-jitter, jitter)))
            g = min(255, max(0, base_val + random.randint(-jitter, jitter)))
            b = min(255, max(0, base_val + random.randint(-jitter, jitter)))
            return (r, g, b)
        except:
            return (0, 0, 0)

    def text_to_image(self, text: str) -> Image.Image:
        image_size = (224, 224)
        
        # Background
        use_bg_texture = False
        if self.bg_images and random.random() < self.bg_prob:
            use_bg_texture = True

        bg_img = None
        if use_bg_texture:
            try:
                bg_path = random.choice(self.bg_images)
                bg_raw = Image.open(bg_path).convert('RGB')
                bg_img = ImageOps.fit(bg_raw, image_size, method=Image.BICUBIC)
                if random.random() < 0.5: bg_img = ImageOps.mirror(bg_img)
                enhancer = ImageEnhance.Brightness(bg_img)
                bg_img = enhancer.enhance(random.uniform(0.6, 1.4))
            except Exception:
                bg_img = None
        
        if bg_img is None:
            random_bg_color = (random.randint(50, 240), random.randint(50, 240), random.randint(50, 240))
            bg_img = Image.new('RGB', image_size, color=random_bg_color)

        # Fonts
        if not text:
            text = " "
            
        font_path, font_category = self.font_lib.detect_best_font(text)
        font_size = random.randint(16, 28) 
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            font = ImageFont.load_default()
            font_category = "Fallback"

        # Render
        text_color = self._get_contrasting_color(bg_img)
        temp_w, temp_h = int(image_size[0] * 1.5), int(image_size[1] * 1.5)
        txt_layer = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        avg_char_width = font_size * 0.7 if "Chinese" in font_category or "Japanese" in font_category else font_size * 0.5
        chars_per_line = max(5, int(image_size[0] * 0.9 / avg_char_width))
        
        if len(text) > 1000:
            text = text[:1000]
            
        lines = textwrap.wrap(text, width=chars_per_line)
        lines = lines[:12]
        
        total_text_height = len(lines) * (font_size + 5)
        start_y = (temp_h - total_text_height) // 2 
        
        stroke_width = 1 if random.random() < 0.4 else 0
        
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                w_line = bbox[2] - bbox[0]
            except:
                w_line = len(line) * font_size * 0.5
            
            start_x = (temp_w - w_line) // 2
            draw.text((start_x, start_y), line, font=font, fill=text_color + (255,), stroke_width=stroke_width, stroke_fill=text_color+(255,))
            start_y += int(font_size * 1.2)

        # Transforms
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            txt_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=False)

        # Composite
        paste_x = (image_size[0] - temp_w) // 2 + random.randint(-20, 20)
        paste_y = (image_size[1] - temp_h) // 2 + random.randint(-20, 20)
        
        bg_img.paste(txt_layer, (paste_x, paste_y), mask=txt_layer)

        # Post-process
        if random.random() < 0.2:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        
        return bg_img

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        
        def process_batch_key(key_name):
            texts = [f[key_name] for f in features]
            pixel_values_list = []
            grid_thw_list = []
            
            for idx, txt in enumerate(texts):
                # Always render text to image
                img = self.text_to_image(txt)
                
                # Debug Logging
                if is_rank_0() and self.debug_counter < 30 and key_name in ["anchor", "positive", "negative"]:
                    if SWANLAB_AVAILABLE:
                         swanlab.log({
                            f"Debug/{key_name}": swanlab.Image(img, caption=txt[:50])
                        }, step=0)
                    elif self.debug_output_dir:
                        try:
                            img.save(os.path.join(self.debug_output_dir, f"{self.debug_counter}_{key_name}.jpg"))
                        except: pass
                
                if key_name == "negative":
                    self.debug_counter += 1

                w, h = img.size
                factor = 28
                MAX_SIDE = 224
                new_w = min(MAX_SIDE, math.ceil(w / factor) * factor)
                new_h = min(MAX_SIDE, math.ceil(h / factor) * factor)
                new_w = max(56, new_w)
                new_h = max(56, new_h)
                img = img.resize((new_w, new_h), resample=Image.BICUBIC)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                out = self.processor(
                    images=[img],
                    text=[text_input],
                    return_tensors="pt",
                    padding=False,
                    do_resize=False
                )
                pixel_values_list.append(out["pixel_values"])
                grid_thw_list.append(out["image_grid_thw"])
            
            return {
                "pixel_values": torch.cat(pixel_values_list, dim=0),
                "image_grid_thw": torch.stack(grid_thw_list, dim=0)
            }

        # Process Anchor, Positive, Negative independently
        anchor_out = process_batch_key("anchor")
        positive_out = process_batch_key("positive")
        negative_out = process_batch_key("negative")

        return {
            "anchor_pixel_values": anchor_out["pixel_values"],
            "anchor_image_grid_thw": anchor_out["image_grid_thw"],
            "positive_pixel_values": positive_out["pixel_values"],
            "positive_image_grid_thw": positive_out["image_grid_thw"],
            "negative_pixel_values": negative_out["pixel_values"],
            "negative_image_grid_thw": negative_out["image_grid_thw"],
        }


class TripletContrastiveTrainer(Trainer):
    def _get_embeddings(self, model, inputs: Dict[str, Any], prefix: str) -> Tuple[torch.Tensor, Any]:
        pixel_values = inputs.get(f"{prefix}_pixel_values")
        grid_thw = inputs.get(f"{prefix}_image_grid_thw")
        
        if pixel_values is None:
            return None, None

        pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)
        grid_thw = grid_thw.to(device=model.device)
        if grid_thw.dim() == 3 and grid_thw.size(1) == 1:
            grid_thw = grid_thw.squeeze(1)

        outputs = model(hidden_states=pixel_values, grid_thw=grid_thw)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs

        batch_size = grid_thw.shape[0]
        H = grid_thw[:, 1]
        W = grid_thw[:, 2]
        output_H = H // 2
        output_W = W // 2
        sizes = (output_H * output_W).long()
        
        total_tokens = hidden_states.shape[0]
        calculated_total = sizes.sum().item()
        if calculated_total != total_tokens:
             sizes[-1] += (total_tokens - calculated_total)

        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=hidden_states.device), sizes
        )

        pooled_sum = torch.zeros(
            (batch_size, hidden_states.shape[-1]), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        pooled_sum.index_add_(0, batch_indices, hidden_states)
        counts = sizes.unsqueeze(1).to(dtype=hidden_states.dtype)
        counts = torch.clamp(counts, min=1.0)
        embeds = pooled_sum / counts
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds, outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implementation of Multiple Negatives Ranking Loss with Hard Negatives.
        Loss = One-way Cross Entropy from Anchor to [Positives + Negatives]
        """
        if self.args.gradient_checkpointing and model.training:
            if "anchor_pixel_values" in inputs: inputs["anchor_pixel_values"].requires_grad_(True)
            if "positive_pixel_values" in inputs: inputs["positive_pixel_values"].requires_grad_(True)
            if "negative_pixel_values" in inputs: inputs["negative_pixel_values"].requires_grad_(True)
        
        anchor_embeds, _ = self._get_embeddings(model, inputs, "anchor")
        positive_embeds, _ = self._get_embeddings(model, inputs, "positive")
        negative_embeds, _ = self._get_embeddings(model, inputs, "negative")
        
        # Gather across devices for DDP
        if dist.is_initialized():
            def gather_tensor(t):
                gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=gathered, tensor=t.contiguous())
                gathered[dist.get_rank()] = t
                return torch.cat(gathered, 0)

            all_anchors = gather_tensor(anchor_embeds)
            all_positives = gather_tensor(positive_embeds)
            all_negatives = gather_tensor(negative_embeds)
        else:
            all_anchors = anchor_embeds
            all_positives = positive_embeds
            all_negatives = negative_embeds
        
        logit_scale = 1.0 / 0.05
        
        # Concatenate Positives and Negatives for the denominator candidates
        # Shape: (Batch_Size, 2 * Batch_Size)
        # The candidates are [P1, P2... Pn, N1, N2... Nn]
        candidates = torch.cat([all_positives, all_negatives], dim=0)
        
        # Calculate similarity logits: Anchor vs All Candidates
        # Shape: (Batch_Size, 2 * Batch_Size)
        scores = torch.matmul(all_anchors, candidates.t()) * logit_scale
        
        # The target for Anchor[i] is Positive[i].
        # In the candidates list, Positive[i] is at index i.
        labels = torch.arange(all_anchors.size(0), device=model.device)
        
        # Cross Entropy Loss
        # This effectively learns: Similarity(A, P) > Similarity(A, N) AND Similarity(A, P) > Similarity(A, other_P/N)
        loss = F.cross_entropy(scores, labels)
        
        if is_rank_0():
            if SWANLAB_AVAILABLE and self.state.global_step % self.args.logging_steps == 0:
                swanlab.log({
                    "loss": loss.item(),
                }, step=self.state.global_step)

        return (loss, None) if return_outputs else loss


def train(
    base_model: str = "./Qwen2.5-VL-ViT-Only",
    data_path: str = "./data/all_nli_triplets", # Path to triplet dataset
    dtd_data_path: str = "./data/dtd/extracted_images", 
    fonts_dir: str = "./font_lib", 
    dtd_prob: float = 0.5,
    output_dir: str = "./AllNLI_ViT_Finetune",
    base_config_path: str = None,
    batch_size: int = 1056,
    per_device_batch_size: int = 4, 
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
    max_length: int = 10000, 
    save_steps: int = 500,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    bf16: bool = True,
    ):

    # DDP setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = batch_size // per_device_batch_size // world_size
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    else:
        device_map = "auto"
        gradient_accumulation_steps = batch_size // per_device_batch_size

    set_seed(seed)

    if is_rank_0() and SWANLAB_AVAILABLE:
        config = {k: v for k, v in locals().items() if k not in ["device_map"]}
        swanlab.init(
            experiment_name=os.path.basename(output_dir), 
            config=config,
            log_dir=os.path.join(output_dir, "swanlab")
        )

    cfg_src = base_config_path or base_model
    rank0_print(f"Loading ViT Config from {cfg_src}...")
    cfg_json = os.path.join(cfg_src, "config.json")
    if os.path.isfile(cfg_json):
        raw_cfg = json.load(open(cfg_json))
    else:
        raw_cfg = {}
    if "depth" in raw_cfg:
        # A ViT-only export or a resumed checkpoint: config.json already *is* the
        # vision tower config. Going through AutoConfig here would silently
        # substitute a default vision_config with the wrong hidden_size.
        config = Qwen2_5_VLVisionConfig(**raw_cfg)
    else:
        config = AutoConfig.from_pretrained(cfg_src).vision_config

    
    rank0_print("Initializing model structure from config...")
    model = Qwen2_5_VisionTransformerPretrainedModel(config)
    
    bin_path = os.path.join(base_model, "pytorch_model.bin")
    safe_path = os.path.join(base_model, "model.safetensors")

    if os.path.exists(bin_path):
        rank0_print(f"Found pytorch_model.bin. Loading from {bin_path}...")
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        
    elif os.path.exists(safe_path):
        rank0_print(f"pytorch_model.bin not found. Found model.safetensors. Loading from {safe_path}...")
        state_dict = load_file(safe_path)
        model.load_state_dict(state_dict, strict=True)
    
    if bf16:
        model = model.to(torch.bfloat16)

    model.requires_grad_(True)

    rank0_print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    rank0_print(f"Initializing Triplet Dataset...")
    full_dataset = TripletTextDataset(data_path=data_path)
    
    total_images = len(full_dataset)
    steps_per_epoch = total_images // batch_size
    rank0_print(f"Total samples: {total_images}. Steps per Epoch: {steps_per_epoch}")
    
    data_collator = DataCollatorForTripletWithRendering(
        processor=processor,
        max_length=max_length,
        dtd_bg_path=dtd_data_path,
        bg_prob=dtd_prob,
        fonts_dir=fonts_dir,
        debug_output_dir=os.path.join(output_dir, "debug_samples")
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=100,
        logging_steps=logging_steps,
        bf16=bf16,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=False,
        run_name=os.path.basename(output_dir),
        deepspeed=deepspeed,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none", 
        max_steps=steps_per_epoch * num_epochs 
    )

    trainer = TripletContrastiveTrainer(
        model=model,
        train_dataset=full_dataset, 
        args=training_args,
        data_collator=data_collator
    )
    
    rank0_print("Starting fine-tuning with Hard Negatives...")
    trainer.train()
    
    if is_rank_0():
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        if SWANLAB_AVAILABLE:
            swanlab.finish()

if __name__ == "__main__":
    fire.Fire(train)