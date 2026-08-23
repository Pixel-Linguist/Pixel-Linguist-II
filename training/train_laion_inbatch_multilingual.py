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
from safetensors.torch import load_file
import types

from datasets import load_from_disk, load_dataset

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
# Improved FontLibrary (Ported from RenderDebugEngine)
# ==============================================================================
class FontLibrary:
    def __init__(self, fonts_root: str):
        self.fonts_root = fonts_root
        self.font_map = {
            "Arabic": [],
            "Chinese": [],
            "Japanese": [],
            "Korean": [],
            "English": [],
            "Code": [],
            "Universal": []
        }
        
        # Noto faces, kept apart as the "safe" Chinese set
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
            "Chinese": ["Chinese_Simplified"], # includes decorative faces as well as Noto
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
                
        # Separate out the safe Chinese fonts (Noto)
        for f_path in self.font_map["Chinese"]:
            f_name = os.path.basename(f_path).lower()
            if "noto" in f_name and "sc" in f_name:
                self.chinese_safe_map.append(f_path)
        
        total = sum(len(v) for v in self.font_map.values())
        rank0_print(f"[FontLib] Loaded total: {total} fonts.")
        rank0_print(f"[FontLib] Identified {len(self.chinese_safe_map)} Safe Chinese Fonts (Noto).")

    def _index_universal_fonts(self):
        """Index the Universal_Coverage filenames by language keyword."""
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
        """
        Whether the text is "safe" simplified Chinese.

        Tries to encode as GB2312, which covers the 6763 most common
        characters. Traditional, rare or unusual glyphs raise, giving False.
        """
        try:
            text.encode('gb2312')
            return True
        except UnicodeEncodeError:
            return False

    def detect_best_font(self, text: str) -> tuple:
        """
        Route the text to a suitable font.

        Returns: (font_path, font_category_name)
        """
        if not text or text.strip() == "":
            candidates = self.font_map["Code"]
            if candidates:
                return random.choice(candidates), "Empty_Fallback"
            return self._get_random_fallback(), "Empty_Fallback"

        # --- 1. Detect CJK and the major scripts first ---
        
        # Arabic
        if re.search(r'[\u0600-\u06ff]', text) and self.font_map["Arabic"]:
            return random.choice(self.font_map["Arabic"]), "Arabic"

        # Korean
        if re.search(r'[\uac00-\ud7af]', text) and self.font_map["Korean"]:
            return random.choice(self.font_map["Korean"]), "Korean"

        # Japanese (Kana) - must be checked before Han
        if re.search(r'[\u3040-\u30ff]', text) and self.font_map["Japanese"]:
            return random.choice(self.font_map["Japanese"]), "Japanese"

        # Chinese CJK (4E00-9FFF)
        if re.search(r'[\u4e00-\u9fff]', text):
            if self._is_simple_simplified_chinese(text) and self.font_map["Chinese"]:
                return random.choice(self.font_map["Chinese"]), "Chinese_Art"
            else:
                if self.chinese_safe_map:
                    return random.choice(self.chinese_safe_map), "Chinese_Trad_Safe"
                elif self.font_map["Chinese"]:
                    return random.choice(self.font_map["Chinese"]), "Chinese_Fallback"

        # --- 2. Universal Coverage ---
        uni_rules = [
            (r'[\u0E00-\u0E7F]', "thai"),           
            (r'[\u0590-\u05FF]', "hebrew"),         
            (r'[\u0900-\u097F]', "devanagari"),     
            (r'[\u0980-\u09FF]', "bengali"),        
            (r'[\u0A80-\u0AFF]', "gujarati"),       
            (r'[\u0A00-\u0A7F]', "gurmukhi"),       
            (r'[\u0C80-\u0CFF]', "kannada"),        
            (r'[\u0D00-\u0D7F]', "malayalam"),      
            (r'[\u0B00-\u0B7F]', "oriya"),          
            (r'[\u0B80-\u0BFF]', "tamil"),          
            (r'[\u0C00-\u0C7F]', "telugu"),         
            (r'[\u1000-\u109F]', "myanmar"),        
            (r'[\u1780-\u17FF]', "khmer"),          
            (r'[\u0E80-\u0EFF]', "lao"),            
            (r'[\u10A0-\u10FF]', "georgian"),       
            (r'[\u1200-\u137F]', "ethiopic"),       
            (r'[\u0530-\u058F]', "armenian"),       
            (r'[\u0D80-\u0DFF]', "sinhala"),        
            (r'[\u0780-\u07BF]', "thaana"),         
            (r'[\u1800-\u18AF]', "mongolian"),      
            (r'[\u13A0-\u13FF]', "cherokee"),       
            (r'[\u1400-\u167F]', "canadian"),       
        ]

        for pattern, keyword in uni_rules:
            if re.search(pattern, text):
                candidates = self.universal_specific_map.get(keyword, [])
                if candidates:
                    return random.choice(candidates), f"Universal_{keyword.title()}"
        
        # --- 3. Fallback ---
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


class JsonTripletDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(current_dir, "merged_dataset_cache.json")
        
        is_main_process = is_rank_0()

        if is_main_process:
            if os.path.exists(cache_file):
                rank0_print(f"[Rank 0] Found cache file at {cache_file}. Loading directly...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
            else:
                rank0_print(f"[Rank 0] Cache not found. Building from {data_path}...")
                self.samples = self._build_and_save_cache(cache_file)
            
            if dist.is_initialized():
                dist.barrier() 
        else:
            if dist.is_initialized():
                dist.barrier() 
            
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
            except FileNotFoundError:
                raise RuntimeError(f"Rank {dist.get_rank()} could not find cache file.")

        if is_rank_0():
            rank0_print(f"Image-Text Dataset Ready. Total valid samples: {len(self.samples)}")

    def _build_and_save_cache(self, cache_file_path):
        json_files = sorted(glob.glob(os.path.join(self.data_path, "*.json")))
        if not json_files:
            raise ValueError(f"No .json files found in {self.data_path}")
        
        rank0_print(f"Found {len(json_files)} json files. Processing...")
        
        valid_samples = []
        count_invalid = 0

        iterator = tqdm(json_files, desc="Merging JSONs", unit="file") if is_rank_0() else json_files

        for json_file in iterator:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    
                if isinstance(data_list, dict):
                    data_list = [data_list]
                
                for item in data_list:
                    img_rel_path = item.get("image")
                    text = item.get("text")
                    
                    if img_rel_path and text:
                        full_img_path = os.path.join(self.data_path, img_rel_path)
                        valid_samples.append({
                            "image_path": full_img_path,
                            "text": text
                        })
                    else:
                        count_invalid += 1

            except Exception as e:
                if is_rank_0():
                    tqdm.write(f"Error reading {json_file}: {e}")
                continue
        
        rank0_print(f"Saving merged dataset ({len(valid_samples)} samples) to {cache_file_path}...")
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False)
            
        rank0_print(f"Cache saved. (Skipped {count_invalid} invalid items)")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        anchor_path = item['image_path']
        pos_text = item['text']

        return {
            "anchor": {"image_path": anchor_path},
            "positive": {"text": pos_text}
        }


class TextPairDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        try:
            rank0_print(f"Loading Text-Text dataset from disk: {data_path}")
            if os.path.exists(os.path.join(data_path, "dataset_info.json")):
                self.dataset = load_from_disk(data_path)
            else:
                self.dataset = load_dataset(data_path, split="train")
        except Exception as e:
            rank0_print(f"Error loading text dataset: {e}")
            raise e

        if hasattr(self.dataset, "keys") and "train" in self.dataset.keys():
             self.dataset = self.dataset["train"]
        
        if is_rank_0():
            rank0_print(f"Text-Text Dataset Ready. Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        s1 = str(item.get('sentence1', ''))
        s2 = str(item.get('sentence2', ''))
        
        return {
            "anchor": {"text": s1},       
            "positive": {"text": s2}
        }


@dataclass
class DataCollatorForAny2AnyContrastive:
    processor: AutoProcessor
    max_length: int
    
    dtd_bg_path: str = None
    bg_prob: float = 0.5
    fonts_dir: str = "./font_lib"
    debug_output_dir: str = "./debug_samples"

    def __post_init__(self):
        # 1. Background images
        self.bg_images = []
        if self.dtd_bg_path and os.path.exists(self.dtd_bg_path):
            patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
            for pat in patterns:
                self.bg_images.extend(glob.glob(os.path.join(self.dtd_bg_path, pat), recursive=True))
            rank0_print(f"Loaded {len(self.bg_images)} background images.")

        # 2. Font library
        self.font_lib = FontLibrary(self.fonts_dir)
        
        # 3. Debug settings
        if is_rank_0() and not SWANLAB_AVAILABLE and self.debug_output_dir and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir, exist_ok=True)
        self.debug_counter = 0

    def _get_contrasting_color(self, bg_img: Image.Image) -> Tuple[int, int, int]:
        """Pick a text colour that contrasts with the background."""
        try:
            mean_color = bg_img.resize((1, 1)).getpixel((0, 0))
            if isinstance(mean_color, int): # Grayscale
                brightness = mean_color
            else:
                brightness = 0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]
            
            # Light background -> dark text and vice versa, with jitter to avoid uniformity
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
        """
        Robust Text-to-Image Rendering
        Draws on an oversized canvas, then rotates and pastes, which avoids
        tofu boxes and clipping at the edges.
        """
        image_size = (224, 224)
        
        # --- 1. Background ---
        use_bg_texture = False
        if self.bg_images and random.random() < self.bg_prob:
            use_bg_texture = True

        bg_img = None
        if use_bg_texture:
            try:
                bg_path = random.choice(self.bg_images)
                bg_raw = Image.open(bg_path).convert('RGB')
                bg_img = ImageOps.fit(bg_raw, image_size, method=Image.BICUBIC)
                # Augmentation
                if random.random() < 0.5: bg_img = ImageOps.mirror(bg_img)
                enhancer = ImageEnhance.Brightness(bg_img)
                bg_img = enhancer.enhance(random.uniform(0.6, 1.4))
            except Exception:
                bg_img = None
        
        if bg_img is None:
            # Solid colour background with noise
            random_bg_color = (random.randint(50, 240), random.randint(50, 240), random.randint(50, 240))
            bg_img = Image.new('RGB', image_size, color=random_bg_color)

        # --- 2. Font selection ---
        if not text:
            text = " "
            
        font_path, font_category = self.font_lib.detect_best_font(text)
        font_size = random.randint(16, 28) 
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            font = ImageFont.load_default()
            font_category = "Fallback"

        # --- 3. Draw the text onto a temporary layer ---
        text_color = self._get_contrasting_color(bg_img)
        
        # Temporary layer 1.5x the target size so rotation does not clip
        temp_w, temp_h = int(image_size[0] * 1.5), int(image_size[1] * 1.5)
        txt_layer = Image.new('RGBA', (temp_w, temp_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Line wrapping, using a per-script character width estimate
        avg_char_width = font_size * 0.7 if "Chinese" in font_category or "Japanese" in font_category else font_size * 0.5
        chars_per_line = max(5, int(image_size[0] * 0.9 / avg_char_width))
        
        # Cap the text length to keep rendering fast
        if len(text) > 1000:
            text = text[:1000]
            
        lines = textwrap.wrap(text, width=chars_per_line)
        lines = lines[:12] # at most 12 lines, to avoid overflow
        
        total_text_height = len(lines) * (font_size + 5)
        start_y = (temp_h - total_text_height) // 2 
        
        stroke_width = 1 if random.random() < 0.4 else 0
        
        for line in lines:
            try:
                # Text bounding box
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                w_line = bbox[2] - bbox[0]
            except:
                w_line = len(line) * font_size * 0.5
            
            # Centre horizontally
            start_x = (temp_w - w_line) // 2
            
            draw.text((start_x, start_y), line, font=font, fill=text_color + (255,), stroke_width=stroke_width, stroke_fill=text_color+(255,))
            start_y += int(font_size * 1.2)

        # --- 4. Geometric augmentation ---
        # Rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            txt_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=False)

        # --- 5. Composite ---
        # Paste position: centred, with a random offset
        paste_x = (image_size[0] - temp_w) // 2 + random.randint(-20, 20)
        paste_y = (image_size[1] - temp_h) // 2 + random.randint(-20, 20)
        
        # Alpha Composite paste
        bg_img.paste(txt_layer, (paste_x, paste_y), mask=txt_layer)

        # --- 6. Post-processing filters ---
        # Gaussian blur
        if random.random() < 0.2:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        
        return bg_img

    def _process_item_to_image(self, item_data: Dict[str, Any]) -> Image.Image:
        pil_image = None
        if item_data.get("pil_image") is not None:
            pil_image = item_data["pil_image"]
        elif item_data.get("image_path"):
            try:
                pil_image = Image.open(item_data["image_path"])
            except:
                pass
        
        if pil_image is not None:
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

        # Render the text to an image
        if pil_image is None and item_data.get("text"):
            text_to_render = item_data["text"]
            pil_image = self.text_to_image(text_to_render)
            
        if pil_image is None:
             pil_image = Image.new('RGB', (224, 224), (255, 255, 255))
        return pil_image

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        def process_image_list_one_by_one(raw_data_list, prefix_name):
            pixel_values_list = []
            grid_thw_list = []
            
            for idx, item in enumerate(raw_data_list):
                img = self._process_item_to_image(item)
                
                # SwanLab Logging Logic (Debug)
                if is_rank_0():
                    if SWANLAB_AVAILABLE and prefix_name in ["anchor", "positive"] and self.debug_counter < 20:
                        if self.debug_counter == 0:
                            print(f"Logging debug images to SwanLab...")
                        swanlab.log({
                            f"Debug Images/{prefix_name}_{self.debug_counter}": swanlab.Image(
                                img, 
                                caption=item.get("text", "Text Image")[:100]
                            )
                        }, step=0)
                        self.debug_counter += 1
                    
                    if not SWANLAB_AVAILABLE and prefix_name in ["anchor", "positive"] and self.debug_counter < 16:
                        if self.debug_output_dir:
                            save_name = os.path.join(self.debug_output_dir, f"aug_{prefix_name}_{self.debug_counter}.jpg")
                            try:
                                img.save(save_name)
                            except:
                                pass
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

        anchors = [f["anchor"] for f in features]
        positives = [f["positive"] for f in features]
        
        anchor_out = process_image_list_one_by_one(anchors, "anchor")
        positive_out = process_image_list_one_by_one(positives, "positive")

        return {
            "positive_pixel_values": positive_out["pixel_values"],
            "positive_image_grid_thw": positive_out["image_grid_thw"],
            "anchor_pixel_values": anchor_out["pixel_values"],
            "anchor_image_grid_thw": anchor_out["image_grid_thw"],
        }


class ContrastiveTrainer(Trainer):
    def _get_embeddings(self, model, inputs: Dict[str, Any], prefix: str) -> Tuple[torch.Tensor, Any]:
        pixel_values = inputs.get(f"{prefix}_pixel_values")
        grid_thw = inputs.get(f"{prefix}_image_grid_thw")
        
        # Guard
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
        if self.args.gradient_checkpointing and model.training:
            if "anchor_pixel_values" in inputs: inputs["anchor_pixel_values"].requires_grad_(True)
            if "positive_pixel_values" in inputs: inputs["positive_pixel_values"].requires_grad_(True)
        
        anchor_embeds, anchor_outputs = self._get_embeddings(model, inputs, "anchor")
        positive_embeds, positive_outputs = self._get_embeddings(model, inputs, "positive")
        
        if dist.is_initialized():
            anchor_embeds_list = [torch.zeros_like(anchor_embeds) for _ in range(dist.get_world_size())]
            positive_embeds_list = [torch.zeros_like(positive_embeds) for _ in range(dist.get_world_size())]
            
            dist.all_gather(tensor_list=anchor_embeds_list, tensor=anchor_embeds.contiguous())
            dist.all_gather(tensor_list=positive_embeds_list, tensor=positive_embeds.contiguous())

            anchor_embeds_list[dist.get_rank()] = anchor_embeds
            positive_embeds_list[dist.get_rank()] = positive_embeds
            
            all_anchors = torch.cat(anchor_embeds_list, 0)
            all_positives = torch.cat(positive_embeds_list, 0)
        else:
            all_anchors = anchor_embeds
            all_positives = positive_embeds
        
        logit_scale = 1.0 / 0.03
        
        logits_per_anchor = torch.matmul(all_anchors, all_positives.t()) * logit_scale
        logits_per_positive = logits_per_anchor.t()
        
        labels = torch.arange(all_anchors.size(0), device=model.device)
        
        loss_i2t = F.cross_entropy(logits_per_anchor, labels)
        loss_t2i = F.cross_entropy(logits_per_positive, labels)

        loss = (loss_i2t + loss_t2i) / 2.0
        
        if is_rank_0():
            if SWANLAB_AVAILABLE and self.state.global_step % self.args.logging_steps == 0:
                swanlab.log({
                    "loss": loss.item(),
                    "loss_i2t": loss_i2t.item(),
                    "loss_t2i": loss_t2i.item()
                }, step=self.state.global_step)

        return (loss, (anchor_outputs, positive_outputs)) if return_outputs else loss


def train(
    base_model: str = "./Qwen2.5-VL-ViT-Only",
    data_path: str = "./data/cc3m-wds", 
    unsupervised_data_path: str = "./data/multilingual-filtered2", 
    dtd_data_path: str = "./data/dtd/extracted_images", 
    fonts_dir: str = "./font_lib", 
    dtd_prob: float = 0.5,
    output_dir: str = "./CC3M_ViT_Full",
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
        rank0_print(f"Loading weights from {bin_path}...")
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif os.path.exists(safe_path):
        rank0_print(f"Loading weights from {safe_path}...")
        state_dict = load_file(safe_path)
        model.load_state_dict(state_dict, strict=True)
    else:
        rank0_print(f"Warning: Weights not found in {base_model}, initializing randomly.")
    
    if bf16:
        model = model.to(torch.bfloat16)

    model.requires_grad_(True)

    rank0_print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    rank0_print(f"Initializing Datasets...")
    
    dataset_img_text = JsonTripletDataset(data_path=data_path)
    rank0_print(f"Image-Text samples: {len(dataset_img_text)}")
    
    dataset_text_text = None
    if unsupervised_data_path and os.path.exists(unsupervised_data_path):
        try:
            dataset_text_text = TextPairDataset(data_path=unsupervised_data_path)
            rank0_print(f"Text-Text samples: {len(dataset_text_text)}")
        except Exception as e:
            rank0_print(f"Warning: Failed to load unsupervised dataset: {e}")
    
    # 3. Concatenate
    if dataset_text_text is not None:
        full_dataset = ConcatDataset([dataset_img_text, dataset_text_text])
        rank0_print(f"Datasets combined. Total samples: {len(full_dataset)}")
    else:
        full_dataset = dataset_img_text
        rank0_print("Using only Image-Text dataset.")

    total_images = len(full_dataset)
    steps_per_epoch = total_images // batch_size
    rank0_print(f"Calculated Steps per Epoch: {steps_per_epoch}")
    
    data_collator = DataCollatorForAny2AnyContrastive(
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
        save_total_limit=5,
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

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=full_dataset, 
        args=training_args,
        data_collator=data_collator
    )
    
    rank0_print("Starting full fine-tuning (In-Batch Negatives)...")
    trainer.train()
    
    if is_rank_0():
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        if SWANLAB_AVAILABLE:
            swanlab.finish()

if __name__ == "__main__":
    fire.Fire(train)