from __future__ import annotations

from functools import partial
from typing import Any, Union
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from safetensors.torch import load_file
import torchvision.transforms as T

from PIL import Image, ImageDraw, ImageFont
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
import glob
import json
import re
import random
import textwrap


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
            print(f"Warning: Fonts directory {self.fonts_root} not found.")
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
        print(f"[FontLib] Loaded total: {total} fonts.")
        print(f"[FontLib] Identified {len(self.chinese_safe_map)} Safe Chinese Fonts (Noto).")

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

class Qwen2_5_VL_ViT_Wrapper1:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        fonts_root = kwargs.get("fonts_dir") or os.environ.get("PIXEL_FONTS_DIR", "./font_lib")

        # --- 1. Load Processor ---
        # This is a vision-only tower, so only the image processor is needed. It
        # yields pixel_values/image_grid_thw identical to the full AutoProcessor
        # called with a dummy text prompt, without requiring tokenizer files.
        self.processor = AutoImageProcessor.from_pretrained(
            kwargs.get("processor_path", model_path)
        )

        # --- 2. Load Model Config & Weights ---
        base_config_path = kwargs.get("base_config_path", model_path)
        cfg_json = os.path.join(base_config_path, "config.json")
        raw_cfg = json.load(open(cfg_json)) if os.path.isfile(cfg_json) else {}
        if "depth" in raw_cfg:
            # A ViT-only export already *is* the vision config; AutoConfig would
            # silently substitute a default vision_config with a wrong hidden_size.
            vision_config = Qwen2_5_VLVisionConfig(**raw_cfg)
        else:
            vision_config = AutoConfig.from_pretrained(base_config_path).vision_config
        self.model = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
        
        # Load weights (Support both bin and safetensors)
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        safe_path = os.path.join(model_path, "model.safetensors")

        if os.path.exists(bin_path):
            print(f"Loading weights from {bin_path}...")
            state_dict = torch.load(bin_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        elif os.path.exists(safe_path):
            print(f"Loading weights from {safe_path}...")
            state_dict = load_file(safe_path)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print(f"Warning: No custom weights found at {model_path}. Initializing random/pretrained base.")

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if os.path.exists(fonts_root):
            self.font_lib = FontLibrary(fonts_root)
        else:
            print(f"Warning: Fonts root {fonts_root} not found. Text rendering might fail or fallback to default.")
            self.font_lib = None

    def _batch_tensor_to_pil(self, batch_tensors: torch.Tensor) -> list[Image.Image]:
        """
        Converts a batch of tensors (B, C, H, W) back to a list of PIL Images.
        Assumes tensors are in [0, 1] range (standard ToTensor).
        """
        pil_images = []
        to_pil = T.ToPILImage()
        
        # Move to CPU for PIL conversion
        batch_tensors = batch_tensors.cpu()
        
        for i in range(batch_tensors.shape[0]):
            img_tensor = batch_tensors[i]
            # Un-normalize if necessary. 
            # Note: MIEB dataloaders usually output standard ToTensor (0-1).
            # If your dataloader applies mean/std norm, you must reverse it here.
            pil_images.append(to_pil(img_tensor))
        return pil_images

    def _process_forward_step(self, images: list[Image.Image]):
        """
        Executes the specific preprocessing and forward pass logic 
        defined in the custom training script.
        """
        # 1. Prepare Inputs with Chat Template
        inputs = self.processor(images=images, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        grid_thw = inputs["image_grid_thw"].to(self.device)

        # 2. Model Forward
        outputs = self.model(hidden_states=pixel_values, grid_thw=grid_thw)
        hidden_states = outputs

        # 3. Pooling Logic (Exact replica of training logic)
        if grid_thw.dim() == 3 and grid_thw.size(1) == 1:
            grid_thw = grid_thw.squeeze(1)

        batch_size = grid_thw.shape[0]
        
        # Calculate tokens per image based on grid dimensions (H//2 * W//2)
        H, W = grid_thw[:, 1], grid_thw[:, 2]
        sizes = ((H // 2) * (W // 2)).long()
        
        # Safety fix for token mismatch
        total_tokens = hidden_states.shape[0]
        if sizes.sum().item() != total_tokens:
            sizes[-1] += (total_tokens - sizes.sum().item())

        # Create batch indices
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), 
            sizes
        )

        # Sum Pooling
        pooled_sum = torch.zeros(
            (batch_size, hidden_states.shape[-1]), 
            dtype=self.dtype, 
            device=self.device
        )
        pooled_sum.index_add_(0, batch_indices, hidden_states)

        # Mean Pooling
        counts = sizes.unsqueeze(1).to(dtype=self.dtype).clamp(min=1.0)
        embeds = pooled_sum / counts

        # 4. Normalize
        embeds = F.normalize(embeds, p=2, dim=-1)
        
        return embeds.cpu()

    def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            # --- CASE 1: MIEB passes a DataLoader ---
            if isinstance(images, DataLoader):
                # 1. Capture the original transform so we can restore it later
                # (The dataset is likely wrapped in the DataLoader)
                dataset = images.dataset
                original_transform = getattr(dataset, "transform", None)

                # 2. THE FIX: Override the transform with an Identity function.
                # This makes the dataset return raw PIL images instead of Tensors.
                dataset.transform = lambda x: x
                
                # 3. Create a clean iterator
                # We use a custom collate because standard collate fails on variable-size PIL images
                # But the MIEB evaluator's 'custom_collate_fn' already returns a list, 
                # so we can actually just iterate the existing loader or make a new one.
                # Making a new one is safer to ensure 'batch_size' and settings are correct.
                new_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: x # Returns list of PIL images
                )

                try:
                    with torch.no_grad():
                        for batch in tqdm(new_loader, desc="Encoding Image Batches"):
                            # 'batch' is now a list of PIL Images.
                            # Qwen handles them directly!
                            
                            # Handle case where dataset returns tuples/dicts (unlikely in this specific evaluator, but good practice)
                            # The provided evaluator returns the image directly, so 'batch' is [PIL, PIL, ...]
                            
                            # Ensure RGB (Evaluator does this, but good to double check)
                            batch_images = [img.convert("RGB") if hasattr(img, "convert") else img for img in batch]

                            batch_embeds = self._process_forward_step(batch_images)
                            all_image_embeddings.append(batch_embeds)
                finally:
                    # 4. Restore the original transform (Politeness)
                    dataset.transform = original_transform

            else:
                return self.encode_batch(images, batch_size=batch_size)

            if not all_image_embeddings:
                return torch.empty(0)

            return torch.cat(all_image_embeddings, dim=0).to(torch.float16).cpu()


    def _render_text_clean(self, text: str) -> Image.Image:
        """
        Deterministic renderer used for evaluation:
        1. plain white background (255, 255, 255)
        2. plain black text (0, 0, 0)
        3. font chosen per script via FontLibrary
        4. centred, with none of the rotation/blur augmentation
        """
        image_size = (224, 224) # standard input size
        bg_img = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(bg_img)

        if not text:
            return bg_img

        # 1. Font selection
        if self.font_lib:
            font_path, _ = self.font_lib.detect_best_font(text)
        else:
            font_path = None # Fallback

        # Fixed size, for consistency across evaluation runs
        font_size = 16
        
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # 2. Wrap the text and compute centring
        # Estimate characters per line (CJK glyphs are wide, Latin narrow)
        is_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
        avg_char_width = font_size * 1.0 if is_cjk else font_size * 0.6
        chars_per_line = max(5, int(image_size[0] * 0.9 / avg_char_width))
        
        # Cap the text length to avoid OOM and overflow
        if len(text) > 500:
            text = text[:500]

        lines = textwrap.wrap(text, width=chars_per_line)
        # Cap the line count to stay inside the image
        lines = lines[: int(image_size[1] * 0.9 / (font_size + 5))] 

        # Total height, for vertical centring
        total_text_height = len(lines) * (font_size + 5)
        current_y = (image_size[1] - total_text_height) // 2

        for line in lines:
            # Per-line width, for horizontal centring
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                w_line = bbox[2] - bbox[0]
            except:
                w_line = len(line) * font_size * 0.5 # rough estimate

            start_x = (image_size[0] - w_line) // 2
            
            # Draw the text in black
            draw.text((start_x, current_y), line, font=font, fill=(0, 0, 0))
            current_y += int(font_size + 5)

        return bg_img


    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        Implementation of text embedding via rendering.
        Pipeline: Text -> Clean Image (Whiteboard) -> ViT Encoder -> Embedding
        """
        # print('in text embedding')
        all_text_embeddings = []

        # 1. Render every text to a PIL image
        # Rendering is fast; tqdm just shows progress
        rendered_images = []
        for t in texts:
            rendered_images.append(self._render_text_clean(t))

        # 2. Run inference in batches
        # Reuses _process_forward_step, identical to get_image_embeddings
        with torch.no_grad():
            for i in tqdm(range(0, len(rendered_images), batch_size), desc="Encoding Rendered Text"):
                batch_imgs = rendered_images[i : i + batch_size]
                
                # The processor accepts a list of PIL images directly
                batch_embeds = self._process_forward_step(batch_imgs)
                all_text_embeddings.append(batch_embeds)

        if not all_text_embeddings:
            return torch.empty(0)

        # 3. Concatenate and return
        return torch.cat(all_text_embeddings, dim=0).to(torch.float16).cpu()

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

# Released checkpoints. Local directories also work in place of the repo ids.
PRETRAIN_MODEL = os.environ.get(
    "PIXEL_LINGUIST_2_PRETRAIN",
    "Pixel-Linguist/Pixel-Linguist-II-Pretrain",
)
MIDTRAIN_MODEL = os.environ.get(
    "PIXEL_LINGUIST_2_MIDTRAIN",
    "Pixel-Linguist/Pixel-Linguist-II-Midtrain",
)
MIDTRAIN_ONLY_MODEL = os.environ.get(
    "PIXEL_LINGUIST_2_MIDTRAIN_ONLY",
    "Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only",
)

_COMMON = dict(
    languages=["eng_Latn"],
    revision="1",
    modalities=["image", "text"],
    n_parameters=676_600_000,
    memory_usage_mb=1290,
    max_tokens=None,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/gowitheflow-1998/LCO-Embedding",
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

pixel_linguist_2_pretrain = ModelMeta(
    loader=partial(Qwen2_5_VL_ViT_Wrapper1, model_path=PRETRAIN_MODEL),
    name="Pixel-Linguist-II-Pretrain",
    release_date="2026-01-03",
    reference="https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Pretrain",
    **_COMMON,
)

pixel_linguist_2_midtrain = ModelMeta(
    loader=partial(Qwen2_5_VL_ViT_Wrapper1, model_path=MIDTRAIN_MODEL),
    name="Pixel-Linguist-II-Midtrain",
    release_date="2026-01-03",
    reference="https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain",
    **_COMMON,
)

pixel_linguist_2_midtrain_only = ModelMeta(
    loader=partial(Qwen2_5_VL_ViT_Wrapper1, model_path=MIDTRAIN_ONLY_MODEL),
    name="Pixel-Linguist-II-Midtrain-Only",
    release_date="2026-01-03",
    reference="https://huggingface.co/Pixel-Linguist/Pixel-Linguist-II-Midtrain-Only",
    **_COMMON,
)

MODELS = {
    "Pixel-Linguist-II-Pretrain": pixel_linguist_2_pretrain,
    "Pixel-Linguist-II-Midtrain": pixel_linguist_2_midtrain,
    "Pixel-Linguist-II-Midtrain-Only": pixel_linguist_2_midtrain_only,
}


def get_model(name: str, **kwargs):
    """Instantiate a released checkpoint ready to hand to `mteb.MTEB(...).run()`.

    MTEB only needs `model.mteb_model_meta` to be set, so this avoids having to
    copy the file into `mteb/models/` and register it in `overview.py`.
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODELS)}")
    meta = MODELS[name]
    model = meta.loader(**kwargs)
    model.mteb_model_meta = meta
    return model
