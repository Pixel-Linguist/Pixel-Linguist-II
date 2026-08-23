import os
import sys
import urllib.request
from collections import Counter
import multiprocessing
import re

from tqdm import tqdm
from datasets import load_from_disk, load_dataset
import fasttext
import fire
import emoji  # requires: pip install emoji

# Silence the FastText load warning
fasttext.FastText.eprint = lambda x: None

# -----------------------------------------------------------------------------
# Unicode ranges for archaic / rare scripts
# -----------------------------------------------------------------------------
# Blocks that essentially never occur in modern corpora
# Ogham, Runic, Linear B, Cuneiform, Egyptian Hieroglyphs, Gothic, Phoenician, ...
ANCIENT_SCRIPTS_REGEX = re.compile(
    r'['
    r'\u1680-\u169F'      # Ogham
    r'\u16A0-\u16FF'      # Runic
    r'\U00010000-\U0001007F'  # Linear B Syllabary
    r'\U00010330-\U0001034F'  # Gothic
    r'\U00010900-\U0001091F'  # Phoenician
    r'\U00012000-\U000123FF'  # Cuneiform (Sumerian / Akkadian)
    r'\U00013000-\U0001342F'  # Egyptian Hieroglyphs
    r'\U00010800-\U0001083F'  # Cypriot Syllabary
    r']', 
    re.UNICODE
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_worker_model = None

def download_model():
    """Download the FastText language identification model."""
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "lid.176.bin"
    if not os.path.exists(model_path):
        print("[*] Downloading the language identification model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

def clean_text(text):
    """Basic text cleanup."""
    if not text: return ""
    text = "".join([c for c in str(text) if ord(c) >= 32 or c in '\n\r\t'])
    return text.replace("\n", " ").strip()

def has_emoji(text):
    """Whether the text contains emoji."""
    return emoji.emoji_count(text) > 0

def has_ancient_script(text):
    """Whether the text contains archaic scripts."""
    if ANCIENT_SCRIPTS_REGEX.search(text):
        return True
    return False

def init_worker_model(model_path):
    """Load the model in each worker."""
    global _worker_model
    if _worker_model is None:
        _worker_model = fasttext.load_model(model_path)
    return _worker_model

def predict_batch_language(batch, model_path):
    """
    Batched dataset.map function: identify the language.
    """
    model = init_worker_model(model_path)
    langs = []
    
    s1_list = batch.get('sentence1', [])
    s2_list = batch.get('sentence2', [])
    
    # Fall back to the first column if the expected one is missing
    if not s1_list and not s2_list and len(batch) > 0:
        first_col = list(batch.values())[0]
        s1_list = first_col
        s2_list = [""] * len(first_col)
    
    for s1, s2 in zip(s1_list, s2_list):
        t1 = clean_text(s1)
        t2 = clean_text(s2)
        combined_text = t1 + " " + t2
        
        if not combined_text.strip():
            langs.append("unknown")
            continue
            
        try:
            predictions = model.predict(combined_text, k=1)
            lang_code = predictions[0][0].replace("__label__", "")
            langs.append(lang_code)
        except Exception:
            langs.append("unknown")
            
    return {"lang": langs}

def check_content_safety(batch, valid_languages):
    """
    Batched dataset.filter function, checking all of:
    1. the language is in the keep list (ratio > 0.01%)
    2. the text contains no emoji
    3. the text contains no archaic scripts
    """
    langs = batch['lang']
    s1_list = batch.get('sentence1', [])
    s2_list = batch.get('sentence2', [])
    
    # Fallback
    if not s1_list:
        s1_list = list(batch.values())[0]
        s2_list = [""] * len(s1_list)

    keep_mask = []
    
    for lang, s1, s2 in zip(langs, s1_list, s2_list):
        # 1. Language filter (keep the frequent ones)
        if lang not in valid_languages:
            keep_mask.append(False)
            continue
            
        combined_text = str(s1) + str(s2)
        
        # 2. Emoji filter
        if has_emoji(combined_text):
            keep_mask.append(False)
            continue
            
        # 3. Archaic script filter
        if has_ancient_script(combined_text):
            keep_mask.append(False)
            continue
            
        keep_mask.append(True)
        
    return keep_mask

def filter_and_save_dataset(
    data_path: str,
    output_path: str,
    split: str = "train",
    batch_size: int = 10000,
    num_workers: int = None,
    threshold_ratio: float = 0.0001  # 0.01%
):
    """
    Entry point.
    """
    model_path = download_model()

    # 1. Load the data
    print(f"[*] Loading dataset: {data_path}")
    try:
        if os.path.exists(os.path.join(data_path, "dataset_info.json")):
            dataset = load_from_disk(data_path)
        else:
            dataset = load_dataset(data_path, split=split)
    except Exception as e:
        print(f"[Error] Failed to load: {e}")
        return

    if hasattr(dataset, "keys") and isinstance(dataset, dict):
        if split in dataset:
            dataset = dataset[split]
        else:
            key = list(dataset.keys())[0]
            dataset = dataset[key]

    total_samples = len(dataset)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"[*] Dataset size: {total_samples}")
    
    # 2. Identify languages (map)
    print(f"[*] Step 1/4: identifying languages in parallel (workers={num_workers})...")
    dataset_with_lang = dataset.map(
        predict_batch_language,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        fn_kwargs={"model_path": model_path},
        desc="Language ID"
    )

    # 3. Tally languages and decide which to keep
    print("[*] Step 2/4: tallying the language distribution and computing the threshold...")
    lang_counts = Counter(dataset_with_lang['lang'])
    
    valid_languages = set()
    total_valid_count = sum(lang_counts.values())
    
    print("\n" + "="*60)
    print(f"{'Lang':<8} | {'Count':<10} | {'Share':<8} | {'Status'}")
    print("-" * 60)
    
    for lang, count in lang_counts.most_common():
        ratio = count / total_valid_count
        is_kept = ratio >= threshold_ratio
        
        if is_kept:
            valid_languages.add(lang)
            
        if ratio > 0.00001: 
            status = "keep" if is_kept else "drop (share too low)"
            print(f"{lang:<8} | {count:<10} | {ratio*100:>6.3f}% | {status}")
    
    print("-" * 60)
    print(f"[*] Threshold: {threshold_ratio*100}%")
    print(f"[*] Languages kept: {len(valid_languages)}")
    print("="*60)

    # 4. Combined filter
    # Language share, emoji and archaic scripts are checked in one filter pass
    print("[*] Step 3/4: applying the combined filter (rare languages + emoji + archaic)...")
    
    filtered_dataset = dataset_with_lang.filter(
        check_content_safety,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        fn_kwargs={"valid_languages": valid_languages},
        desc="Filtering Safe Content"
    )

    # 5. Save
    print("[*] Step 4/4: saving...")
    print(f"   - before: {len(dataset_with_lang)}")
    print(f"   - after:  {len(filtered_dataset)}")
    print(f"   - removed: {100 * (1 - len(filtered_dataset)/len(dataset_with_lang)):.2f}%")
    
    # Drop the temporary lang column
    if "lang" in filtered_dataset.column_names:
        filtered_dataset = filtered_dataset.remove_columns(["lang"])

    filtered_dataset.save_to_disk(output_path)
    print(f"[*] Done. Saved to: {output_path}")

if __name__ == "__main__":
    fire.Fire(filter_and_save_dataset)