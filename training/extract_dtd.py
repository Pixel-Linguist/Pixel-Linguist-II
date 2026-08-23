import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_parquet_to_images(parquet_path, output_dir):
    """
    Read a Parquet file and write its image column out as image files.
    """
    if not os.path.exists(parquet_path):
        print(f"Skipping {parquet_path}, file not found.")
        return

    print(f"Loading {parquet_path}...")
    # Read the Parquet with pandas
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return

    if 'image' not in df.columns:
        print(f"Error: 'image' column not found in {parquet_path}")
        print(f"Available columns: {df.columns}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting images to {output_dir}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_data = row['image']
        
        # In HuggingFace Parquet, images are usually a dict {'bytes': b'...', 'path': ...}
        # or plain bytes
        img_bytes = None
        
        if isinstance(image_data, dict) and 'bytes' in image_data:
            img_bytes = image_data['bytes']
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            # Occasionally already decoded, but reading Parquet normally gives bytes or a dict
            continue
            
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                # Convert to RGB to avoid format issues
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Write the file, named by index
                save_name = f"{os.path.basename(parquet_path).split('.')[0]}_{idx}.jpg"
                img.save(os.path.join(output_dir, save_name), quality=95)
            except Exception as e:
                print(f"Failed to save image {idx}: {e}")

# --- Paths ---
# Parquet files live under data/dtd/data/
base_parquet_dir = "data/dtd/data" 
output_image_dir = "data/dtd/extracted_images" # images are extracted here

# Process the train and test files
parquet_files = [
    os.path.join(base_parquet_dir, "train-00000-of-00001.parquet"),
    os.path.join(base_parquet_dir, "test-00000-of-00001.parquet")
]

if __name__ == "__main__":
    for p_file in parquet_files:
        extract_parquet_to_images(p_file, output_image_dir)
        
    print(f"\nDone! Images are saved in: {output_image_dir}")
    # Report the count
    if os.path.exists(output_image_dir):
        num_files = len(os.listdir(output_image_dir))
        print(f"Total extracted images: {num_files}")