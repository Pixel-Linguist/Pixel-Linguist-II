import os
import multiprocessing
from datasets import load_dataset

def process_dataset_parallel():
    # 1. Paths
    input_path = 'data/wikispan-all'
    output_path = 'data/wikispan-clean'
    
    # Use every available CPU core
    # Set num_proc to a fixed number to leave some cores free
    num_proc = multiprocessing.cpu_count()
    print(f"Detected {num_proc} CPU cores; enabling multiprocessing.")

    if not os.path.exists(input_path):
        print(f"Error: path not found: {input_path}")
        return

    try:
        # 2. Load the dataset
        print(f"Loading dataset: {input_path} ...")
        dataset = load_dataset(input_path)
        
        # 3. Select columns (metadata-only, so this is instant)
        target_columns = ['sentence1', 'sentence2']
        processed_dataset = dataset.select_columns(target_columns)
        
        print("\n=== Dataset ready, writing in parallel ===")
        print(f"Target path: {output_path}")
        
        # 4. Save in parallel (this is the slow part)
        # save_to_disk shards the data and writes the shards concurrently
        processed_dataset.save_to_disk(output_path, num_proc=num_proc)
        
        print("\nDone. Dataset saved.")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    # On Windows/macOS multiprocessing must be guarded by __main__
    process_dataset_parallel()