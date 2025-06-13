import os
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
CONSOLIDATED_DATA_DIR = "consolidated"
MAX_FILE_SIZE_GB = 4.5
MAX_FILE_BYTES = int(MAX_FILE_SIZE_GB * 1024**3)

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        print(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def parse_iso_date(date_string):
    if not date_string:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        # Handles formats like '2024-12-29T01:00:36.406Z'
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)

def main():
    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi()

    print("Fetching list of raw data files...")
    try:
        raw_files = list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=hf_token,
            paths=[RAW_DATA_DIR]
        )
    except Exception as e:
        print(f"Could not list repo files. Does '{RAW_DATA_DIR}' directory exist? Error: {e}")
        sys.exit(1)

    if not raw_files:
        print("No raw data files found to consolidate. Exiting.")
        sys.exit(0)

    print(f"Found {len(raw_files)} raw files. Starting deduplication...")
    movies_db = {}
    
    for file_path_in_repo in raw_files:
        print(f"  Processing {file_path_in_repo}...")
        try:
            local_file_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=file_path_in_repo,
                repo_type="dataset",
                token=hf_token
            )
            with gzip.open(local_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        movie = json.loads(line)
                        movie_id = movie.get("id")
                        updated_at_str = movie.get("updatedAt")
                        
                        if not movie_id or not updated_at_str:
                            continue
                            
                        updated_at_dt = parse_iso_date(updated_at_str)

                        if movie_id not in movies_db or updated_at_dt > movies_db[movie_id]['updatedAt_dt']:
                            movie['updatedAt_dt'] = updated_at_dt
                            movies_db[movie_id] = movie
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"    Could not process file {file_path_in_repo}. Skipping. Error: {e}")

    print(f"\nDeduplication complete. Unique movies found: {len(movies_db)}")

    sorted_movies = sorted(movies_db.values(), key=lambda x: x['id'])
    
    output_dir = Path("output_consolidated")
    output_dir.mkdir(exist_ok=True)
    
    chunk_num = 1
    current_size = 0
    current_file_path = None
    gzip_file = None
    
    print("Writing consolidated files...")
    for movie in sorted_movies:
        if gzip_file is None:
            current_file_path = output_dir / f"consolidated_data_{chunk_num:03d}.jsonl.gz"
            gzip_file = gzip.open(current_file_path, 'wt', encoding='utf-8')
            print(f"  Creating chunk: {current_file_path.name}")

        movie.pop('updatedAt_dt', None)
        line_to_write = json.dumps(movie, ensure_ascii=False) + '\n'
        line_bytes = line_to_write.encode('utf-8')
        
        gzip_file.write(line_to_write)
        current_size += len(line_bytes)
        
        if current_size >= MAX_FILE_BYTES:
            gzip_file.close()
            print(f"  Chunk {chunk_num} complete. Size: {current_size / 1024**3:.2f} GB")
            
            print(f"  Uploading {current_file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(current_file_path),
                path_in_repo=f"{CONSOLIDATED_DATA_DIR}/{current_file_path.name}",
                repo_id=DATASET_ID,
                repo_type="dataset",
                token=hf_token
            )
            
            chunk_num += 1
            current_size = 0
            gzip_file = None

    if gzip_file:
        gzip_file.close()
        print(f"  Final chunk {chunk_num} complete. Size: {current_size / 1024**3:.2f} GB")
        print(f"  Uploading {current_file_path.name}...")
        api.upload_file(
            path_or_fileobj=str(current_file_path),
            path_in_repo=f"{CONSOLIDATED_DATA_DIR}/{current_file_path.name}",
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=hf_token
        )

    print("\nConsolidation run successful.")

if __name__ == "__main__":
    main()

