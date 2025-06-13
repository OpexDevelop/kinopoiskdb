import os
import gzip
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
CONSOLIDATED_DATA_DIR = "consolidated"
CONSOLIDATED_FILENAME = "consolidated_data.jsonl.gz"

# --- Настройка логирования ---
log_filename = f"log_consolidator_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def parse_iso_date(date_string):
    if not date_string:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)

def main():
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f"LOG_FILE_PATH={log_filename}\n")

    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi()

    logging.info("Fetching list of raw data files...")
    try:
        raw_files = list_repo_files(
            repo_id=DATASET_ID, repo_type="dataset",
            token=hf_token, paths=[RAW_DATA_DIR]
        )
    except Exception as e:
        logging.critical(f"Could not list repo files. Does '{RAW_DATA_DIR}' directory exist? Error: {e}")
        sys.exit(1)

    if not raw_files:
        logging.warning("No raw data files found to consolidate. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(raw_files)} raw files. Starting deduplication...")
    movies_db = {}
    
    for file_path_in_repo in raw_files:
        logging.info(f"  Processing {file_path_in_repo}...")
        try:
            local_file_path = hf_hub_download(
                repo_id=DATASET_ID, filename=file_path_in_repo,
                repo_type="dataset", token=hf_token
            )
            with gzip.open(local_file_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
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
                        logging.warning(f"    JSON decode error in {file_path_in_repo} at line {i+1}. Skipping line.")
                        continue
        except Exception as e:
            logging.error(f"    Could not process file {file_path_in_repo}. Skipping. Error: {e}")

    logging.info(f"\nDeduplication complete. Unique movies found: {len(movies_db)}")

    sorted_movies = sorted(movies_db.values(), key=lambda x: x['id'])
    
    output_dir = Path("output_consolidated")
    output_dir.mkdir(exist_ok=True)
    
    final_archive_path = output_dir / CONSOLIDATED_FILENAME
    
    logging.info(f"Writing all unique movies to a single archive: {final_archive_path}...")
    with gzip.open(final_archive_path, 'wt', encoding='utf-8') as f:
        for movie in sorted_movies:
            movie.pop('updatedAt_dt', None)
            f.write(json.dumps(movie, ensure_ascii=False) + '\n')
            
    final_size_gb = final_archive_path.stat().st_size / 1024**3
    logging.info(f"Finished writing. Final archive size: {final_size_gb:.3f} GB")

    logging.info(f"Uploading {final_archive_path.name} to dataset...")
    api.upload_file(
        path_or_fileobj=str(final_archive_path),
        path_in_repo=f"{CONSOLIDATED_DATA_DIR}/{final_archive_path.name}",
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token,
        commit_message="Consolidate and deduplicate all raw data"
    )

    logging.info("\nConsolidation run completed successfully.")

if __name__ == "__main__":
    main()
