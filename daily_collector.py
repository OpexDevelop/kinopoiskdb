import os
import requests
import gzip
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
METADATA_FILENAME = "_metadata.json"
RAW_DATA_DIR = "raw_data"
MAX_REQUESTS_PER_RUN = 5 
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 240 # ИЗМЕНЕНО: 4 минуты
MAX_RETRIES = 10 

# --- Настройка логирования ---
log_filename = f"log_daily_collector_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
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

def get_start_page(token):
    try:
        logging.info(f"Attempting to download metadata file: {METADATA_FILENAME}")
        metadata_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            token=token
        )
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        last_page = metadata.get("last_successful_page", 0)
        logging.info(f"Found metadata. Last successful page was {last_page}. Starting from {last_page + 1}.")
        return last_page + 1
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logging.warning("Metadata file not found. This must be the first run. Starting from page 1.")
            return 1
        else:
            logging.error(f"An HTTP error occurred while fetching metadata: {e}")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while fetching metadata: {e}")
        sys.exit(1)

def fetch_page_with_retries(page, api_key):
    params = [
        ('page', page), ('limit', 250),
        ('sortField', 'votes.kp'), ('sortField', 'votes.imdb'), ('sortField', 'rating.imdb'),
        ('sortType', -1), ('sortType', -1), ('sortType', -1),
        ('selectFields', '')
    ]
    headers = {"X-API-KEY": api_key}
    
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Requesting page {page}, attempt {attempt + 1}/{MAX_RETRIES}...")
            response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            logging.info(f"Page {page}: Received status code {response.status_code}. Content-Length: {response.headers.get('Content-Length')}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Page {page}, attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt
                logging.info(f"Will retry in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"All {MAX_RETRIES} retries for page {page} failed.")
                raise
    return None

def main():
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f"LOG_FILE_PATH={log_filename}\n")

    kinopoisk_api_key = get_required_env_var("KINOPOISK_API_KEY")
    hf_token = get_required_env_var("HF_TOKEN")
    
    api = HfApi()
    
    logging.info(f"Ensuring dataset '{DATASET_ID}' exists...")
    api.create_repo(repo_id=DATASET_ID, repo_type="dataset", token=hf_token, exist_ok=True)
    logging.info("Dataset check complete.")
    
    start_page = get_start_page(hf_token)
    current_page = start_page
    last_successful_page = start_page - 1
    pages_processed = 0
    total_api_pages = float('inf')
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    temp_archive_path = None
    gzip_file = None

    try:
        while pages_processed < MAX_REQUESTS_PER_RUN:
            # --- ИЗМЕНЕНИЕ: Логика бесконечного цикла ---
            if 'total_api_pages' in locals() and current_page > total_api_pages and total_api_pages != float('inf'):
                logging.info(f"Reached the end of available pages ({total_api_pages}). Looping back to page 1 for updates.")
                current_page = 1
            
            data = fetch_page_with_retries(current_page, kinopoisk_api_key)
            if data is None:
                logging.critical(f"Could not fetch page {current_page} after all retries. Aborting run.")
                sys.exit(1)

            if pages_processed == 0:
                total_api_pages = data.get('pages', total_api_pages)
                logging.info(f"Total pages available in API: {total_api_pages}")
                
                end_page_in_run = start_page + MAX_REQUESTS_PER_RUN - 1
                date_str = datetime.utcnow().strftime('%Y-%m-%d')
                archive_filename = f"part_{date_str}_{start_page:05d}-{end_page_in_run:05d}.jsonl.gz"
                temp_archive_path = output_dir / archive_filename
                gzip_file = gzip.open(temp_archive_path, 'wt', encoding='utf-8')

            if 'docs' in data and data['docs']:
                movie_count = len(data['docs'])
                logging.info(f"Page {current_page}: Found {movie_count} movies. Writing to archive.")
                for movie in data['docs']:
                    gzip_file.write(json.dumps(movie, ensure_ascii=False) + '\n')
            else:
                logging.warning(f"Page {current_page}: 'docs' key not found or is empty in the response.")
            
            last_successful_page = current_page
            pages_processed += 1
            current_page += 1

    except Exception as e:
        logging.critical(f"A critical, unrecoverable error occurred during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if gzip_file:
            gzip_file.close()

    if pages_processed == 0:
        logging.info("No new pages were processed. Exiting successfully.")
        sys.exit(0)

    logging.info(f"\nProcessing finished. Total pages processed in this run: {pages_processed}.")
    logging.info(f"Last successfully processed page: {last_successful_page}.")
        
    logging.info(f"Uploading {temp_archive_path.name} to dataset...")
    api.upload_file(
        path_or_fileobj=str(temp_archive_path),
        path_in_repo=f"{RAW_DATA_DIR}/{temp_archive_path.name}",
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token
    )

    new_metadata = {"last_successful_page": last_successful_page}
    local_metadata_path = output_dir / METADATA_FILENAME
    with open(local_metadata_path, 'w') as f:
        json.dump(new_metadata, f)

    logging.info(f"Uploading {METADATA_FILENAME} to dataset...")
    api.upload_file(
        path_or_fileobj=str(local_metadata_path),
        path_in_repo=METADATA_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token,
        commit_message=f"Update metadata to page {last_successful_page}"
    )
    
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f"ARTIFACT_CHUNK_PATH={temp_archive_path}\n")

    logging.info("\nDaily collection run completed successfully.")

if __name__ == "__main__":
    main()
