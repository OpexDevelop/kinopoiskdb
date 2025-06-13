import os
import requests
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
METADATA_FILENAME = "_metadata.json"
RAW_DATA_DIR = "raw_data"
MAX_REQUESTS_PER_RUN = 5 # Измените на 200, когда будете готовы
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        print(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def get_start_page(token):
    try:
        metadata_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            token=token
        )
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        last_page = metadata.get("last_successful_page", 0)
        print(f"Found metadata. Last successful page was {last_page}. Starting from {last_page + 1}.")
        return last_page + 1
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print("Metadata file not found. This must be the first run. Starting from page 1.")
            return 1
        else:
            print(f"An HTTP error occurred while fetching metadata: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while fetching metadata: {e}")
        sys.exit(1)

def main():
    kinopoisk_api_key = get_required_env_var("KINOPOISK_API_KEY")
    hf_token = get_required_env_var("HF_TOKEN")
    
    api = HfApi()
    
    # --- ИЗМЕНЕНИЕ: Автоматическое создание датасета ---
    print(f"Ensuring dataset '{DATASET_ID}' exists...")
    api.create_repo(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=hf_token,
        exist_ok=True # Не вызовет ошибку, если датасет уже существует
    )
    print("Dataset exists or was created successfully.")
    
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
        while pages_processed < MAX_REQUESTS_PER_RUN and current_page <= total_api_pages:
            print(f"Processing page {current_page}...")
            
            # --- ИЗМЕНЕНИЕ: Точный URL-запрос со всеми параметрами ---
            params = [
                ('page', current_page),
                ('limit', 250),
                ('sortField', 'votes.kp'),
                ('sortField', 'votes.imdb'),
                ('sortField', 'rating.imdb'),
                ('sortType', -1),
                ('sortType', -1),
                ('sortType', -1),
                ('selectFields', '')
            ]
            headers = {"X-API-KEY": kinopoisk_api_key}
            
            response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=90)
            response.raise_for_status()
            data = response.json()

            if pages_processed == 0:
                total_api_pages = data.get('pages', total_api_pages)
                print(f"Total pages available in API: {total_api_pages}")
                
                end_page_in_run = min(start_page + MAX_REQUESTS_PER_RUN - 1, total_api_pages)
                date_str = datetime.utcnow().strftime('%Y-%m-%d')
                archive_filename = f"part_{date_str}_{start_page:05d}-{end_page_in_run:05d}.jsonl.gz"
                temp_archive_path = output_dir / archive_filename
                gzip_file = gzip.open(temp_archive_path, 'wt', encoding='utf-8')

            if 'docs' in data and data['docs']:
                for movie in data['docs']:
                    gzip_file.write(json.dumps(movie, ensure_ascii=False) + '\n')
            
            last_successful_page = current_page
            pages_processed += 1
            current_page += 1

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with Kinopoisk API request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
    finally:
        if gzip_file:
            gzip_file.close()

    if pages_processed == 0:
        print("No new pages were processed. Exiting.")
        sys.exit(0)

    print(f"\nProcessing finished. Total pages processed in this run: {pages_processed}.")
    print(f"Last successfully processed page: {last_successful_page}.")
        
    print(f"Uploading {temp_archive_path.name} to dataset...")
    api.upload_file(
        path_or_fileobj=str(temp_archive_path),
        path_in_repo=f"{RAW_DATA_DIR}/{temp_archive_path.name}",
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=hf_token
    )

    new_metadata = {"last_successful_page": last_successful_page}
    local_metadata_path = output_dir / METADATA_FILENAME
    with open(local_metadata_path, 'w') as f:
        json.dump(new_metadata, f)

    print(f"Uploading {METADATA_FILENAME} to dataset...")
    api.upload_file(
        path_or_fileobj=str(local_metadata_path),
        path_in_repo=METADATA_FILENAME,
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Update metadata to page {last_successful_page}"
    )
    
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f"ARTIFACT_PATH={temp_archive_path}\n")

    print("\nDaily collection run successful.")

if __name__ == "__main__":
    main()

