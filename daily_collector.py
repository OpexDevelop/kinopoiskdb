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

# --- КОНФИГУРАЦИЯ ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
METADATA_FILENAME = "_metadata_collector_state.json"
RAW_DATA_DIR = "raw_data"
MAX_REQUESTS_PER_RUN = 30
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 240
MAX_RETRIES = 5 # Можно уменьшить, если ошибка лимита частая
DEFAULT_START_DATE_ISO = "1970-01-01T00:00:00.000Z"

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
log_filename = f"log_collector_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Кастомное исключение для обработки лимита API ---
class RateLimitException(Exception):
    pass

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def get_collector_state(token):
    try:
        logging.info(f"Attempting to download collector state file: {METADATA_FILENAME}")
        metadata_path = hf_hub_download(
            repo_id=DATASET_ID, filename=METADATA_FILENAME,
            repo_type="dataset", token=token, cache_dir="hf_cache"
        )
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        last_date = metadata.get("last_processed_update_iso", DEFAULT_START_DATE_ISO)
        logging.info(f"Found state. Last processed update date was {last_date}.")
        return last_date
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logging.warning("Collector state file not found. Starting from the beginning.")
            return DEFAULT_START_DATE_ISO
        else:
            logging.error(f"An HTTP error occurred while fetching state: {e}")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while fetching state: {e}")
        sys.exit(1)

def fetch_page_with_retries(page, api_key, start_date_iso):
    try:
        start_dt_object = datetime.fromisoformat(start_date_iso.replace('Z', '+00:00'))
        start_date_dmy = start_dt_object.strftime('%d.%m.%Y')
    except ValueError:
        logging.error(f"Could not parse date: {start_date_iso}. Using default start date.")
        start_date_dmy = "01.01.1970"
    end_date_dmy = "31.12.2099"
    
    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': f"{start_date_dmy}-{end_date_dmy}", 'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}

    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Requesting page {page} with updatedAt={params['updatedAt']} (attempt {attempt + 1}/{MAX_RETRIES})...")
            response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            
            # ИЗМЕНЕНИЕ: Проверяем на ошибку лимита ПЕРЕД вызовом raise_for_status
            if response.status_code == 403:
                logging.warning(f"Attempt {attempt + 1}: Received 403 Forbidden. This is likely a rate limit issue.")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt) # Экспоненциальная задержка
                    continue
                else:
                    # Если все попытки не увенчались успехом, вызываем наше исключение
                    raise RateLimitException("Rate limit exceeded after all retries.")

            response.raise_for_status()
            
            payload_size = len(response.content)
            logging.info(f"Page {page}: Received status code {response.status_code}. Payload size: {payload_size / 1024:.2f} KB")

            if not payload_size:
                logging.warning(f"Page {page}: Received an empty response body. Assuming end of data.")
                return None
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Page {page}, attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                logging.info(f"Will retry in {2 ** attempt} seconds...")
            else:
                logging.error(f"All {MAX_RETRIES} retries for page {page} failed.")
                raise # Передаем ошибку выше, если это не 403
        except json.JSONDecodeError as e:
            logging.error(f"Page {page}: Failed to decode JSON from response. Error: {e}", exc_info=True)
            raise

    return None

def main():
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as f:
            f.write(f"LOG_FILE_PATH={log_filename}\n")

    kinopoisk_api_key = get_required_env_var("KINOPOISK_API_KEY")
    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi()

    logging.info(f"Ensuring dataset repo '{DATASET_ID}' exists...")
    api.create_repo(repo_id=DATASET_ID, repo_type="dataset", token=hf_token, exist_ok=True)

    start_date_iso = get_collector_state(hf_token)
    
    current_page = 1
    requests_processed = 0
    movies_collected_in_run = 0
    latest_timestamp_this_run = start_date_iso

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
    archive_filename = f"part_{date_str}_updates_since_{start_date_iso.replace(':', '-')}.jsonl.gz"
    temp_archive_path = output_dir / archive_filename
    
    try:
        with gzip.open(temp_archive_path, 'wt', encoding='utf-8') as gzip_file:
            while requests_processed < MAX_REQUESTS_PER_RUN:
                data = fetch_page_with_retries(current_page, kinopoisk_api_key, start_date_iso)
                requests_processed += 1
                if not data or not data.get('docs'):
                    logging.info("No more movies found. Stopping collection for this run.")
                    break
                docs = data['docs']
                movie_count = len(docs)
                movies_collected_in_run += movie_count
                logging.info(f"Page {current_page}: Found {movie_count} movies. Writing to archive.")
                for movie in docs:
                    gzip_file.write(json.dumps(movie, ensure_ascii=False) + '\n')
                last_movie_in_batch_date = docs[-1].get("updatedAt")
                if last_movie_in_batch_date:
                    latest_timestamp_this_run = last_movie_in_batch_date
                if movie_count < 250:
                    logging.info("Received a partial page, which means it's the last page of results. Stopping.")
                    break
                current_page += 1

    # ИЗМЕНЕНИЕ: Отлавливаем ошибку лимита и выходим успешно
    except RateLimitException as e:
        logging.warning(f"Stopping run due to API rate limit: {e}")
        # Если какие-то фильмы успели собраться до ошибки лимита, сохраняем их
        if movies_collected_in_run > 0:
            pass # Продолжаем выполнение и сохраняем то, что есть
        else: # Если ничего не собрали, выходим с успехом
            if temp_archive_path.exists():
                os.remove(temp_archive_path)
            logging.info("Exiting gracefully. No new data was collected before hitting the rate limit.")
            sys.exit(0)
    
    except Exception as e:
        logging.critical(f"A critical, unrecoverable error occurred: {e}", exc_info=True)
        if temp_archive_path.exists():
            os.remove(temp_archive_path)
        sys.exit(1)

    # ... остальная часть кода для сохранения файлов ...
    if movies_collected_in_run == 0:
        logging.info("No new data was collected. Deleting empty archive and exiting.")
        if temp_archive_path.exists():
            os.remove(temp_archive_path)
        sys.exit(0)

    logging.info(f"\nProcessing finished. Requests made: {requests_processed}. Movies collected: {movies_collected_in_run}.")
    logging.info(f"New state timestamp will be: {latest_timestamp_this_run}")

    logging.info(f"Uploading data chunk {temp_archive_path.name} to dataset...")
    api.upload_file(
        path_or_fileobj=str(temp_archive_path),
        path_in_repo=f"{RAW_DATA_DIR}/{temp_archive_path.name}",
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token
    )

    new_metadata = {"last_processed_update_iso": latest_timestamp_this_run}
    local_metadata_path = output_dir / METADATA_FILENAME
    with open(local_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f)

    logging.info(f"Uploading new state file {METADATA_FILENAME}...")
    api.upload_file(
        path_or_fileobj=str(local_metadata_path),
        path_in_repo=METADATA_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token,
        commit_message=f"Collector state update: {latest_timestamp_this_run}"
    )

    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as f:
            f.write(f"ARTIFACT_CHUNK_PATH={temp_archive_path}\n")

    logging.info("\nCollection run completed successfully.")

if __name__ == "__main__":
    main()


