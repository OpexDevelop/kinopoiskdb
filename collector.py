import os
import requests
import gzip
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# ИСПРАВЛЕНО: Правильный импорт модуля с классами ошибок
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
STATE_FILENAME = "collector_state.json"
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 240
MAX_RETRIES = 5
DEFAULT_START_DATE_ISO = "1970-01-01T00:00:00.000Z"

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RateLimitException(Exception):
    pass

def get_env_var(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def get_collector_state(api: HfApi):
    """Получает состояние сборщика (последняя дата) из репозитория."""
    try:
        logging.info(f"Attempting to download state file: {STATE_FILENAME}")
        state_path = api.hf_hub_download(
            repo_id=DATASET_ID,
            filename=STATE_FILENAME,
            repo_type="dataset",
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        last_date = state.get("last_processed_update_iso", DEFAULT_START_DATE_ISO)
        logging.info(f"Found state. Last processed date: {last_date}")
        return last_date
    # ИСПРАВЛЕНО: Используем правильный класс ошибки для 404
    except EntryNotFoundError:
        logging.warning("State file not found. This is the first run. Starting from the beginning.")
        return DEFAULT_START_DATE_ISO
    except Exception as e:
        logging.critical(f"An unexpected error occurred while fetching state: {e}")
        sys.exit(1)

def update_collector_state(api: HfApi, new_date_iso: str):
    """Загружает новый файл состояния в репозиторий."""
    state = {"last_processed_update_iso": new_date_iso}
    local_state_path = Path(STATE_FILENAME)
    with open(local_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f)
    
    logging.info(f"Uploading new state file. New date: {new_date_iso}")
    api.upload_file(
        path_or_fileobj=str(local_state_path),
        path_in_repo=STATE_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset"
    )

def fetch_page(page, api_key, start_date_iso):
    """Запрашивает одну страницу из API."""
    try:
        start_dt_object = datetime.fromisoformat(start_date_iso.replace('Z', '+00:00'))
        start_date_dmy = start_dt_object.strftime('%d.%m.%Y')
    except ValueError:
        start_date_dmy = "01.01.1970"

    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': f"{start_date_dmy}-31.12.2099", 'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}
    
    logging.info(f"Requesting page {page} with updatedAt > {start_date_dmy}...")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

    if response.status_code == 403:
        raise RateLimitException("Rate limit exceeded (403 Forbidden).")
    
    response.raise_for_status()
    return response.json()

def main():
    """Основная функция сбора данных."""
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided in KINOPOISK_API_KEYS.")
        sys.exit(1)

    num_tokens = len(api_keys)
    current_hour = datetime.utcnow().hour
    
    # --- Логика умного распределения ---
    # Определяем интервал между запусками, чтобы равномерно использовать ключи в течение суток
    interval = 24 // num_tokens
    if interval == 0: interval = 1 # На случай если ключей > 24

    if current_hour % interval != 0:
        logging.info(f"Current hour {current_hour} is not a scheduled slot for {num_tokens} tokens with interval {interval}. Skipping run.")
        sys.exit(0)
    
    key_index = (current_hour // interval) % num_tokens
    api_key = api_keys[key_index]
    logging.info(f"This is a scheduled run. Using API key #{key_index}.")
    
    api = HfApi(token=hf_token)
    
    start_date_iso = get_collector_state(api)
    
    current_page = 1
    requests_processed = 0
    collected_movies = []
    
    while requests_processed < max_requests:
        try:
            data = fetch_page(current_page, api_key, start_date_iso)
            requests_processed += 1

            if not data or not data.get('docs'):
                logging.info("No more movies found for the given period. Stopping collection.")
                break
            
            docs = data['docs']
            collected_movies.extend(docs)
            logging.info(f"Page {current_page}: Found {len(docs)} movies.")

            if len(docs) < 250:
                logging.info("Received a partial page, meaning it's the last page. Stopping.")
                break
            
            current_page += 1
            time.sleep(1)

        except RateLimitException as e:
            logging.warning(f"Stopping run due to API rate limit: {e}")
            break
        except Exception as e:
            logging.error(f"A critical error occurred during fetch: {e}", exc_info=True)
            sys.exit(1)

    if not collected_movies:
        logging.info("No new movies were collected in this run. Exiting without changing state.")
        sys.exit(0)

    logging.info(f"\nCollection finished. Total movies collected: {len(collected_movies)}.")
    
    output_dir = Path("output_raw")
    output_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
    local_archive_path = output_dir / archive_filename
    
    with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
        for movie in collected_movies:
            f.write(json.dumps(movie, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved data to local file: {local_archive_path}")

    try:
        logging.info(f"Uploading {archive_filename} to dataset...")
        api.upload_file(
            path_or_fileobj=str(local_archive_path),
            path_in_repo=f"{RAW_DATA_DIR}/{archive_filename}",
            repo_id=DATASET_ID, repo_type="dataset",
        )
        logging.info("Raw chunk upload successful.")

        # --- Обновляем состояние только после успешной загрузки файла ---
        latest_update_iso = collected_movies[-1].get("updatedAt", start_date_iso)
        update_collector_state(api, latest_update_iso)

    except Exception as e:
        logging.critical(f"Failed to upload data or state file to Hugging Face: {e}", exc_info=True)
        sys.exit(1)

    logging.info("\nCollector run completed successfully.")

if __name__ == "__main__":
    main()


