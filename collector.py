import os
import requests
import gzip
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
STATE_FILENAME = "collector_state.json"
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_START_DATE = "1970-01-01"
# Количество одновременных запросов к API
MAX_CONCURRENT_REQUESTS = 15

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RateLimitException(Exception):
    """Кастомное исключение для ошибок лимита запросов."""
    pass

def get_env_var(var_name, default=None):
    """Получает переменную окружения или завершает работу, если она не установлена."""
    value = os.getenv(var_name, default)
    if value is None:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def get_collector_state(api: HfApi):
    """Получает состояние сборщика из репозитория."""
    try:
        logging.info(f"Attempting to download state file: {STATE_FILENAME}")
        state_path = hf_hub_download(
            repo_id=DATASET_ID, filename=STATE_FILENAME, repo_type="dataset",
            force_download=True, resume_download=False
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        last_date = state.get("last_date", DEFAULT_START_DATE)
        last_page_completed = state.get("last_page_completed", 0)
        failed_pages = state.get("failed_pages", [])
        logging.info(f"Found state. Last completed: {last_date}, page {last_page_completed}. Failed pages: {len(failed_pages)}")
        return last_date, last_page_completed, failed_pages
    except (HfHubHTTPError, EntryNotFoundError):
        logging.warning("State file not found. Starting from the beginning.")
        return DEFAULT_START_DATE, 0, []
    except (json.JSONDecodeError, Exception) as e:
        logging.critical(f"Could not read state file: {e}. Starting from scratch.")
        return DEFAULT_START_DATE, 0, []

def update_collector_state(api: HfApi, date_str: str, page_num: int, failed_pages: list):
    """Обновляет файл состояния в репозитории."""
    unique_failed = [dict(t) for t in {tuple(d.items()) for d in failed_pages}]
    state = {"last_date": date_str, "last_page_completed": page_num, "failed_pages": unique_failed}
    local_state_path = Path(STATE_FILENAME)
    with open(local_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    logging.info(f"Uploading new state. Last completed: {date_str}, page {page_num}. Failed pages: {len(unique_failed)}")
    api.upload_file(
        path_or_fileobj=str(local_state_path),
        path_in_repo=STATE_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset"
    )

def fetch_page(date_str_for_req, page, api_key):
    """Запрашивает одну страницу для определенной даты (формат ДД.ММ.ГГГГ)."""
    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': date_str_for_req,
        'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}
    logging.info(f"Requesting page {page} for date {date_str_for_req}...")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 403:
        raise RateLimitException(f"Rate limit exceeded (403 Forbidden) for key ending '...{api_key[-4:]}'.")
    response.raise_for_status()
    return response.json()

def process_failed_tasks(failed_pages, api_key, max_tasks):
    """Параллельно обрабатывает список ранее неудавшихся задач."""
    if not failed_pages:
        return [], [], []

    logging.info(f"--- Starting RETRY phase for {len(failed_pages)} failed pages ---")
    tasks_to_retry = failed_pages[:max_tasks]
    remaining_failed = failed_pages[max_tasks:]
    
    collected_movies = []
    still_failing = list(remaining_failed)
    successfully_retried = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        future_to_task = {}
        for task in tasks_to_retry:
            date_obj = datetime.strptime(task['date'], '%Y-%m-%d').date()
            date_str_for_req = date_obj.strftime('%d.%m.%Y')
            future = executor.submit(fetch_page, date_str_for_req, task['page'], api_key)
            future_to_task[future] = task
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if data and data.get('docs'):
                    collected_movies.extend(data['docs'])
                    successfully_retried.append(task)
                    logging.info(f"[RETRY SUCCESS] Fetched page {task['page']} for date {task['date']}")
                else:
                    logging.warning(f"[RETRY OK] Page {task['page']} for {task['date']} was empty. Removing from retry list.")
                    successfully_retried.append(task) # Считаем успешным, чтобы не повторять снова
            except Exception as exc:
                logging.error(f"[RETRY FAIL] Page {task['page']} for date {task['date']} failed again: {exc}")
                still_failing.append(task)
    
    # Убираем успешно обработанные из общего списка сбойных
    final_failed_list = [t for t in still_failing if t not in successfully_retried]
    return collected_movies, final_failed_list, tasks_to_retry


def main():
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests_total = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided in KINOPOISK_API_KEYS secret.")
        sys.exit(1)

    current_hour = datetime.utcnow().hour
    api_key = api_keys[current_hour % len(api_keys)]
    logging.info(f"Current UTC hour is {current_hour}. Using API key index #{current_hour % len(api_keys)}.")

    api = HfApi(token=hf_token)
    
    # Получаем исходное состояние
    last_date_str, last_page_completed, failed_pages = get_collector_state(api)
    
    requests_processed = 0
    all_collected_movies = []

    try:
        # --- ФАЗА 1: ПОВТОРНАЯ ПОПЫТКА ДЛЯ СБОЙНЫХ СТРАНИЦ ---
        retry_movies, failed_pages, tasks_retried = process_failed_tasks(
            failed_pages, api_key, max_tasks=max_requests_total
        )
        all_collected_movies.extend(retry_movies)
        requests_processed += len(tasks_retried)

        # --- ФАЗА 2: ОСНОВНОЙ СБОР - ПОСЛЕДОВАТЕЛЬНО ПО ДНЯМ ---
        date_to_process = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        
        while requests_processed < max_requests_total:
            if date_to_process > datetime.utcnow().date() + timedelta(days=1):
                logging.info("Target date is in the future. Stopping.")
                break
            
            # Если мы обрабатываем дату из состояния, начинаем с последней страницы.
            # Если это новый день, начинаем с 1-й страницы.
            page = last_page_completed + 1 if date_to_process.strftime('%Y-%m-%d') == last_date_str else 1
            
            date_str_for_req = date_to_process.strftime('%d.%m.%Y')
            logging.info(f"--- Starting to process date {date_str_for_req} from page {page} ---")
            
            while requests_processed < max_requests_total:
                try:
                    data = fetch_page(date_str_for_req, page, api_key)
                    requests_processed += 1
                    
                    if not data or not data.get('docs'):
                        logging.info(f"No more data for date {date_str_for_req}. Moving to the next day.")
                        # День закончен, переходим к следующему
                        date_to_process += timedelta(days=1)
                        last_page_completed = 0 # Сбрасываем счетчик страниц для нового дня
                        break # Выходим из цикла страниц, чтобы перейти к следующей дате
                    
                    all_collected_movies.extend(data['docs'])
                    last_page_completed = page
                    logging.info(f"Success on page {page} for {date_str_for_req}. Total pages: {data.get('pages', page)}.")
                    page += 1

                except Exception as exc:
                    logging.error(f"Request failed for date {date_str_for_req}, page {page}: {exc}")
                    failed_pages.append({"date": date_to_process.strftime('%Y-%m-%d'), "page": page})
                    if isinstance(exc, RateLimitException):
                        raise # Если кончился лимит, нет смысла продолжать
                    
                    # Если другая ошибка, просто переходим к следующему дню, чтобы не застрять
                    date_to_process += timedelta(days=1)
                    last_page_completed = 0
                    break
            
            # Сохраняем последнюю дату, до которой дошли
            last_date_str = date_to_process.strftime('%Y-%m-%d')

    except (RateLimitException, KeyboardInterrupt) as e:
        logging.warning(f"Collection stopped unexpectedly by {type(e).__name__}: {e}. Saving current state.")
    finally:
        # --- ФАЗА 3: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ И СОСТОЯНИЯ ---
        if all_collected_movies:
            logging.info(f"\nCollected {len(all_collected_movies)} movies this run.")
            output_dir = Path("output_raw")
            output_dir.mkdir(exist_ok=True)
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
            local_archive_path = output_dir / archive_filename
            
            all_collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
            
            with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
                for movie in all_collected_movies:
                    f.write(json.dumps(movie, ensure_ascii=False) + '\n')
            logging.info(f"Saved data to {local_archive_path}")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(local_archive_path),
                    path_in_repo=f"{RAW_DATA_DIR}/{archive_filename}",
                    repo_id=DATASET_ID, repo_type="dataset",
                )
                logging.info("Raw chunk upload successful.")
            except Exception as e:
                logging.critical(f"Failed to upload data chunk: {e}", exc_info=True)
        else:
            logging.info("No new movies collected in this run.")
        
        try:
            update_collector_state(api, last_date_str, last_page_completed, failed_pages)
        except Exception as e:
            logging.critical(f"CRITICAL: Failed to update FINAL state: {e}", exc_info=True)
            sys.exit(1)

    logging.info("\nCollector run completed.")

if __name__ == "__main__":
    main()


