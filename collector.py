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
# Измените здесь количество параллельных запросов
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
            force_download=True, resume_download=False # Для избежания кэша в Actions
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        last_date = state.get("last_date", DEFAULT_START_DATE)
        last_page = state.get("last_page", 0)
        failed_pages = state.get("failed_pages", [])
        logging.info(f"Found state. Position: {last_date}, page {last_page}. Failed to retry: {len(failed_pages)}")
        return last_date, last_page, failed_pages
    except (HfHubHTTPError, EntryNotFoundError):
        logging.warning("State file not found. Starting from the beginning.")
        return DEFAULT_START_DATE, 0, []
    except (json.JSONDecodeError, Exception) as e:
        logging.critical(f"Could not read state file: {e}. Starting from scratch.")
        return DEFAULT_START_DATE, 0, []

def update_collector_state(api: HfApi, date_str: str, page_num: int, failed_pages: list):
    """Обновляет файл состояния в репозитории."""
    unique_failed = [dict(t) for t in {tuple(d.items()) for d in failed_pages}]
    state = {"last_date": date_str, "last_page": page_num, "failed_pages": unique_failed}
    local_state_path = Path(STATE_FILENAME)
    with open(local_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    logging.info(f"Uploading new state. Position: {date_str}, page {page_num}. Failed pages: {len(unique_failed)}")
    api.upload_file(
        path_or_fileobj=str(local_state_path),
        path_in_repo=STATE_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset"
    )

def fetch_page_for_date(page, date_str, api_key):
    """Запрашивает одну страницу для определенной даты."""
    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': date_str,
        'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}
    logging.info(f"Requesting page {page} for date {date_str}...")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 403:
        raise RateLimitException(f"Rate limit exceeded (403 Forbidden) for key ending '...{api_key[-4:]}'.")
    response.raise_for_status()
    return date_str, page, response.json()

def main():
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided in KINOPOISK_API_KEYS secret.")
        sys.exit(1)

    # --- ИСПРАВЛЕННАЯ ЛОГИКА ВЫБОРА КЛЮЧА ---
    current_hour = datetime.utcnow().hour # Час по UTC (от 0 до 23)
    try:
        api_key = api_keys[current_hour]
        logging.info(f"Current UTC hour is {current_hour}. Using API key with index #{current_hour}.")
    except IndexError:
        logging.info(f"Current UTC hour is {current_hour}, but no API key found for this index. Skipping run.")
        sys.exit(0) # Завершаем работу, если для этого часа нет ключа
    # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

    api = HfApi(token=hf_token)
    current_date_str, last_page_for_date, failed_pages = get_collector_state(api)
    current_date_obj = datetime.strptime(current_date_str, '%Y-%m-%d').date()
    current_page = last_page_for_date + 1

    requests_processed = 0
    collected_movies = []
    # Переменные для отслеживания реального прогресса для сохранения
    final_date_to_save = current_date_obj
    final_page_to_save = last_page_for_date

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # --- ФАЗА 1: ПОВТОРНАЯ ПОПЫТКА ДЛЯ СБОЙНЫХ СТРАНИЦ ---
            if failed_pages and requests_processed < max_requests:
                logging.info(f"--- Starting RETRY phase for {len(failed_pages)} failed pages ---")
                retries_to_attempt = failed_pages[:(max_requests - requests_processed)]
                retry_futures = {
                    executor.submit(fetch_page_for_date, item['page'], item['date'].replace('-', '.'), api_key): item
                    for item in retries_to_attempt
                }
                still_failing = []
                remaining_failures = failed_pages[len(retries_to_attempt):]
                for future in as_completed(retry_futures):
                    item = retry_futures[future]
                    requests_processed += 1
                    try:
                        _date_str, _page, data = future.result()
                        if data and data.get('docs'):
                            collected_movies.extend(data['docs'])
                            logging.info(f"[RETRY SUCCESS] Fetched page {item['page']} for date {item['date']}")
                        else:
                            logging.warning(f"[RETRY OK] Page {item['page']} for {item['date']} was empty. Removing from retry list.")
                    except Exception as exc:
                        logging.error(f"[RETRY FAIL] Page {item['page']} for date {item['date']} failed again: {exc}")
                        still_failing.append(item)
                failed_pages = still_failing + remaining_failures
                logging.info(f"--- RETRY phase finished. {len(failed_pages)} pages still failing. ---")

            # --- ФАЗА 2: ОСНОВНОЙ СБОР ---
            logging.info(f"--- Starting MAIN collection from {current_date_obj.strftime('%Y-%m-%d')} page {current_page} ---")
            while requests_processed < max_requests:
                if current_date_obj > datetime.utcnow().date() + timedelta(days=1):
                    logging.info("Target date is in the future. Stopping.")
                    break
                date_str_for_req = current_date_obj.strftime('%d.%m.%Y')
                
                # 1. Пробный запрос
                try:
                    _date, _page, initial_data = fetch_page_for_date(current_page, date_str_for_req, api_key)
                    requests_processed += 1
                except Exception as exc:
                    logging.error(f"Probe request failed for date {date_str_for_req}, page {current_page}: {exc}")
                    if not isinstance(exc, RateLimitException):
                        failed_pages.append({"date": current_date_obj.strftime('%Y-%m-%d'), "page": current_page})
                    raise # Прерываем выполнение, чтобы сохранить состояние
                
                # 2. Обработка пробного запроса
                if not initial_data or not initial_data.get('docs'):
                    logging.info(f"No more data for date {date_str_for_req} at page {current_page}. Moving to next day.")
                    final_date_to_save = current_date_obj + timedelta(days=1)
                    final_page_to_save = 0
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                    continue
                
                collected_movies.extend(initial_data['docs'])
                total_pages_for_day = initial_data.get('pages', current_page)
                logging.info(f"Probe successful. Date {date_str_for_req} has {total_pages_for_day} pages. Current at page {current_page}.")
                final_date_to_save = current_date_obj
                final_page_to_save = current_page

                # 3. Формирование пакета для параллельной загрузки
                pages_to_fetch_start = current_page + 1
                if pages_to_fetch_start > total_pages_for_day:
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                    continue

                batch_limit = max_requests - requests_processed
                pages_to_fetch_end = min(total_pages_for_day + 1, pages_to_fetch_start + batch_limit)
                pages_to_fetch = range(pages_to_fetch_start, pages_to_fetch_end)
                if not pages_to_fetch: break
                
                # 4. Параллельная загрузка пакета
                futures = { executor.submit(fetch_page_for_date, page, date_str_for_req, api_key): page for page in pages_to_fetch }
                max_successful_page_in_batch = current_page
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        _date, _page, data = future.result()
                        requests_processed += 1
                        if data and data.get('docs'):
                            collected_movies.extend(data['docs'])
                            max_successful_page_in_batch = max(max_successful_page_in_batch, _page)
                    except Exception as exc:
                        logging.error(f"Page {page_num} for {date_str_for_req} failed, adding to retry list: {exc}")
                        failed_pages.append({"date": current_date_obj.strftime('%Y-%m-%d'), "page": page_num})
                
                final_page_to_save = max_successful_page_in_batch
                current_page = final_page_to_save + 1
                if final_page_to_save >= total_pages_for_day:
                    current_date_obj += timedelta(days=1)
                    current_page = 1

    except (RateLimitException, KeyboardInterrupt, Exception) as e:
        logging.warning(f"Collection stopped unexpectedly by {type(e).__name__}: {e}. Saving current state.")
    finally:
        # Сохранение результатов и состояния
        if collected_movies:
            logging.info(f"\nCollection finished. Total movies collected this run: {len(collected_movies)}.")
            output_dir = Path("output_raw")
            output_dir.mkdir(exist_ok=True)
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
            local_archive_path = output_dir / archive_filename
            collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
            with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
                for movie in collected_movies:
                    f.write(json.dumps(movie, ensure_ascii=False) + '\n')
            logging.info(f"Saved data to {local_archive_path}")
            try:
                logging.info(f"Uploading {archive_filename} to dataset...")
                api.upload_file(
                    path_or_fileobj=str(local_archive_path),
                    path_in_repo=f"{RAW_DATA_DIR}/{archive_filename}",
                    repo_id=DATASET_ID, repo_type="dataset",
                )
                logging.info("Raw chunk upload successful.")
            except Exception as e:
                logging.critical(f"Failed to upload data chunk: {e}", exc_info=True)
                sys.exit(1) # Выходим с ошибкой, чтобы не обновлять состояние
        else:
            logging.info("No new movies collected in this run.")
        
        # Всегда обновляем состояние в конце
        try:
            # Если мы закончили день, сохраняем следующую дату и 0-ю страницу
            if current_page > final_page_to_save and final_page_to_save > 0:
                 update_collector_state(api, final_date_to_save.strftime('%Y-%m-%d'), 0, failed_pages)
            else:
                 update_collector_state(api, final_date_to_save.strftime('%Y-%m-%d'), final_page_to_save, failed_pages)
        except Exception as e:
            logging.critical(f"CRITICAL: Failed to update FINAL state: {e}", exc_info=True)
            sys.exit(1)

    logging.info("\nCollector run completed successfully.")

if __name__ == "__main__":
    main()


