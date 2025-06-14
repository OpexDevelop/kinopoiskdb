import os
import requests
import gzip
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# --- Configuration ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
STATE_FILENAME = "collector_state.json"
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_START_DATE = "1970-01-01"
MAX_CONCURRENT_REQUESTS = 10

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RateLimitException(Exception):
    pass

def get_env_var(var_name, default=None):
    """Gets an environment variable or exits if it's not set."""
    value = os.getenv(var_name, default)
    if value is None:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def get_collector_state(api: HfApi):
    """
    Gets the collector state from the repository.
    Format: {"last_date": "YYYY-MM-DD", "last_page": N, "failed_pages": []}
    """
    try:
        logging.info(f"Attempting to download state file: {STATE_FILENAME}")
        state_path = api.hf_hub_download(
            repo_id=DATASET_ID, filename=STATE_FILENAME, repo_type="dataset"
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        last_date = state.get("last_date", DEFAULT_START_DATE)
        last_page = state.get("last_page", 0)
        failed_pages = state.get("failed_pages", [])
        logging.info(f"Found state. Main position: {last_date}, page {last_page}. Failed pages to retry: {len(failed_pages)}")
        return last_date, last_page, failed_pages
    except EntryNotFoundError:
        logging.warning("State file not found. Starting from the beginning.")
        return DEFAULT_START_DATE, 0, []
    except (json.JSONDecodeError, Exception) as e:
        logging.critical(f"An unexpected error occurred while fetching state: {e}")
        return DEFAULT_START_DATE, 0, []

def update_collector_state(api: HfApi, date_str: str, page_num: int, failed_pages: list):
    """Updates the state file in the repository."""
    unique_failed = [dict(t) for t in {tuple(d.items()) for d in failed_pages}]
    state = {"last_date": date_str, "last_page": page_num, "failed_pages": unique_failed}
    
    local_state_path = Path(STATE_FILENAME)
    with open(local_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    logging.info(f"Uploading new state. Main position: {date_str}, page {page_num}. Failed pages: {len(unique_failed)}")
    api.upload_file(
        path_or_fileobj=str(local_state_path),
        path_in_repo=STATE_FILENAME,
        repo_id=DATASET_ID, repo_type="dataset"
    )

def fetch_page_for_date(page, date_str, api_key):
    """Fetches a single page for a specific date, returning page number, date, and data."""
    try:
        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # Формируем диапазон с текущего дня до СЛЕДУЮЩЕГО, чтобы получить данные за один день.
        current_day_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        next_day_obj = current_day_obj + timedelta(days=1)
        
        date_from = current_day_obj.strftime('%d.%m.%Y')
        date_to = next_day_obj.strftime('%d.%m.%Y')
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    except ValueError:
        logging.error(f"Invalid date format: {date_str}. Skipping.")
        return date_str, page, None

    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': f"{date_from}-{date_to}", # Используем правильный диапазон
        'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}
    
    logging.info(f"Requesting page {page} for date range {date_from}-{date_to}...")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

    if response.status_code == 403:
        raise RateLimitException("Rate limit exceeded (403 Forbidden).")
    
    response.raise_for_status()
    return date_str, page, response.json()

def main():
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided.")
        sys.exit(1)

    # Key selection logic
    num_tokens = len(api_keys)
    current_hour = datetime.utcnow().hour
    interval = 24 // num_tokens if num_tokens > 0 else 24
    if interval == 0: interval = 1
    if current_hour % interval != 0 and os.getenv("GITHUB_EVENT_NAME") == "schedule":
        logging.info("Not a scheduled slot for this token. Skipping.")
        sys.exit(0)
    key_index = (current_hour // interval) % num_tokens
    api_key = api_keys[key_index]
    logging.info(f"Using API key #{key_index}.")
    
    api = HfApi(token=hf_token)
    
    current_date_str, last_page_for_date, failed_pages = get_collector_state(api)
    current_page = last_page_for_date + 1
    current_date_obj = datetime.strptime(current_date_str, '%Y-%m-%d').date()

    requests_processed = 0
    collected_movies = []
    
    final_date_to_save = current_date_obj
    final_page_to_save = last_page_for_date
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # --- PHASE 1: RETRY FAILED PAGES ---
            if failed_pages and requests_processed < max_requests:
                logging.info(f"--- Starting RETRY phase for {len(failed_pages)} failed pages ---")
                
                retries_to_attempt = failed_pages[:max_requests]
                
                retry_futures = {
                    executor.submit(fetch_page_for_date, item['page'], item['date'], api_key): item
                    for item in retries_to_attempt
                }
                
                still_failing = []
                for future in as_completed(retry_futures):
                    item = retry_futures[future]
                    requests_processed += 1
                    try:
                        _date, _page, data = future.result()
                        if data and data.get('docs'):
                            collected_movies.extend(data['docs'])
                            logging.info(f"[RETRY SUCCESS] Successfully fetched page {item['page']} for date {item['date']}")
                        else:
                            logging.warning(f"[RETRY OK] Page {item['page']} for {item['date']} was empty. Removing from retry list.")
                    except Exception as exc:
                        logging.error(f"[RETRY FAIL] Page {item['page']} for date {item['date']} failed again: {exc}")
                        still_failing.append(item)
                
                failed_pages = still_failing + failed_pages[len(retries_to_attempt):]
                logging.info(f"--- RETRY phase finished. {len(still_failing)} pages still failing. ---")


            # --- PHASE 2: MAIN COLLECTION ---
            logging.info(f"--- Starting MAIN collection phase ---")
            while requests_processed < max_requests:
                if current_date_obj > datetime.utcnow().date():
                    logging.info("Target date is in the future. Stopping.")
                    break

                # 1. Probe request
                try:
                    _date, _page, initial_data = fetch_page_for_date(current_page, current_date_obj.strftime('%Y-%m-%d'), api_key)
                    requests_processed += 1
                except Exception as exc:
                    logging.error(f"Probe request failed for date {current_date_obj}, page {current_page}: {exc}")
                    failed_pages.append({"date": current_date_obj.strftime('%Y-%m-%d'), "page": current_page})
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                    continue
                
                if not initial_data or not initial_data.get('docs'):
                    logging.info(f"No data for date {current_date_obj} starting at page {current_page}. Moving to next day.")
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                    continue

                # 2. Save initial data and get metadata
                collected_movies.extend(initial_data['docs'])
                total_pages_for_day = initial_data.get('pages', current_page)
                logging.info(f"Probe successful. Date {current_date_obj} has {total_pages_for_day} total pages.")
                final_date_to_save = current_date_obj
                final_page_to_save = current_page

                # 3. Create a batch for the remaining pages
                remaining_pages = range(current_page + 1, total_pages_for_day + 1)
                if not remaining_pages:
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                    continue

                pages_to_fetch_count = min(len(remaining_pages), max_requests - requests_processed)
                pages_to_fetch = list(remaining_pages)[:pages_to_fetch_count]

                if not pages_to_fetch: break

                futures = {
                    executor.submit(fetch_page_for_date, page, current_date_obj.strftime('%Y-%m-%d'), api_key): page
                    for page in pages_to_fetch
                }
                
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        _date, _page, data = future.result()
                        requests_processed += 1
                        if data and data.get('docs'):
                            collected_movies.extend(data['docs'])
                            final_page_to_save = _page 
                    except Exception as exc:
                        logging.error(f"Page {page_num} for {current_date_obj} failed, adding to retry list: {exc}")
                        failed_pages.append({"date": current_date_obj.strftime('%Y-%m-%d'), "page": page_num})
                
                current_page = final_page_to_save + 1
                if final_page_to_save >= total_pages_for_day:
                    logging.info(f"Finished collecting all pages for {current_date_obj}. Moving to the next day.")
                    current_date_obj += timedelta(days=1)
                    current_page = 1
                
    except (RateLimitException, KeyboardInterrupt) as e:
        logging.warning(f"Collection stopped: {type(e).__name__}")
    finally:
        if not collected_movies:
            logging.info("No new movies collected in this run.")
            update_collector_state(api, final_date_to_save.strftime('%Y-%m-%d'), final_page_to_save, failed_pages)
            sys.exit(0)

        logging.info(f"\nCollection finished. Total movies collected this run: {len(collected_movies)}.")
        
        output_dir = Path("output_raw")
        output_dir.mkdir(exist_ok=True)
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
        local_archive_path = output_dir / archive_filename
        
        with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
            collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
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
            update_collector_state(api, final_date_to_save.strftime('%Y-%m-%d'), final_page_to_save, failed_pages)
        except Exception as e:
            logging.critical(f"Failed to upload data or state: {e}", exc_info=True)
            sys.exit(1)

    logging.info("\nCollector run completed successfully.")

if __name__ == "__main__":
    main()

