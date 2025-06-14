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
import threading

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
        state_path = hf_hub_download(
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

def fetch_page(page, date_str, api_key):
    """Fetches a single page for a specific date."""
    try:
        current_day_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        next_day_obj = current_day_obj + timedelta(days=1)
        
        date_from = current_day_obj.strftime('%d.%m.%Y')
        date_to = next_day_obj.strftime('%d.%m.%Y')
    except ValueError:
        logging.error(f"Invalid date format: {date_str}. Skipping.")
        return None

    params = {
        'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1',
        'updatedAt': f"{date_from}-{date_to}",
        'selectFields': ''
    }
    headers = {"X-API-KEY": api_key}
    
    logging.info(f"Requesting page {page} for date range {date_from}-{date_to}...")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

    if response.status_code == 403:
        raise RateLimitException("Rate limit exceeded (403 Forbidden).")
    
    response.raise_for_status()
    return response.json()

class SequentialPageCollector:
    """Manages sequential page collection with parallel execution."""
    
    def __init__(self, api_key, max_concurrent_requests=10):
        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests
        self.collected_movies = []
        self.failed_pages = []
        self.requests_made = 0
        self.lock = threading.Lock()
    
    def collect_pages_for_date(self, date_str, start_page, max_requests):
        """Collect pages for a specific date sequentially but with parallel execution."""
        current_page = start_page
        
        while self.requests_made < max_requests:
            # Get initial page to understand total pages
            try:
                data = fetch_page(current_page, date_str, self.api_key)
                with self.lock:
                    self.requests_made += 1
                
                if not data or not data.get('docs'):
                    logging.info(f"No more data for date {date_str} starting at page {current_page}")
                    return current_page - 1, True  # Finished this date
                
                # Add movies from initial page
                with self.lock:
                    self.collected_movies.extend(data['docs'])
                
                total_pages = data.get('pages', current_page)
                logging.info(f"Date {date_str}, page {current_page}/{total_pages}")
                
                if current_page >= total_pages:
                    return current_page, True  # Finished this date
                
                # Calculate how many more pages we can fetch
                remaining_requests = max_requests - self.requests_made
                if remaining_requests <= 0:
                    return current_page, False  # Hit request limit
                
                # Determine next batch of pages to fetch
                next_page = current_page + 1
                last_page_to_fetch = min(total_pages, current_page + remaining_requests)
                pages_to_fetch = list(range(next_page, last_page_to_fetch + 1))
                
                if not pages_to_fetch:
                    return current_page, True  # No more pages
                
                # Fetch pages in parallel
                self._fetch_pages_parallel(pages_to_fetch, date_str)
                
                current_page = last_page_to_fetch
                
                if current_page >= total_pages:
                    return current_page, True  # Finished this date
                
                current_page += 1
                
            except Exception as e:
                logging.error(f"Error fetching page {current_page} for date {date_str}: {e}")
                with self.lock:
                    self.failed_pages.append({"date": date_str, "page": current_page})
                current_page += 1
        
        return current_page - 1, False  # Hit request limit
    
    def _fetch_pages_parallel(self, pages, date_str):
        """Fetch multiple pages in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = {
                executor.submit(fetch_page, page, date_str, self.api_key): page
                for page in pages
            }
            
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    data = future.result()
                    with self.lock:
                        self.requests_made += 1
                    
                    if data and data.get('docs'):
                        with self.lock:
                            self.collected_movies.extend(data['docs'])
                        logging.info(f"Successfully fetched page {page_num} for date {date_str}")
                    else:
                        logging.warning(f"Empty page {page_num} for date {date_str}")
                        
                except Exception as e:
                    logging.error(f"Failed to fetch page {page_num} for date {date_str}: {e}")
                    with self.lock:
                        self.failed_pages.append({"date": date_str, "page": page_num})
    
    def retry_failed_pages(self, failed_pages_list, max_requests):
        """Retry failed pages in parallel."""
        if not failed_pages_list or self.requests_made >= max_requests:
            return []
        
        remaining_requests = max_requests - self.requests_made
        pages_to_retry = failed_pages_list[:remaining_requests]
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = {
                executor.submit(fetch_page, item['page'], item['date'], self.api_key): item
                for item in pages_to_retry
            }
            
            still_failing = []
            for future in as_completed(futures):
                item = futures[future]
                try:
                    data = future.result()
                    with self.lock:
                        self.requests_made += 1
                    
                    if data and data.get('docs'):
                        with self.lock:
                            self.collected_movies.extend(data['docs'])
                        logging.info(f"[RETRY SUCCESS] Page {item['page']} for date {item['date']}")
                    else:
                        logging.warning(f"[RETRY EMPTY] Page {item['page']} for date {item['date']}")
                        
                except Exception as e:
                    logging.error(f"[RETRY FAIL] Page {item['page']} for date {item['date']}: {e}")
                    still_failing.append(item)
        
        # Return pages that still failed + pages we didn't attempt
        return still_failing + failed_pages_list[len(pages_to_retry):]

def main():
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided.")
        sys.exit(1)

    # ИСПРАВЛЕННАЯ логика выбора ключа
    num_keys = len(api_keys)
    current_hour = datetime.utcnow().hour
    
    # Каждый ключ должен работать один раз в 24 часа
    # Ключ 0 работает в часы 0, 24, 48... (0 % 24 == 0)
    # Ключ 1 работает в часы 1, 25, 49... (1 % 24 == 1)
    # и т.д.
    key_index = current_hour % num_keys
    
    # Если это запуск по расписанию, проверяем, должен ли этот ключ работать сейчас
    if os.getenv("GITHUB_EVENT_NAME") == "schedule":
        hours_per_key = 24 // num_keys
        if hours_per_key == 0:
            hours_per_key = 1
        
        # Проверяем, что текущий час соответствует слоту для этого ключа
        key_hour_slot = key_index * hours_per_key
        if current_hour != key_hour_slot:
            logging.info(f"Not the scheduled time for key #{key_index}. Current hour: {current_hour}, key slot: {key_hour_slot}. Skipping.")
            sys.exit(0)
    
    api_key = api_keys[key_index]
    logging.info(f"Using API key #{key_index} (hour {current_hour}).")
    
    api = HfApi(token=hf_token)
    
    current_date_str, last_page_for_date, failed_pages = get_collector_state(api)
    current_date_obj = datetime.strptime(current_date_str, '%Y-%m-%d').date()
    
    collector = SequentialPageCollector(api_key, MAX_CONCURRENT_REQUESTS)
    
    try:
        # Phase 1: Retry failed pages
        if failed_pages:
            logging.info(f"--- RETRY PHASE: {len(failed_pages)} failed pages ---")
            failed_pages = collector.retry_failed_pages(failed_pages, max_requests)
            logging.info(f"--- RETRY FINISHED: {len(failed_pages)} pages still failing ---")
        
        # Phase 2: Main collection
        logging.info("--- MAIN COLLECTION PHASE ---")
        final_date = current_date_obj
        final_page = last_page_for_date
        
        while collector.requests_made < max_requests:
            if current_date_obj > datetime.utcnow().date():
                logging.info("Reached future date. Stopping.")
                break
            
            start_page = last_page_for_date + 1 if current_date_obj.strftime('%Y-%m-%d') == current_date_str else 1
            
            last_page, date_finished = collector.collect_pages_for_date(
                current_date_obj.strftime('%Y-%m-%d'), 
                start_page, 
                max_requests
            )
            
            final_date = current_date_obj
            final_page = last_page
            
            if date_finished:
                logging.info(f"Finished collecting all pages for {current_date_obj}")
                current_date_obj += timedelta(days=1)
                last_page_for_date = 0
            else:
                logging.info(f"Hit request limit on {current_date_obj} at page {last_page}")
                last_page_for_date = last_page
                break
        
        # Merge collector's failed pages with existing ones
        failed_pages.extend(collector.failed_pages)
        
    except (RateLimitException, KeyboardInterrupt) as e:
        logging.warning(f"Collection stopped: {type(e).__name__}")
        final_date = current_date_obj
        final_page = last_page_for_date
        failed_pages.extend(collector.failed_pages)
    
    # Save results
    if not collector.collected_movies:
        logging.info("No new movies collected in this run.")
        update_collector_state(api, final_date.strftime('%Y-%m-%d'), final_page, failed_pages)
        sys.exit(0)

    logging.info(f"Collection finished. Total movies collected: {len(collector.collected_movies)}, Requests made: {collector.requests_made}")
    
    output_dir = Path("output_raw")
    output_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
    local_archive_path = output_dir / archive_filename
    
    with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
        collector.collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
        for movie in collector.collected_movies:
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
        update_collector_state(api, final_date.strftime('%Y-%m-%d'), final_page, failed_pages)
    except Exception as e:
        logging.critical(f"Failed to upload data or state: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Collector run completed successfully.")

if __name__ == "__main__":
    main()
