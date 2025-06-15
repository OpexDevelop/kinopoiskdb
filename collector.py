import os
import requests
import gzip
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

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
    """
    Получает состояние сборщика из репозитория.
    Состояние хранит последнюю *успешно и последовательно* обработанную дату и страницу.
    """
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
    """
    Обновляет файл состояния в репозитории.
    Удаляет дубликаты из списка сбойных страниц перед сохранением.
    """
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

def fetch_page(task, api_key):
    """
    Запрашивает одну страницу по данным из объекта 'task'.
    'task' - это словарь вида {'date': 'ГГГГ-ММ-ДД', 'page': N}.
    """
    date_obj = datetime.strptime(task['date'], '%Y-%m-%d').date()
    # API ожидает дату в формате ДД.ММ.ГГГГ
    date_str_for_req = date_obj.strftime('%d.%m.%Y')
    page = task['page']

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

    response.raise_for_status() # Вызовет исключение для других ошибок (4xx, 5xx)
    return response.json()

def calculate_new_state(last_known_date_str, last_known_page, successful_tasks):
    """
    Вычисляет новое состояние на основе списка успешно выполненных задач.
    Находит последнюю НЕПРЕРЫВНУЮ последовательность успешно скачанных страниц.
    """
    # Создаем словарь, где ключ - дата, значение - отсортированный список успешных страниц
    completed_by_date = defaultdict(list)
    for task in successful_tasks:
        completed_by_date[task['date']].append(task['page'])
    for date in completed_by_date:
        completed_by_date[date].sort()

    # Начинаем с последней известной точки
    current_date = datetime.strptime(last_known_date_str, '%Y-%m-%d').date()
    current_page = last_known_page
    
    # Сортируем даты, чтобы проверять их по порядку
    sorted_dates = sorted(completed_by_date.keys())

    while True:
        current_date_str = current_date.strftime('%Y-%m-%d')
        # Если текущей даты нет в словаре успешных, значит последовательность прервалась
        if current_date_str not in completed_by_date:
            break

        # Проверяем страницы для текущей даты
        expected_page = current_page + 1
        pages_for_date = completed_by_date[current_date_str]
        
        found_break = False
        while expected_page in pages_for_date:
            current_page = expected_page
            expected_page += 1
        
        # Если мы не нашли следующую страницу, но это не последняя страница
        # (т.е. есть пропуски), то последовательность прервана
        if current_page < max(pages_for_date):
             # Проверяем, есть ли разрывы
            if (current_page + 1) not in pages_for_date:
                 found_break = True

        if found_break:
            break

        # Если дошли до конца дня, переходим на следующий день
        current_date += timedelta(days=1)
        current_page = 0
    
    return current_date_str, current_page


def main():
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("No API keys provided in KINOPOISK_API_KEYS secret.")
        sys.exit(1)

    # Используем циклический доступ к ключам, чтобы работать с любым их количеством
    current_hour = datetime.utcnow().hour
    api_key = api_keys[current_hour % len(api_keys)]
    logging.info(f"Current UTC hour is {current_hour}. Using API key index #{current_hour % len(api_keys)}.")

    api = HfApi(token=hf_token)
    
    # Получаем исходное состояние
    last_date_str, last_page_completed, failed_pages = get_collector_state(api)

    requests_processed = 0
    collected_movies = []
    successful_tasks = []
    
    # Этот словарь будет отслеживать дни, для которых мы знаем, что страниц больше нет
    # Ключ: дата (str), Значение: общее кол-во страниц (int)
    known_total_pages = {}

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = {}

            # --- ФАЗА 1: ПОВТОРНАЯ ПОПЫТКА ДЛЯ СБОЙНЫХ СТРАНИЦ ---
            if failed_pages and requests_processed < max_requests:
                logging.info(f"--- Starting RETRY phase for {len(failed_pages)} failed pages ---")
                
                # Создаем задачи для сбойных страниц
                tasks_to_retry = failed_pages[:(max_requests - requests_processed)]
                for task in tasks_to_retry:
                    # Подаем задачу в executor и сохраняем future
                    future = executor.submit(fetch_page, task, api_key)
                    futures[future] = task
                
                # Оставшиеся сбои, которые не попали в этот запуск
                failed_pages = failed_pages[len(tasks_to_retry):]

            # --- ФАЗА 2: ГЕНЕРАЦИЯ И ВЫПОЛНЕНИЕ НОВЫХ ЗАДАЧ ---
            logging.info(f"--- Starting MAIN collection from {last_date_str} page {last_page_completed + 1} ---")
            
            # Начинаем генерацию задач с последней известной точки
            date_to_process = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            page_to_process = last_page_completed + 1

            while len(futures) < max_requests:
                # Остановка, если дата ушла слишком далеко в будущее
                if date_to_process > datetime.utcnow().date() + timedelta(days=1):
                    logging.info("Target date is in the future. Stopping task generation.")
                    break
                
                current_date_str = date_to_process.strftime('%Y-%m-%d')

                # Если мы знаем, что для этой даты больше нет страниц, пропускаем ее
                if current_date_str in known_total_pages and page_to_process > known_total_pages[current_date_str]:
                    date_to_process += timedelta(days=1)
                    page_to_process = 1
                    continue
                
                # Создаем и отправляем новую задачу
                task = {'date': current_date_str, 'page': page_to_process}
                future = executor.submit(fetch_page, task, api_key)
                futures[future] = task

                page_to_process += 1

            # --- ФАЗА 3: ОБРАБОТКА РЕЗУЛЬТАТОВ ПО МЕРЕ ИХ ПОСТУПЛЕНИЯ ---
            logging.info(f"--- Processing {len(futures)} tasks in parallel ---")
            for future in as_completed(futures):
                task = futures[future]
                requests_processed += 1
                try:
                    data = future.result()
                    docs = data.get('docs')
                    
                    if docs:
                        collected_movies.extend(docs)
                        successful_tasks.append(task)
                        logging.info(f"[SUCCESS] Fetched page {task['page']} for date {task['date']}. Movies: {len(docs)}")
                        
                        # Сохраняем информацию об общем количестве страниц для этой даты
                        total_pages = data.get('pages', task['page'])
                        known_total_pages[task['date']] = total_pages
                    else:
                        # Если данных нет, значит, это последняя страница для этой даты
                        logging.warning(f"[OK/EMPTY] Page {task['page']} for {task['date']} was empty. Assuming end of day.")
                        known_total_pages[task['date']] = task['page'] - 1

                except Exception as exc:
                    logging.error(f"[FAIL] Page {task['page']} for date {task['date']} failed: {exc}")
                    failed_pages.append(task)
                    if isinstance(exc, RateLimitException):
                        # При исчерпании лимита ключа нет смысла продолжать
                        raise

    except (RateLimitException, KeyboardInterrupt, Exception) as e:
        logging.warning(f"Collection stopped unexpectedly by {type(e).__name__}: {e}. Saving current state.")
    finally:
        # --- ФАЗА 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ И СОСТОЯНИЯ ---
        if collected_movies:
            logging.info(f"\nCollected {len(collected_movies)} movies this run.")
            output_dir = Path("output_raw")
            output_dir.mkdir(exist_ok=True)
            timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
            local_archive_path = output_dir / archive_filename
            
            # Сортировка для консистентности, хотя и не строго обязательна
            collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
            
            with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
                for movie in collected_movies:
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
                sys.exit(1)
        else:
            logging.info("No new movies collected in this run.")
        
        try:
            # Вычисляем новое состояние на основе всех успешных задач
            new_date_str, new_page_num = calculate_new_state(last_date_str, last_page_completed, successful_tasks)
            update_collector_state(api, new_date_str, new_page_num, failed_pages)
        except Exception as e:
            logging.critical(f"CRITICAL: Failed to update FINAL state: {e}", exc_info=True)
            sys.exit(1)

    logging.info("\nCollector run completed.")

if __name__ == "__main__":
    main()


