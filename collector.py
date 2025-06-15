import os
import requests
import gzip
import json
import sys
import logging
import threading
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- КОНФИГУРАЦИЯ ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
STATE_FILENAME = "collector_state.json"
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_START_DATE = "2000-01-01"
MAX_MINER_WORKERS = 15
MAX_RETRY_WORKERS = 10

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Подключение зависимостей Hugging Face Hub
try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
except ImportError:
    logging.error("Hugging Face Hub library not found. Please install it: pip install huggingface_hub")
    sys.exit(1)

class RateLimitException(Exception):
    """Кастомное исключение для ошибок лимита запросов."""
    pass

def get_env_var(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        logging.critical(f"Критическая ошибка: Переменная окружения {var_name} не установлена.")
        sys.exit(1)
    return value

# --- УПРАВЛЕНИЕ СОСТОЯНИЕМ (STATE) ---
STATE_FILE_LOCK = threading.Lock()

def get_state(api: HfApi):
    """Скачивает файл состояния из репозитория Hugging Face."""
    default_state = {
        "scout_state": {"next_scan_start_date": DEFAULT_START_DATE},
        "miner_state": {"is_drilling": False, "drilling_date": None, "last_completed_page": 0, "total_pages": 0},
        "failed_pages": []
    }
    try:
        logging.info(f"Попытка скачать файл состояния: {STATE_FILENAME}")
        state_path = hf_hub_download(
            repo_id=DATASET_ID, filename=STATE_FILENAME, repo_type="dataset",
            force_download=True, resume_download=False
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        state.setdefault("scout_state", default_state["scout_state"])
        state.setdefault("miner_state", default_state["miner_state"])
        state.setdefault("failed_pages", default_state["failed_pages"])
        logging.info(f"Файл состояния успешно загружен. Найдено сбойных страниц: {len(state['failed_pages'])}")
        return state
    except (HfHubHTTPError, EntryNotFoundError):
        logging.warning("Файл состояния не найден. Будет создан новый.")
        return default_state
    except Exception as e:
        logging.critical(f"Критическая ошибка при чтении файла состояния: {e}. Начинаем с нуля.")
        return default_state

def update_state(api: HfApi, state: dict):
    """Сохраняет объект состояния в локальный файл и загружает его в репозиторий."""
    with STATE_FILE_LOCK:
        if "failed_pages" in state and state["failed_pages"]:
            unique_failed = [dict(t) for t in {tuple(d.items()) for d in state["failed_pages"]}]
            state["failed_pages"] = unique_failed
        
        logging.info(f"Сохранение нового состояния: {state}")
        local_state_path = Path(STATE_FILENAME)
        with open(local_state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        try:
            api.upload_file(
                path_or_fileobj=str(local_state_path),
                path_in_repo=STATE_FILENAME,
                repo_id=DATASET_ID, repo_type="dataset"
            )
            logging.info("Файл состояния успешно загружен на Hugging Face.")
        except Exception as e:
            logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить файл состояния! {e}")

def save_data_chunk(api: HfApi, collected_movies: list):
    """Сохраняет собранные фильмы в .jsonl.gz и загружает в репозиторий."""
    if not collected_movies:
        logging.info("Новых фильмов для сохранения нет.")
        return

    logging.info(f"\nСохранение {len(collected_movies)} фильмов...")
    output_dir = Path("output_raw")
    output_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    archive_filename = f"raw_chunk_{timestamp_str}.jsonl.gz"
    local_archive_path = output_dir / archive_filename
    
    collected_movies.sort(key=lambda m: m.get('updatedAt', ''))
    
    with gzip.open(local_archive_path, 'wt', encoding='utf-8') as f:
        for movie in collected_movies:
            f.write(json.dumps(movie, ensure_ascii=False) + '\n')
    logging.info(f"Данные сохранены в {local_archive_path}")
    
    try:
        api.upload_file(
            path_or_fileobj=str(local_archive_path),
            path_in_repo=f"{RAW_DATA_DIR}/{archive_filename}",
            repo_id=DATASET_ID, repo_type="dataset",
        )
        logging.info("Фрагмент данных успешно загружен.")
    except Exception as e:
        logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить фрагмент данных: {e}", exc_info=True)


# --- ЛОГИКА API ЗАПРОСОВ ---

def log_data_summary(docs: list, req_type: str):
    """Создает и логирует детальную сводку по полученным данным."""
    if not docs:
        logging.info(f"[{req_type}] Найдено 0 фильмов.")
        return None
    
    movie_count = len(docs)
    
    date_counts = Counter(d.get('updatedAt', '').split('T')[0] for d in docs if d.get('updatedAt'))
    unique_dates_count = len(date_counts)
    
    date_distribution_parts = [f"{date}: {count} фильм(ов)" for date, count in sorted(date_counts.items())]
    date_distribution_str = "; ".join(date_distribution_parts)
    
    summary = (f"Найдено {movie_count} фильмов в {unique_dates_count} уникальных датах. "
               f"Распределение: [ {date_distribution_str} ]")
    
    logging.info(f"[{req_type}] {summary}")
    
    return date_counts

def fetch_page(api_key: str, date_for_req: str, page: int, req_type: str = "REQUEST"):
    """Универсальная функция для запросов к API."""
    params = {'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1', 'updatedAt': date_for_req, 'selectFields': ''}
    headers = {"X-API-KEY": api_key}
    logging.info(f"[{req_type}] Запрос: страница {page}, дата/диапазон {date_for_req}")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 403:
        raise RateLimitException("Лимит запросов исчерпан (403 Forbidden).")
    response.raise_for_status()
    return response.json()


# --- РЕЖИМЫ РАБОТЫ ---

def run_retrier(failed_pages_list: list, api_key: str, requests_limit: int):
    """"Восстановитель". Пытается параллельно скачать страницы из списка сбоев."""
    if not failed_pages_list or requests_limit <= 0:
        return [], 0, [], failed_pages_list

    logging.info(f"--- Запуск 'Восстановителя' для {len(failed_pages_list)} сбойных страниц. ---")
    
    collected_movies = []
    requests_used = 0
    successfully_retried_tasks = []
    still_failing_tasks = []

    tasks_to_retry = failed_pages_list[:requests_limit]

    with ThreadPoolExecutor(max_workers=MAX_RETRY_WORKERS) as executor:
        future_to_task = {}
        for task in tasks_to_retry:
            date_obj = datetime.strptime(task['date'], '%Y-%m-%d').date()
            next_date = date_obj + timedelta(days=1)
            date_range = f"{date_obj.strftime('%d.%m.%Y')}-{next_date.strftime('%d.%m.%Y')}"
            future = executor.submit(fetch_page, api_key, date_range, task['page'], "RETRIER")
            future_to_task[future] = task
            requests_used += 1

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                docs = data.get('docs', [])
                log_data_summary(docs, f"RETRIER-SUCCESS page {task['page']}")
                if docs:
                    collected_movies.extend(docs)
                successfully_retried_tasks.append(task)
            except Exception as e:
                logging.error(f"[RETRIER FAIL] Страница {task['page']} для {task['date']} снова не удалась: {e}")
                still_failing_tasks.append(task)
    
    remaining_tasks = [t for t in failed_pages_list if t not in tasks_to_retry]
    final_failed_list = still_failing_tasks + remaining_tasks

    return collected_movies, requests_used, successfully_retried_tasks, final_failed_list


def run_miner(state: dict, api_key: str, requests_limit: int):
    """РЕЖИМ "ШАХТЁРА" - параллельное скачивание всех страниц найденного "месторождения"."""
    miner_state = state["miner_state"]
    drilling_date_str = miner_state["drilling_date"]
    start_page = miner_state["last_completed_page"] + 1
    total_pages = miner_state["total_pages"]
    
    requests_used = 0
    collected_movies = []
    newly_failed_pages = []

    drilling_date_obj = datetime.strptime(drilling_date_str, '%Y-%m-%d').date()
    next_date_obj = drilling_date_obj + timedelta(days=1)
    date_range_for_req = f"{drilling_date_obj.strftime('%d.%m.%Y')}-{next_date_obj.strftime('%d.%m.%Y')}"

    if total_pages == 0:
        if requests_used >= requests_limit: 
            return collected_movies, requests_used, False, newly_failed_pages
            
        logging.info(f"[MINER] Контрольный запрос для диапазона {date_range_for_req}, чтобы узнать общее кол-во страниц.")
        data = fetch_page(api_key, date_range_for_req, 1, "MINER_CONTROL")
        requests_used += 1
        total_pages = data.get('pages', 0)
        miner_state["total_pages"] = total_pages
        logging.info(f"[MINER] В дне {drilling_date_str} найдено {total_pages} страниц.")

        # Добавляем первую страницу к собранным фильмам
        docs = data.get('docs', [])
        if docs:
            collected_movies.extend(docs)
            miner_state["last_completed_page"] = 1
            
        # Запрашиваем вторую страницу
        if total_pages > 1 and requests_used < requests_limit:
            try:
                data = fetch_page(api_key, date_range_for_req, 2, "MINER_SECOND_PAGE")
                requests_used += 1
                docs = data.get('docs', [])
                log_data_summary(docs, "MINER_SECOND_PAGE")
                if docs:
                    collected_movies.extend(docs)
                    miner_state["last_completed_page"] = 2
            except Exception as e:
                logging.error(f"[MINER_SECOND_PAGE] Ошибка при скачивании страницы 2 для {drilling_date_str}: {e}")
                newly_failed_pages.append({"date": drilling_date_str, "page": 2})
                
    # Страницы 3+ идут параллельно
    pages_to_drill = list(range(max(start_page, 3), total_pages + 1))
    if not pages_to_drill:
        return collected_movies, requests_used, True, []

    with ThreadPoolExecutor(max_workers=MAX_MINER_WORKERS) as executor:
        future_to_page = {}
        for page in pages_to_drill:
            if requests_used >= requests_limit:
                logging.warning("[MINER] Достигнут лимит запросов, приостановка бурения.")
                break
            
            future = executor.submit(fetch_page, api_key, date_range_for_req, page, "MINER_WORKER")
            future_to_page[future] = page
            requests_used += 1

        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                data = future.result()
                docs = data.get('docs', [])
                log_data_summary(docs, f"MINER-WORKER page {page}")
                if docs:
                    collected_movies.extend(docs)
                miner_state["last_completed_page"] = max(miner_state.get("last_completed_page", 0), page)
            except Exception as e:
                logging.error(f"[MINER_WORKER] Ошибка при скачивании страницы {page} для {drilling_date_str}: {e}")
                newly_failed_pages.append({"date": drilling_date_str, "page": page})
    
    is_day_finished = miner_state["last_completed_page"] >= total_pages
    return collected_movies, requests_used, is_day_finished, newly_failed_pages


def run_scout(state: dict, api_key: str, requests_limit: int):
    """РЕЖИМ "РАЗВЕДЧИКА" - последовательный поиск "месторождений" данных."""
    if requests_limit <= 0: 
        return [], 0, None, []

    scout_state = state["scout_state"]
    start_date_str = scout_state["next_scan_start_date"]
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    
    end_date_str = "01.01.2050"
    
    date_range_for_req = f"{start_date.strftime('%d.%m.%Y')}-{end_date_str}"
    
    requests_used = 0
    collected_movies = []
    newly_failed_pages = []
    
    # Начинаем с первой страницы
    logging.info(f"[SCOUT] Зондирование диапазона {date_range_for_req}")
    try:
        data = fetch_page(api_key, date_range_for_req, 1, "SCOUT_PROBE")
        requests_used += 1
        docs = data.get('docs', [])
        date_counts = log_data_summary(docs, "SCOUT_PROBE")
        
        total_pages = data.get('pages', 0)
        
        if total_pages <= 1 and not docs:
            logging.info("[SCOUT] Диапазон пуст. Сохраняем текущую начальную дату для повторной проверки в следующем запуске.")
            return [], requests_used, None, []
            
        # Проверяем, есть ли на странице 1 только одна дата (месторождение)
        if date_counts and len(date_counts) == 1:
            motherlode_date = list(date_counts.keys())[0]
            logging.info(f"[SCOUT] Найдено 'месторождение' данных на первой странице с датой {motherlode_date}!")
            next_day = datetime.strptime(motherlode_date, '%Y-%m-%d').date() + timedelta(days=1)
            scout_state["next_scan_start_date"] = next_day.strftime('%Y-%m-%d')
            return docs, requests_used, motherlode_date, []
            
        # Добавляем первую страницу к собранным фильмам
        collected_movies.extend(docs)
        
        # Последовательно проверяем страницы 2+ того же диапазона, пока не найдем месторождение
        page = 2
        while page <= total_pages and requests_used < requests_limit:
            try:
                data = fetch_page(api_key, date_range_for_req, page, "SCOUT_COLLECT")
                requests_used += 1
                docs_page = data.get('docs', [])
                date_counts = log_data_summary(docs_page, f"SCOUT_COLLECT page {page}")
                
                if not docs_page:
                    break
                    
                # Если на текущей странице все фильмы имеют одну дату - нашли "месторождение"
                if date_counts and len(date_counts) == 1:
                    motherlode_date = list(date_counts.keys())[0]
                    logging.info(f"[SCOUT] Найдено 'месторождение' данных на странице {page} с датой {motherlode_date}!")
                    next_day = datetime.strptime(motherlode_date, '%Y-%m-%d').date() + timedelta(days=1)
                    scout_state["next_scan_start_date"] = next_day.strftime('%Y-%m-%d')
                    collected_movies.extend(docs_page)
                    return collected_movies, requests_used, motherlode_date, []
                
                collected_movies.extend(docs_page)
                page += 1
                
            except Exception as e:
                logging.error(f"[SCOUT_COLLECT] Ошибка при сборе страницы {page} из диапазона {date_range_for_req}: {e}")
                newly_failed_pages.append({"date": start_date_str, "page": page})
                break
                
    except Exception as e:
        logging.error(f"[SCOUT_PROBE] Ошибка при зондировании диапазона {date_range_for_req}: {e}")
        return [], 0, None, []
    
    # Если "месторождение" не найдено, обновляем начальную дату для следующего запуска
    # на основе последнего собранного фильма
    if collected_movies:
        last_movie_date_str = collected_movies[-1].get('updatedAt', '').split('T')[0]
        if last_movie_date_str:
             next_day = datetime.strptime(last_movie_date_str, '%Y-%m-%d').date() + timedelta(days=1)
             scout_state["next_scan_start_date"] = next_day.strftime('%Y-%m-%d')

    return collected_movies, requests_used, None, newly_failed_pages


def main():
    """Главная функция-оркестратор."""
    hf_token = get_env_var("HF_TOKEN")
    api_keys_str = get_env_var("KINOPOISK_API_KEYS")
    max_requests = int(get_env_var("MAX_REQUESTS_PER_RUN", "200"))
    github_event_name = os.getenv("GITHUB_EVENT_NAME", "local")

    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("API ключи не найдены в KINOPOISK_API_KEYS.")
        sys.exit(1)

    current_hour = datetime.utcnow().hour
    api_key = None
    if current_hour < len(api_keys):
        api_key = api_keys[current_hour]
        logging.info(f"Текущий час UTC: {current_hour}. Выбран ключ API #{current_hour}.")
    elif github_event_name == "workflow_dispatch":
        api_key = random.choice(api_keys)
        logging.warning(f"Нет ключа для часа {current_hour}, но запуск ручной. Выбран случайный ключ API.")
    else:
        logging.critical(f"Нет доступного ключа API для текущего часа ({current_hour}) при плановом запуске. Остановка.")
        sys.exit(0)

    api = HfApi(token=hf_token)
    state = get_state(api)
    
    requests_processed = 0
    all_collected_movies = []
    
    try:
        # Сначала обрабатываем сбойные страницы
        remaining_req = max_requests - requests_processed
        retry_movies, req_used, _, state["failed_pages"] = run_retrier(
            state.get("failed_pages", []), api_key, remaining_req
        )
        all_collected_movies.extend(retry_movies)
        requests_processed += req_used

        # Основной цикл
        while requests_processed < max_requests:
            remaining_req = max_requests - requests_processed
            
            # Если активен режим "Шахтёра" - обрабатываем его приоритетно
            if state["miner_state"]["is_drilling"]:
                logging.info("--- Обнаружена активная задача 'Шахтёра'. Возобновление. ---")
                miner_movies, req_used, finished, failed = run_miner(state, api_key, remaining_req)
                all_collected_movies.extend(miner_movies)
                requests_processed += req_used
                state["failed_pages"].extend(failed)
                
                if finished:
                    logging.info(f"--- 'Шахтёр' ЗАВЕРШИЛ работу для даты {state['miner_state']['drilling_date']} ---")
                    state["miner_state"] = {"is_drilling": False, "drilling_date": None, "last_completed_page": 0, "total_pages": 0}
                else:
                    logging.info("--- 'Шахтёр' ПРИОСТАНОВЛЕН (достигнут лимит запросов). ---")

                update_state(api, state)
                continue

            # Если "Шахтёр" не активен - запускаем "Разведчика"
            logging.info("--- 'Шахтёр' неактивен. Запуск 'Разведчика'. ---")
            scout_movies, req_used, motherlode_date, failed = run_scout(state, api_key, remaining_req)
            all_collected_movies.extend(scout_movies)
            requests_processed += req_used
            state["failed_pages"].extend(failed)

            if motherlode_date:
                logging.info(f"--- 'Разведчик' передал задачу 'Шахтёру' для даты {motherlode_date} ---")
                state["miner_state"] = {"is_drilling": True, "drilling_date": motherlode_date, "last_completed_page": 0, "total_pages": 0}
            
            if req_used == 0 or not motherlode_date:
                 logging.info("--- 'Разведчик' завершил свой цикл на этот запуск. ---")
                 break

    except (RateLimitException, KeyboardInterrupt) as e:
        logging.warning(f"Работа прервана: {e}. Сохранение текущего состояния.")
    except Exception as e:
        logging.critical(f"Непредвиденная критическая ошибка: {e}", exc_info=True)
    finally:
        save_data_chunk(api, all_collected_movies)
        update_state(api, state)
        logging.info(f"\nЗапуск завершен. Всего использовано запросов: {requests_processed}.")

if __name__ == "__main__":
    main()
