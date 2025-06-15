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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- КОНФИГУРАЦИЯ ---
# (Конфигурация остается без изменений)
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
STATE_FILENAME = "collector_state.json"
API_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie"
REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_START_DATE = "2000-01-01"
MAX_MINER_WORKERS = 15
# ИЗМЕНЕНИЕ: Максимальное количество одновременных запросов для повторных попыток
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
    """
    Скачивает файл состояния из репозитория Hugging Face.
    ИЗМЕНЕНИЕ: Теперь также загружает список сбойных страниц.
    """
    default_state = {
        "scout_state": {"next_scan_start_date": DEFAULT_START_DATE},
        "miner_state": {"is_drilling": False, "drilling_date": None, "last_completed_page": 0, "total_pages": 0},
        "failed_pages": [] # НОВОЕ: список для сбойных страниц
    }
    try:
        logging.info(f"Попытка скачать файл состояния: {STATE_FILENAME}")
        state_path = hf_hub_download(
            repo_id=DATASET_ID, filename=STATE_FILENAME, repo_type="dataset",
            force_download=True, resume_download=False
        )
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        # Убедимся, что все ключи на месте
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
    """
    Сохраняет объект состояния в локальный файл и загружает его в репозиторий.
    ИЗМЕНЕНИЕ: Удаляет дубликаты из `failed_pages` перед сохранением.
    """
    with STATE_FILE_LOCK:
        # НОВОЕ: Дедупликация списка сбойных страниц перед сохранением
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

def fetch_page(api_key: str, date_for_req: str, page: int, req_type: str = "REQUEST"):
    """
    Универсальная функция для запросов к API.
    Принимает дату в формате ДД.ММ.ГГГГ для параметра 'updatedAt'.
    """
    params = {'page': page, 'limit': 250, 'sortField': 'updatedAt', 'sortType': '1', 'updatedAt': date_for_req, 'selectFields': ''}
    headers = {"X-API-KEY": api_key}
    logging.info(f"[{req_type}] Запрос: страница {page}, дата {date_for_req}")
    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 403:
        raise RateLimitException("Лимит запросов исчерпан (403 Forbidden).")
    response.raise_for_status()
    return response.json()


# --- РЕЖИМЫ РАБОТЫ ---

def run_retrier(failed_pages_list: list, api_key: str, requests_limit: int):
    """
    НОВАЯ ФУНКЦИЯ: "Восстановитель".
    Пытается параллельно скачать страницы из списка сбоев.
    """
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
            # Форматируем дату для запроса 'ДД.ММ.ГГГГ'
            date_obj = datetime.strptime(task['date'], '%Y-%m-%d').date()
            date_str_for_req = date_obj.strftime('%d.%m.%Y')
            future = executor.submit(fetch_page, api_key, date_str_for_req, task['page'], "RETRIER")
            future_to_task[future] = task
            requests_used += 1

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if data and data.get('docs'):
                    collected_movies.extend(data['docs'])
                # Даже если страница пуста, считаем попытку успешной
                successfully_retried_tasks.append(task)
                logging.info(f"[RETRIER SUCCESS] Страница {task['page']} для даты {task['date']} успешно скачана.")
            except Exception as e:
                logging.error(f"[RETRIER FAIL] Страница {task['page']} для {task['date']} снова не удалась: {e}")
                still_failing_tasks.append(task)
    
    # Создаем новый список сбойных страниц: те, что не удалось повторить + те, что не пытались
    remaining_tasks = [t for t in failed_pages_list if t not in tasks_to_retry]
    final_failed_list = still_failing_tasks + remaining_tasks

    return collected_movies, requests_used, successfully_retried_tasks, final_failed_list


def run_miner(state: dict, api_key: str, requests_limit: int):
    """
    РЕЖИМ "ШАХТЁРА".
    ИЗМЕНЕНИЕ: Теперь возвращает список страниц, которые не удалось скачать.
    """
    miner_state = state["miner_state"]
    drilling_date_str = miner_state["drilling_date"]
    start_page = miner_state["last_completed_page"] + 1
    total_pages = miner_state["total_pages"]
    
    requests_used = 0
    collected_movies = []
    newly_failed_pages = [] # Список для сбоев в этом сеансе

    drilling_date_obj = datetime.strptime(drilling_date_str, '%Y-%m-%d').date()
    date_for_req = drilling_date_obj.strftime('%d.%m.%Y')

    # Шаг 1: Контрольный запрос.
    if total_pages == 0:
        if requests_used >= requests_limit: return collected_movies, requests_used, False, newly_failed_pages
        logging.info(f"[MINER] Контрольный запрос для даты {date_for_req}, чтобы узнать общее кол-во страниц.")
        data = fetch_page(api_key, date_for_req, 2, "MINER_CONTROL")
        requests_used += 1
        total_pages = data.get('pages', start_page)
        miner_state["total_pages"] = total_pages
        logging.info(f"[MINER] В дне {drilling_date_str} найдено {total_pages} страниц.")

    # Шаг 2: Формируем список страниц.
    pages_to_drill = list(range(start_page, total_pages + 1))
    if not pages_to_drill:
        return [], 0, True, []

    # Шаг 3: Параллельная загрузка.
    with ThreadPoolExecutor(max_workers=MAX_MINER_WORKERS) as executor:
        future_to_page = {}
        for page in pages_to_drill:
            if requests_used >= requests_limit:
                logging.warning("[MINER] Достигнут лимит запросов, приостановка бурения.")
                break
            
            future = executor.submit(fetch_page, api_key, date_for_req, page, "MINER_WORKER")
            future_to_page[future] = page
            requests_used += 1

        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                data = future.result()
                if data and data.get('docs'):
                    collected_movies.extend(data['docs'])
                # Обновляем последнюю завершенную страницу, только если она больше текущей
                miner_state["last_completed_page"] = max(miner_state.get("last_completed_page", 0), page)
            except Exception as e:
                # ИЗМЕНЕНИЕ: Регистрируем сбойную страницу
                logging.error(f"[MINER_WORKER] Ошибка при скачивании страницы {page} для {drilling_date_str}: {e}")
                newly_failed_pages.append({"date": drilling_date_str, "page": page})
    
    is_day_finished = miner_state["last_completed_page"] >= total_pages
    return collected_movies, requests_used, is_day_finished, newly_failed_pages


def run_scout(state: dict, api_key: str, requests_limit: int):
    """
    РЕЖИM "РАЗВЕДЧИКА".
    ИЗМЕНЕНИЕ: Теперь возвращает список страниц, которые не удалось скачать.
    """
    if requests_limit <= 0: return [], 0, None, []

    scout_state = state["scout_state"]
    start_date_str = scout_state["next_scan_start_date"]
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date_str = (start_date + timedelta(days=365*5)).strftime('%d.%m.%Y')
    
    date_range_for_req = f"{start_date.strftime('%d.%m.%Y')}-{end_date_str}"
    
    requests_used = 0
    collected_movies = []
    newly_failed_pages = []
    
    # Шаг 1: Зондирование.
    logging.info(f"[SCOUT] Фаза 1: Зондирование диапазона {date_range_for_req}")
    try:
        data = fetch_page(api_key, date_range_for_req, 1, "SCOUT_PROBE")
        requests_used += 1
    except Exception as e:
        logging.error(f"[SCOUT_PROBE] Ошибка при зондировании диапазона {date_range_for_req}: {e}")
        # Если даже первый запрос не удался, нет смысла продолжать
        return [], 0, None, []
        
    docs = data.get('docs', [])
    pages = data.get('pages', 0)

    # Случай А: Пустой диапазон.
    if pages <= 1 and not docs:
        logging.info("[SCOUT] Диапазон пуст. Пропускаем вперед.")
        next_day = datetime.strptime(end_date_str, '%d.%m.%Y').date() + timedelta(days=1)
        scout_state["next_scan_start_date"] = next_day.strftime('%Y-%m-%d')
        return [], requests_used, None, []

    dates_on_page = {d.get('updatedAt', '').split('T')[0] for d in docs if d.get('updatedAt')}
    is_single_date = len(dates_on_page) == 1

    # Случай В: Найдено "месторождение".
    if is_single_date:
        motherlode_date = list(dates_on_page)[0]
        logging.info(f"[SCOUT] Найдено 'месторождение' данных на дату {motherlode_date}!")
        next_day = datetime.strptime(motherlode_date, '%Y-%m-%d').date() + timedelta(days=1)
        scout_state["next_scan_start_date"] = next_day.strftime('%Y-%m-%d')
        return docs, requests_used, motherlode_date, []

    # Случай Б: Разрозненные данные.
    logging.info("[SCOUT] Найдены разрозненные данные. Последовательный сбор...")
    collected_movies.extend(docs)
    page = 2
    while page <= pages and requests_used < requests_limit:
        try:
            data = fetch_page(api_key, date_range_for_req, page, "SCOUT_COLLECT")
            requests_used += 1
            if not data or not data.get('docs'):
                break
            collected_movies.extend(data['docs'])
            page += 1
        except Exception as e:
            logging.error(f"[SCOUT_COLLECT] Ошибка при сборе страницы {page} из диапазона {date_range_for_req}: {e}")
            # Мы не знаем точную дату сбоя, поэтому не можем добавить в failed_pages.
            # Просто прерываем сбор этого диапазона.
            break
    
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
    # ИЗМЕНЕНИЕ: Получаем имя события для выбора ключа
    github_event_name = os.getenv("GITHUB_EVENT_NAME", "local")

    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not api_keys:
        logging.critical("API ключи не найдены в KINOPOISK_API_KEYS.")
        sys.exit(1)

    # --- ИЗМЕНЕНИЕ: Обновленная логика выбора API ключа ---
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
        sys.exit(0) # Выходим с кодом 0, чтобы не считать это ошибкой воркфлоу
    # --- Конец изменения ---

    api = HfApi(token=hf_token)
    state = get_state(api)
    
    requests_processed = 0
    all_collected_movies = []
    
    try:
        # --- НОВАЯ ФАЗА: Повторные попытки для сбойных страниц ---
        remaining_req = max_requests - requests_processed
        retry_movies, req_used, _, state["failed_pages"] = run_retrier(
            state.get("failed_pages", []), api_key, remaining_req
        )
        all_collected_movies.extend(retry_movies)
        requests_processed += req_used
        # --- Конец новой фазы ---

        while requests_processed < max_requests:
            remaining_req = max_requests - requests_processed
            
            # Шаг 1: Приоритет "Шахтёра".
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

            # Шаг 2: Запуск "Разведчика".
            logging.info("--- 'Шахтёр' неактивен. Запуск 'Разведчика'. ---")
            scout_movies, req_used, motherlode_date, failed = run_scout(state, api_key, remaining_req)
            all_collected_movies.extend(scout_movies)
            requests_processed += req_used
            state["failed_pages"].extend(failed)

            if motherlode_date:
                logging.info(f"--- 'Разведчик' передал задачу 'Шахтёру' для даты {motherlode_date} ---")
                state["miner_state"] = {"is_drilling": True, "drilling_date": motherlode_date, "last_completed_page": 1, "total_pages": 0}
            
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

