import os
import gzip
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

# ИСПРАВЛЕНО: Правильный импорт модуля с классами ошибок
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# --- Конфигурация ---
# Обязательно укажите ваше имя пользователя на Hugging Face
HF_USERNAME = "opex792" 
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
CONSOLIDATED_DIR = "consolidated"
CONSOLIDATED_FILENAME = "kinopoisk.jsonl.gz"
OUTPUT_FILENAME = "update_stats.txt"

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_required_env_var(var_name):
    """Получает переменную окружения или завершает работу, если она не установлена."""
    value = os.getenv(var_name)
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def main():
    """Основная функция для анализа датасета."""
    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi(token=hf_token)

    consolidated_repo_path = f"{CONSOLIDATED_DIR}/{CONSOLIDATED_FILENAME}"
    local_consolidated_path = None

    # --- Скачивание основного файла ---
    try:
        logging.info(f"Attempting to download main consolidated file: {consolidated_repo_path}...")
        local_consolidated_path = hf_hub_download(
            repo_id=DATASET_ID, filename=consolidated_repo_path, repo_type="dataset"
        )
        logging.info(f"Successfully downloaded to {local_consolidated_path}")
    except EntryNotFoundError:
        logging.critical("Main consolidated file not found. Nothing to analyze. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred when downloading the main file: {e}", exc_info=True)
        sys.exit(1)

    # --- Подсчет статистики ---
    date_counter = Counter()
    logging.info("Starting to read the dataset and count updatedAt dates...")

    try:
        with gzip.open(local_consolidated_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    movie = json.loads(line)
                    updated_at_str = movie.get("updatedAt")

                    if updated_at_str:
                        # Преобразуем ISO строку в объект datetime, а затем в дату YYYY-MM-DD
                        dt_object = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                        date_key = dt_object.strftime('%Y-%m-%d')
                        date_counter[date_key] += 1
                
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"Could not process line {i+1}. Error: {e}. Skipping line.")

    except Exception as e:
        logging.critical(f"Failed to read or process the gzipped file: {e}", exc_info=True)
        sys.exit(1)
        
    if not date_counter:
        logging.warning("No dates found in the dataset. The resulting file will be empty.")

    # --- Сохранение результата ---
    # Сортируем даты для наглядности
    sorted_dates = sorted(date_counter.items())

    logging.info(f"Writing statistics to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("Статистика обновлений по дням:\n")
        f.write("===============================\n")
        for date, count in sorted_dates:
            # Не выводим дни с нулевым количеством, Counter и так их не добавит
            f.write(f"{date}: {count}\n")

    logging.info(f"Analysis complete. Results saved to {OUTPUT_FILENAME}.")


if __name__ == "__main__":
    main()

