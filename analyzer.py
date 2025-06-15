import os
import gzip
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# ИСПРАВЛЕНО: Правильный импорт модуля с классами ошибок
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# --- Конфигурация ---
# Обязательно укажите ваше имя пользователя на Hugging Face
HF_USERNAME = "opex792" 
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
CONSOLIDATED_DIR = "consolidated"
CONSOLIDATED_FILENAME = "kinopoisk.jsonl.gz"
# Имя файла для сохранения отчета теперь более описательное
OUTPUT_FILENAME = "analysis_report.txt"

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
        [span_0](start_span)logging.critical(f"Error: Environment variable {var_name} is not set.")[span_0](end_span)
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
            [span_1](start_span)repo_id=DATASET_ID, filename=consolidated_repo_path, repo_type="dataset"[span_1](end_span)
        )
        logging.info(f"Successfully downloaded to {local_consolidated_path}")
    except EntryNotFoundError:
        [span_2](start_span)logging.critical("Main consolidated file not found. Nothing to analyze. Exiting.")[span_2](end_span)
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred when downloading the main file: {e}", exc_info=True)
        sys.exit(1)

    # --- Структуры для сбора расширенной статистики ---
    total_entries = 0
    entries_with_errors = 0
    date_counter = Counter()
    type_counter = Counter()
    year_counter = Counter()
    genre_counter = Counter()
    country_counter = Counter()
    # defaultdict для удобного группирования рейтинга по целым числам
    rating_kp_counter = defaultdict(int)
    missing_data_counter = Counter()
    first_update = None
    last_update = None

    logging.info("Starting to read the dataset and gather detailed statistics...")

    try:
        with gzip.open(local_consolidated_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_entries += 1
                try:
                    [span_3](start_span)movie = json.loads(line)[span_3](end_span)

                    # 1. Анализ по дате обновления (updatedAt)
                    updated_at_str = movie.get("updatedAt")
                    if updated_at_str:
                        [span_4](start_span)dt_object = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))[span_4](end_span)
                        date_key = dt_object.strftime('%Y-%m-%d')
                        date_counter[date_key] += 1
                        if first_update is None or dt_object < first_update:
                            first_update = dt_object
                        if last_update is None or dt_object > last_update:
                            last_update = dt_object
                    else:
                        missing_data_counter['updatedAt'] += 1

                    # 2. Анализ по типу (type)
                    movie_type = movie.get('type')
                    if movie_type:
                        type_counter[movie_type] += 1
                    else:
                        missing_data_counter['type'] += 1

                    # 3. Анализ по году (year)
                    year = movie.get('year')
                    if year:
                        year_counter[year] += 1
                    else:
                        missing_data_counter['year'] += 1

                    # 4. Анализ по жанрам (genres)
                    genres = movie.get('genres')
                    if genres and isinstance(genres, list):
                        for genre_item in genres:
                            if isinstance(genre_item, dict) and 'name' in genre_item:
                                 genre_counter[genre_item['name']] += 1
                    else:
                        missing_data_counter['genres'] += 1

                    # 5. Анализ по странам (countries)
                    countries = movie.get('countries')
                    if countries and isinstance(countries, list):
                        for country_item in countries:
                             if isinstance(country_item, dict) and 'name' in country_item:
                                 country_counter[country_item['name']] += 1
                    else:
                        missing_data_counter['countries'] += 1
                    
                    # 6. Анализ по рейтингу Кинопоиска (rating.kp)
                    rating = movie.get('rating', {})
                    kp_rating_val = rating.get('kp') if isinstance(rating, dict) else None
                    if kp_rating_val is not None:
                        # Группируем рейтинг по диапазонам: 8.0-8.9, 7.0-7.9 и т.д.
                        rating_bin = f"{int(kp_rating_val)}.0-{int(kp_rating_val)}.9"
                        rating_kp_counter[rating_bin] += 1
                    else:
                        missing_data_counter['rating.kp'] += 1

                [span_5](start_span)except (json.JSONDecodeError, ValueError) as e:[span_5](end_span)
                    [span_6](start_span)logging.warning(f"Could not process line {i+1}. Error: {e}. Skipping line.")[span_6](end_span)
                    entries_with_errors += 1

    except Exception as e:
        logging.critical(f"Failed to read or process the gzipped file: {e}", exc_info=True)
        sys.exit(1)

    # --- Сохранение подробного отчета ---
    logging.info(f"Writing detailed analysis to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("          Аналитический отчет по датасету Kinopoisk\n")
        f.write("="*60 + "\n")
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Отчет сгенерирован: {report_time}\n\n")

        # --- Общая статистика ---
        f.write("--- Общая статистика ---\n")
        f.write(f"Всего записей в файле: {total_entries}\n")
        if first_update and last_update:
            f.write(f"Период обновлений: с {first_update.strftime('%Y-%m-%d')} по {last_update.strftime('%Y-%m-%d')}\n")
        f.write(f"Строк с ошибками парсинга JSON: {entries_with_errors}\n\n")

        # --- Статистика по отсутствующим данным ---
        f.write("--- Статистика по отсутствующим ключевым полям ---\n")
        if not missing_data_counter:
            f.write("Все ключевые поля присутствуют во всех проанализированных записях.\n")
        else:
            for field, count in sorted(missing_data_counter.items()):
                f.write(f"Записей без поля '{field}': {count}\n")
        f.write("\n")

        # --- Распределение по типам ---
        f.write("--- Распределение по типам (топ 20) ---\n")
        if not type_counter:
            f.write("Данные по типам отсутствуют.\n")
        else:
            for item, count in type_counter.most_common(20):
                f.write(f"{item}: {count}\n")
        f.write("\n")

        # --- Распределение по годам выпуска ---
        f.write("--- Распределение по годам выпуска (топ 30 свежих) ---\n")
        if not year_counter:
            f.write("Данные по годам выпуска отсутствуют.\n")
        else:
            sorted_years = sorted(year_counter.items(), key=lambda x: x[0], reverse=True)
            for item, count in sorted_years[:30]:
                f.write(f"{item} год: {count}\n")
        f.write("\n")

        # --- Распределение по жанрам ---
        f.write("--- Распределение по жанрам (топ 30) ---\n")
        if not genre_counter:
            f.write("Данные по жанрам отсутствуют.\n")
        else:
            for item, count in genre_counter.most_common(30):
                f.write(f"{item}: {count}\n")
        f.write("\n")

        # --- Распределение по странам ---
        f.write("--- Распределение по странам (топ 30) ---\n")
        if not country_counter:
            f.write("Данные по странам отсутствуют.\n")
        else:
            for item, count in country_counter.most_common(30):
                f.write(f"{item}: {count}\n")
        f.write("\n")
        
        # --- Распределение по рейтингу Кинопоиска ---
        f.write("--- Распределение по рейтингу Кинопоиска ---\n")
        if not rating_kp_counter:
            f.write("Данные по рейтингу Кинопоиска отсутствуют.\n")
        else:
            sorted_ratings = sorted(rating_kp_counter.items(), key=lambda x: x[0], reverse=True)
            for item, count in sorted_ratings:
                f.write(f"Рейтинг в диапазоне {item}: {count}\n")
        f.write("\n")

        # --- Статистика обновлений по дням ---
        [span_7](start_span)f.write("--- Статистика обновлений по дням (все дни) ---\n")[span_7](end_span)
        if not date_counter:
            f.write("Даты обновлений не найдены.\n")
        else:
            sorted_dates = sorted(date_counter.items())
            for date, count in sorted_dates:
                f.write(f"{date}: {count}\n")
        f.write("\n")

    logging.info(f"Analysis complete. Report saved to {OUTPUT_FILENAME}.")


if __name__ == "__main__":
    main()
            
