import os
import gzip
import json
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
CONSOLIDATED_DATA_DIR = "consolidated"
CONSOLIDATED_FILENAME = "consolidated_data.jsonl.gz"
# ИМЯ ВРЕМЕННОЙ БАЗЫ ДАННЫХ
DB_FILENAME = "temp_movie_db.sqlite"

# --- Настройка логирования ---
log_filename = f"log_consolidator_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_required_env_var(var_name):
    """Получает переменную окружения, прекращая работу, если она не установлена."""
    value = os.getenv(var_name)
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def parse_iso_date_to_timestamp(date_string):
    """Преобразует строку ISO в Unix timestamp для удобного сравнения."""
    if not date_string:
        return 0
    try:
        # Преобразуем в aware datetime объект
        dt_object = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return int(dt_object.timestamp())
    except (ValueError, TypeError):
        return 0

def setup_database(db_path):
    """Создает и настраивает базу данных SQLite."""
    logging.info(f"Setting up temporary database at {db_path}...")
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    # Создаем таблицу для хранения фильмов
    # id: уникальный идентификатор фильма
    # updated_at_ts: время обновления в виде timestamp для быстрого сравнения
    # data: полные данные о фильме в формате JSON
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY,
        updated_at_ts INTEGER NOT NULL,
        data TEXT NOT NULL
    )
    ''')
    # Индекс по updatedAt для ускорения поиска
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_updated_at ON movies(updated_at_ts)')
    db_conn.commit()
    logging.info("Database setup complete.")
    return db_conn

def main():
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as f:
            f.write(f"LOG_FILE_PATH={log_filename}\n")

    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi()

    # Инициализация временной базы данных
    db_conn = setup_database(DB_FILENAME)
    db_cursor = db_conn.cursor()

    logging.info("Fetching list of all files from the dataset...")
    try:
        all_repo_files = list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=hf_token
        )
        raw_files = [f for f in all_repo_files if f.startswith(f"{RAW_DATA_DIR}/")]
    except Exception as e:
        logging.critical(f"Could not list repo files. Error: {e}", exc_info=True)
        db_conn.close()
        sys.exit(1)

    if not raw_files:
        logging.warning(f"No files found in the '{RAW_DATA_DIR}' directory. Exiting.")
        db_conn.close()
        sys.exit(0)

    logging.info(f"Found {len(raw_files)} raw files. Starting deduplication using database...")

    total_movies_processed = 0
    for file_path_in_repo in raw_files:
        logging.info(f"  Processing {file_path_in_repo}...")
        try:
            local_file_path = hf_hub_download(
                repo_id=DATASET_ID, filename=file_path_in_repo,
                repo_type="dataset", token=hf_token
            )
            with gzip.open(local_file_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_movies_processed += 1
                    try:
                        movie = json.loads(line)
                        movie_id = movie.get("id")
                        updated_at_str = movie.get("updatedAt")

                        if not movie_id or not updated_at_str:
                            continue

                        updated_at_ts = parse_iso_date_to_timestamp(updated_at_str)
                        movie_data_json = json.dumps(movie, ensure_ascii=False)

                        # Проверяем, есть ли фильм в базе
                        db_cursor.execute("SELECT updated_at_ts FROM movies WHERE id = ?", (movie_id,))
                        result = db_cursor.fetchone()

                        if result is None:
                            # Фильма нет - вставляем
                            db_cursor.execute(
                                "INSERT INTO movies (id, updated_at_ts, data) VALUES (?, ?, ?)",
                                (movie_id, updated_at_ts, movie_data_json)
                            )
                        elif updated_at_ts > result[0]:
                            # Фильм есть, но новый свежее - обновляем
                            db_cursor.execute(
                                "UPDATE movies SET updated_at_ts = ?, data = ? WHERE id = ?",
                                (updated_at_ts, movie_data_json, movie_id)
                            )
                        # Иначе ничего не делаем, в базе более новая версия

                    except json.JSONDecodeError:
                        logging.warning(f"    JSON decode error in {file_path_in_repo} at line {i+1}. Skipping line.")
                        continue
            # Коммитим изменения после каждого файла для надежности
            db_conn.commit()
        except Exception as e:
            logging.error(f"    Could not process file {file_path_in_repo}. Skipping. Error: {e}")

    logging.info(f"\nScanned {total_movies_processed} movie entries across all files.")

    db_cursor.execute("SELECT COUNT(id) FROM movies")
    unique_movies_count = db_cursor.fetchone()[0]
    logging.info(f"Deduplication complete. Unique movies found: {unique_movies_count}")

    output_dir = Path("output_consolidated")
    output_dir.mkdir(exist_ok=True)
    final_archive_path = output_dir / CONSOLIDATED_FILENAME

    logging.info(f"Writing all unique movies to a single archive: {final_archive_path}...")
    # Выбираем все фильмы, отсортированные по ID
    db_cursor.execute("SELECT data FROM movies ORDER BY id ASC")

    with gzip.open(final_archive_path, 'wt', encoding='utf-8') as f:
        while True:
            # Читаем данные из базы порциями для экономии памяти
            rows = db_cursor.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                f.write(row[0] + '\n')

    logging.info("Finished writing to archive.")
    db_cursor.close()
    db_conn.close()

    final_size_gb = final_archive_path.stat().st_size / 1024**3
    logging.info(f"Final archive size: {final_size_gb:.3f} GB")

    logging.info(f"Uploading {final_archive_path.name} to dataset...")
    api.upload_file(
        path_or_fileobj=str(final_archive_path),
        path_in_repo=f"{CONSOLIDATED_DATA_DIR}/{final_archive_path.name}",
        repo_id=DATASET_ID, repo_type="dataset", token=hf_token,
        commit_message=f"Consolidate: {unique_movies_count} unique movies"
    )

    logging.info("\nConsolidation run completed successfully.")

    # Очистка - удаляем временную базу данных
    try:
        os.remove(DB_FILENAME)
        logging.info(f"Successfully removed temporary database: {DB_FILENAME}")
    except OSError as e:
        logging.error(f"Error removing temporary database file: {e}")


if __name__ == "__main__":
    main()


