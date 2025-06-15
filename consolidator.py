import os
import gzip
import json
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfFolder
from huggingface_hub.errors import HfHubHTTPError

# --- Конфигурация ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
CONSOLIDATED_DIR = "consolidated"
CONSOLIDATED_FILENAME = "kinopoisk.jsonl.gz"
DB_FILENAME = "consolidation_db.sqlite"
BATCH_SIZE = 5000

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_required_env_var(var_name):
    """Получает переменную окружения или токен из Hugging Face."""
    value = os.getenv(var_name)
    if not value:
        value = HfFolder.get_token()
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def parse_iso_date_to_timestamp(date_string):
    """Парсит дату ISO в Unix timestamp."""
    if not date_string: return 0
    try:
        return int(datetime.fromisoformat(date_string.replace('Z', '+00:00')).timestamp())
    except (ValueError, TypeError): return 0

def setup_database(db_path):
    """Создает и настраивает временную базу данных SQLite."""
    if os.path.exists(db_path):
        os.remove(db_path)
    logging.info(f"Setting up temporary database at {db_path}...")
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.executescript('''
    CREATE TABLE movies (
        id INTEGER PRIMARY KEY,
        updated_at_ts INTEGER NOT NULL,
        data TEXT NOT NULL
    );
    CREATE INDEX idx_updated_at ON movies(updated_at_ts);
    ''')
    db_conn.commit()
    return db_conn

def process_file_into_db(file_path, db_conn, file_type="File"):
    """Читает файл .jsonl.gz и вставляет/обновляет данные в БД, избегая перезаписи более свежих данных старыми."""
    logging.info(f"Processing {file_type} into database: {Path(file_path).name}...")
    cursor = db_conn.cursor()

    sql_upsert = """
    INSERT INTO movies (id, updated_at_ts, data)
    VALUES (?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        updated_at_ts = excluded.updated_at_ts,
        data = excluded.data
    WHERE excluded.updated_at_ts > movies.updated_at_ts;
    """

    batch = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                movie = json.loads(line)
                movie_id = movie.get("id")
                if not movie_id: continue
                
                updated_at_str = movie.get("updatedAt")
                updated_at_ts = parse_iso_date_to_timestamp(updated_at_str)
                movie_data_json = json.dumps(movie, ensure_ascii=False)
                
                batch.append((movie_id, updated_at_ts, movie_data_json))

                if len(batch) >= BATCH_SIZE:
                    cursor.executemany(sql_upsert, batch)
                    db_conn.commit()
                    batch = []

            except json.JSONDecodeError:
                logging.warning(f"JSON decode error in {Path(file_path).name}. Skipping line.")

    if batch:
        cursor.executemany(sql_upsert, batch)
        db_conn.commit()

def main():
    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi(token=hf_token)
    db_conn = setup_database(DB_FILENAME)
    
    consolidated_repo_path = f"{CONSOLIDATED_DIR}/{CONSOLIDATED_FILENAME}"
    try:
        logging.info(f"Attempting to download main consolidated file: {consolidated_repo_path}...")
        local_consolidated_path = hf_hub_download(
            repo_id=DATASET_ID, filename=consolidated_repo_path, repo_type="dataset", token=hf_token
        )
        process_file_into_db(local_consolidated_path, db_conn, file_type="Main")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logging.warning("Main consolidated file not found. A new one will be created.")
        else:
            logging.error(f"HTTP error downloading main file: {e}", exc_info=True)
            db_conn.close()
            sys.exit(1)
    
    try:
        raw_files_in_repo = [f for f in api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset") if f.startswith(f"{RAW_DATA_DIR}/")]
    except Exception as e:
        logging.critical(f"Could not list repo files. Error: {e}", exc_info=True)
        db_conn.close()
        sys.exit(1)
    
    if raw_files_in_repo:
        logging.info(f"Found {len(raw_files_in_repo)} raw files to consolidate.")
        for file_path in raw_files_in_repo:
            try:
                local_raw_path = hf_hub_download(repo_id=DATASET_ID, filename=file_path, repo_type="dataset", token=hf_token)
                process_file_into_db(local_raw_path, db_conn, file_type="Raw")
            except Exception as e:
                logging.error(f"Failed to process raw file {file_path}: {e}", exc_info=True)
    
    output_dir = Path("output_consolidated")
    output_dir.mkdir(exist_ok=True)
    final_archive_path = output_dir / CONSOLIDATED_FILENAME
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(id) FROM movies")
    total_unique_movies = cursor.fetchone()[0]
    
    if total_unique_movies == 0:
        logging.warning("Database is empty. Nothing to upload. Exiting.")
        db_conn.close()
        if os.path.exists(DB_FILENAME):
            os.remove(DB_FILENAME)
        sys.exit(0)

    logging.info(f"Writing {total_unique_movies} unique movies to new archive: {final_archive_path}...")
    cursor.execute("SELECT data FROM movies ORDER BY id ASC")
    with gzip.open(final_archive_path, 'wt', encoding='utf-8') as f:
        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows: break
            for row in rows: f.write(row[0] + '\n')
    db_conn.close()

    try:
        logging.info(f"Uploading new consolidated file to {consolidated_repo_path}...")
        api.upload_file(
            path_or_fileobj=str(final_archive_path),
            path_in_repo=consolidated_repo_path,
            repo_id=DATASET_ID, repo_type="dataset"
        )
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to upload new consolidated file: {e}", exc_info=True)
        sys.exit(1)

    if raw_files_in_repo:
        logging.info(f"Deleting {len(raw_files_in_repo)} processed raw files...")
        try:
            api.delete_files(repo_id=DATASET_ID, repo_type="dataset", paths_in_repo=raw_files_in_repo)
            logging.info("Successfully deleted raw files.")
        except Exception as e:
            logging.error(f"Could not delete raw files. Please delete them manually. Error: {e}")
    
    if os.path.exists(DB_FILENAME):
        os.remove(DB_FILENAME)
    logging.info("\nConsolidation run completed successfully.")

if __name__ == "__main__":
    main()


