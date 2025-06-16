import os
import gzip
import json
import sys
import logging
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from huggingface_hub import HfApi, hf_hub_download, create_repo
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
FORCE_FULL_CLEANUP = True  # Установите True для принудительной очистки

# Файлы, которые нужно сохранить при очистке
FILES_TO_PRESERVE = ["README.md", ".gitattributes"]

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

def full_repo_cleanup(api, hf_token, dataset_id, consolidated_file_path):
    """Выполняет полную очистку репозитория, сохраняя только консолидированный файл и важные файлы."""
    logging.info("Performing full repository cleanup to remove history and reduce storage usage...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Создаем новый временный репозиторий
        temp_repo_name = f"{dataset_id}_temp"
        try:
            # Проверяем существует ли временный репозиторий и удаляем его если существует
            try:
                api.repo_info(repo_id=temp_repo_name, repo_type="dataset")
                logging.info(f"Deleting existing temporary repository: {temp_repo_name}")
                api.delete_repo(repo_id=temp_repo_name, repo_type="dataset")
            except:
                pass

            # Создаем новый чистый репозиторий
            logging.info(f"Creating new temporary repository: {temp_repo_name}")
            create_repo(repo_id=temp_repo_name, repo_type="dataset", token=hf_token)

            # Копируем консолидированный файл во временную директорию
            temp_file_path = temp_dir_path / "consolidated" / CONSOLIDATED_FILENAME
            os.makedirs(temp_file_path.parent, exist_ok=True)
            shutil.copy2(consolidated_file_path, temp_file_path)

            # Загружаем файл во временный репозиторий
            logging.info(f"Uploading consolidated file to temporary repository")
            api.upload_file(
                path_or_fileobj=str(temp_file_path),
                path_in_repo=f"consolidated/{CONSOLIDATED_FILENAME}",
                repo_id=temp_repo_name,
                repo_type="dataset"
            )

            # Сохраняем важные файлы (README.md и другие)
            preserved_files = {}
            for file_to_preserve in FILES_TO_PRESERVE:
                try:
                    logging.info(f"Attempting to preserve {file_to_preserve}")
                    local_path = hf_hub_download(
                        repo_id=dataset_id, 
                        filename=file_to_preserve, 
                        repo_type="dataset", 
                        token=hf_token
                    )
                    with open(local_path, 'rb') as f:
                        preserved_files[file_to_preserve] = f.read()

                    # Загружаем важный файл во временный репозиторий
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=file_to_preserve,
                        repo_id=temp_repo_name,
                        repo_type="dataset"
                    )
                    logging.info(f"Successfully preserved {file_to_preserve}")
                except Exception as e:
                    logging.warning(f"Could not preserve {file_to_preserve}: {e}")

            # Удаляем оригинальный репозиторий
            logging.info(f"Deleting original repository: {dataset_id}")
            api.delete_repo(repo_id=dataset_id, repo_type="dataset")

            # Создаем новый чистый репозиторий с оригинальным именем
            logging.info(f"Creating new clean repository: {dataset_id}")
            create_repo(repo_id=dataset_id, repo_type="dataset", token=hf_token)

            # Загружаем консолидированный файл в новый репозиторий
            logging.info(f"Uploading consolidated file to new clean repository")
            api.upload_file(
                path_or_fileobj=str(temp_file_path),
                path_in_repo=f"consolidated/{CONSOLIDATED_FILENAME}",
                repo_id=dataset_id,
                repo_type="dataset"
            )

            # Восстанавливаем важные файлы в новый репозиторий
            for file_name, content in preserved_files.items():
                temp_file = temp_dir_path / file_name
                with open(temp_file, 'wb') as f:
                    f.write(content)

                logging.info(f"Restoring {file_name} to new repository")
                api.upload_file(
                    path_or_fileobj=str(temp_file),
                    path_in_repo=file_name,
                    repo_id=dataset_id,
                    repo_type="dataset"
                )

            # Удаляем временный репозиторий
            logging.info(f"Deleting temporary repository: {temp_repo_name}")
            api.delete_repo(repo_id=temp_repo_name, repo_type="dataset")

            logging.info("Full repository cleanup completed successfully")
            return True

        except Exception as e:
            logging.error(f"Error during full repository cleanup: {e}", exc_info=True)
            return False

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

    # Если есть raw-файлы или установлен флаг принудительной очистки,
    # выполняем полную очистку репозитория
    if raw_files_in_repo or FORCE_FULL_CLEANUP:
        if full_repo_cleanup(api, hf_token, DATASET_ID, final_archive_path):
            logging.info("Repository has been completely cleaned up with history removal.")
        else:
            # Если полная очистка не удалась, пробуем обычный метод загрузки и удаления
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
                    api.delete_files(repo_id=DATASET_ID, repo_type="dataset", path_or_paths=raw_files_in_repo)
                    logging.info("Successfully deleted raw files.")
                except Exception as e:
                    logging.error(f"Could not delete raw files. Please delete them manually. Error: {e}")
    else:
        # Обычный режим загрузки без полной очистки
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

    if os.path.exists(DB_FILENAME):
        os.remove(DB_FILENAME)
    logging.info("\nConsolidation run completed successfully.")

if __name__ == "__main__":
    main()
