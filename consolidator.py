import os
import gzip
import json
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, delete_file

# --- Configuration ---
HF_USERNAME = "opex792"
DATASET_ID = f"{HF_USERNAME}/kinopoisk"
RAW_DATA_DIR = "raw_data"
CONSOLIDATED_DIR = "consolidated"
CONSOLIDATED_FILENAME = "kinopoisk.jsonl.gz"
DB_FILENAME = "consolidation_db.sqlite"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_required_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        logging.critical(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)
    return value

def parse_iso_date_to_timestamp(date_string):
    if not date_string: return 0
    try:
        return int(datetime.fromisoformat(date_string.replace('Z', '+00:00')).timestamp())
    except (ValueError, TypeError): return 0

def setup_database(db_path):
    """Creates and configures the SQLite database."""
    if os.path.exists(db_path):
        os.remove(db_path)
    logging.info(f"Setting up temporary database at {db_path}...")
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute('''
    CREATE TABLE movies (
        id INTEGER PRIMARY KEY,
        updated_at_ts INTEGER NOT NULL,
        data TEXT NOT NULL
    )''')
    db_conn.commit()
    return db_conn

def process_file_into_db(file_path, db_cursor, file_type="File"):
    """Reads a .jsonl.gz file and inserts/updates data in the DB."""
    logging.info(f"Processing {file_type} into database: {Path(file_path).name}...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                movie = json.loads(line)
                movie_id = movie.get("id")
                updated_at_str = movie.get("updatedAt")
                if not movie_id or not updated_at_str: continue

                updated_at_ts = parse_iso_date_to_timestamp(updated_at_str)
                movie_data_json = json.dumps(movie, ensure_ascii=False)
                # INSERT OR REPLACE is an atomic operation based on the PRIMARY KEY
                db_cursor.execute(
                    "INSERT OR REPLACE INTO movies (id, updated_at_ts, data) VALUES (?, ?, ?)",
                    (movie_id, updated_at_ts, movie_data_json)
                )
            except json.JSONDecodeError:
                logging.warning(f"JSON decode error in {Path(file_path).name}. Skipping line.")

def main():
    hf_token = get_required_env_var("HF_TOKEN")
    api = HfApi(token=hf_token)
    db_conn = setup_database(DB_FILENAME)
    db_cursor = db_conn.cursor()
    
    # 1. Download and process the main consolidated file (if it exists)
    consolidated_repo_path = f"{CONSOLIDATED_DIR}/{CONSOLIDATED_FILENAME}"
    try:
        local_consolidated_path = api.hf_hub_download(
            repo_id=DATASET_ID, filename=consolidated_repo_path, repo_type="dataset"
        )
        process_file_into_db(local_consolidated_path, db_cursor, file_type="Main")
        db_conn.commit()
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logging.warning("Main consolidated file not found. A new one will be created.")
        else: raise
    
    # 2. Download and process all raw chunk files
    try:
        raw_files_in_repo = [f for f in api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset") if f.startswith(f"{RAW_DATA_DIR}/")]
    except Exception as e:
        logging.critical(f"Could not list repo files. Error: {e}", exc_info=True)
        sys.exit(1)

    if not raw_files_in_repo:
        logging.info("No new raw files to process. Consolidation not needed. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(raw_files_in_repo)} raw files to consolidate.")
    for file_path in raw_files_in_repo:
        local_raw_path = api.hf_hub_download(repo_id=DATASET_ID, filename=file_path, repo_type="dataset")
        process_file_into_db(local_raw_path, db_cursor, file_type="Raw")
        db_conn.commit()

    # 3. Save the merged database to a new consolidated file
    output_dir = Path("output_consolidated")
    output_dir.mkdir(exist_ok=True)
    final_archive_path = output_dir / CONSOLIDATED_FILENAME
    
    db_cursor.execute("SELECT COUNT(id) FROM movies")
    total_unique_movies = db_cursor.fetchone()[0]
    logging.info(f"Writing {total_unique_movies} unique movies to new archive: {final_archive_path}...")
    
    db_cursor.execute("SELECT data FROM movies ORDER BY id ASC")
    with gzip.open(final_archive_path, 'wt', encoding='utf-8') as f:
        while True:
            rows = db_cursor.fetchmany(5000)
            if not rows: break
            for row in rows: f.write(row[0] + '\n')
    db_conn.close()

    # 4. Upload the new consolidated file
    logging.info(f"Uploading new consolidated file to {consolidated_repo_path}...")
    api.upload_file(
        path_or_fileobj=str(final_archive_path),
        path_in_repo=consolidated_repo_path,
        repo_id=DATASET_ID, repo_type="dataset"
    )

    # 5. Delete the processed raw files
    logging.info("Deleting processed raw files from repository...")
    for file_path in raw_files_in_repo:
        try:
            logging.info(f"  Deleting {file_path}...")
            delete_file(repo_id=DATASET_ID, repo_type="dataset", path_in_repo=file_path, token=hf_token)
        except Exception as e:
            logging.error(f"Could not delete file {file_path}. Please delete it manually. Error: {e}")

    # 6. Cleanup local files
    os.remove(DB_FILENAME)
    logging.info("\nConsolidation run completed successfully.")

if __name__ == "__main__":
    main()


