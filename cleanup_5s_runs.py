import os
import sqlite3
import sys

# Add current directory to sys.path so we can import hudes
sys.path.append(os.getcwd())

from hudes.high_scores import DB_DEFAULT_PATH


def cleanup():
    path = os.environ.get("HIGH_SCORES_PATH", DB_DEFAULT_PATH)
    print(f"Opening DB at {path}")

    if not os.path.exists(path):
        print("Database file not found!")
        return

    try:
        with sqlite3.connect(path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM high_scores WHERE duration = 5")
            count = cur.fetchone()[0]
            print(f"Found {count} runs with duration 5s")

            if count > 0:
                conn.execute("DELETE FROM high_scores WHERE duration = 5")
                conn.commit()
                print(f"Successfully deleted {count} records.")
            else:
                print("No records found to delete.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    cleanup()
