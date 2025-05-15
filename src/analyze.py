import json
import sqlite3

from config import Config


def equivalence_classes():
    logger = Config.logger
    logger.info("Starting database analysis.")
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        raw_table_names = [row[0] for row in cursor.fetchall()]
        # Filter out sqlite_sequence and other internal sqlite tables
        table_names = [name for name in raw_table_names if not name.startswith("sqlite_")]

        if not table_names:
            logger.info("No user tables found in the database to analyze.")
            return

        for table_name in table_names:
            try:
                cursor.execute(f'SELECT date, top_tokens, logprobs FROM "{table_name}"')
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                logger.error(f"Could not query table {table_name}: {e}")
                continue

            if not rows:
                continue

            equivalence_classes = {}  # (top_tokens_tuple, logprobs_tuple) -> [dates]

            for date_str, top_tokens_json, logprobs_json in rows:
                if not top_tokens_json or not logprobs_json:
                    logger.warning(
                        f"Skipping row with empty JSON data in {table_name} for date {date_str}"
                    )
                    continue
                # Convert JSON strings to tuples to be hashable dictionary keys
                top_tokens = tuple(json.loads(top_tokens_json))
                logprobs = tuple(json.loads(logprobs_json))

                key = (top_tokens, logprobs)

                if key not in equivalence_classes:
                    equivalence_classes[key] = []
                equivalence_classes[key].append(date_str)

            equivalence_classes = dict(
                sorted(equivalence_classes.items(), key=lambda x: len(x[1]), reverse=True)
            )

            print(f"\n# {table_name}")
            for (_, logprobs_key), dates in equivalence_classes.items():
                logprobs_display = list(logprobs_key)[:5] + ["..."]
                num_instances = len(dates)
                sorted_dates = sorted(dates)
                dates_str = ", ".join(sorted_dates)
                print(f"## {num_instances} times ({dates_str}):")
                print(f"  {logprobs_display}")

    except sqlite3.Error as e:
        logger.error(f"An error occurred during database analysis: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    equivalence_classes()
