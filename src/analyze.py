import json
import math
import sqlite3
import statistics

from config import Config


def get_db_data() -> dict[str, list[tuple[str, str, list, list]]]:
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        raw_table_names = [row[0] for row in cursor.fetchall()]
        # Filter out sqlite_sequence and other internal sqlite tables
        table_names = [name for name in raw_table_names if not name.startswith("sqlite_")]
        results = {}
        for table_name in table_names:
            cursor.execute(f'SELECT date, prompt, top_tokens, logprobs FROM "{table_name}"')
            rows = cursor.fetchall()
            results[table_name] = []
            for date, prompt, top_tokens, logprobs in rows:
                results[table_name].append(
                    (date, prompt, list(json.loads(top_tokens)), list(json.loads(logprobs)))
                )
        return results
    except sqlite3.Error as e:
        Config.logger.error(f"An error occurred during database analysis: {e}")
    finally:
        conn.close()


def equivalence_classes():
    data = get_db_data()
    for table_name, rows in data.items():
        equivalence_classes = {}  # (top_tokens_tuple, logprobs_tuple) -> [dates]

        for date_str, _prompt, top_tokens, logprobs in rows:
            key = (tuple(top_tokens), tuple(logprobs))

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


def top_logprob_variability():
    data = get_db_data()
    for table_name, rows in data.items():
        _, _, first_top_tokens, _ = rows[0]
        top_token = first_top_tokens[0]
        top_token_logprobs = []
        for _date_str, _prompt, top_tokens, top_logprobs in rows:
            try:
                top_token_index = top_tokens.index(top_token)
                top_token_logprobs.append(top_logprobs[top_token_index])
            except ValueError:
                pass

        top_token_probs = [math.exp(logprob) for logprob in top_token_logprobs]

        print(f"\n# {table_name}")
        print(f"n_values: {len(top_token_logprobs)}")
        print("## logprobs")
        print(f"min: {min(top_token_logprobs)}")
        print(f"max: {max(top_token_logprobs)}")
        print(f"std: {statistics.stdev(top_token_logprobs)}")
        print("## probs")
        print(f"min: {min(top_token_probs)}")
        print(f"max: {max(top_token_probs)}")
        print(f"std: {statistics.stdev(top_token_probs)}")


if __name__ == "__main__":
    # equivalence_classes()
    top_logprob_variability()
