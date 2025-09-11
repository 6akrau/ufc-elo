# src/db_check.py
import os, sqlite3

DB = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "elo.db"))
print("DB:", DB)

con = sqlite3.connect(DB)
cur = con.cursor()

tables = ["elo_runs", "elo_fight_ratings", "features_by_fight", "elo_standings"]
for t in tables:
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n} rows")
    except sqlite3.Error as e:
        print(f"{t}: (missing?) {e}")

print("\nSample elo_fight_ratings rows:")
for row in cur.execute("""
    SELECT elo_run_id, fight_key, fight_year, fighter_id, opponent_id,
           pre_elo, post_elo, delta_elo
    FROM elo_fight_ratings
    LIMIT 5
"""):
    print(row)

con.close()
