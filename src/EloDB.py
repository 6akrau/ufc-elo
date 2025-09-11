# src/EloDB.py
import sqlite3, json, hashlib, os
from contextlib import contextmanager

DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "elo.db"))

@contextmanager
def db(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    try:
        yield con
        con.commit()
    finally:
        con.close()

def create_schema(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS fighters(
      fighter_id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      name_norm TEXT UNIQUE NOT NULL,
      default_wc INTEGER,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS elo_runs(
      elo_run_id INTEGER PRIMARY KEY,
      rules_version TEXT NOT NULL,
      params_json TEXT NOT NULL,
      source_hash TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS elo_fight_ratings(
      elo_run_id INTEGER NOT NULL,
      fight_key TEXT NOT NULL,
      fight_date TEXT,
      fight_year INTEGER,
      fighter_id INTEGER NOT NULL,
      opponent_id INTEGER,
      opp_name_norm TEXT,
      weight_class INTEGER,
      is_title INTEGER NOT NULL,
      rank_opp TEXT,
      method TEXT,
      result TEXT,
      pre_elo REAL NOT NULL,
      post_elo REAL NOT NULL,
      delta_elo REAL NOT NULL,
      recency_weight REAL NOT NULL,
      PRIMARY KEY(elo_run_id, fight_key, fighter_id)
    );
    /* features frozen as-of the bout (per side) */
    CREATE TABLE IF NOT EXISTS features_by_fight(
      elo_run_id INTEGER NOT NULL,
      fight_key TEXT NOT NULL,
      fighter_id INTEGER NOT NULL,
      sig_strike_acc REAL, sig_strike_def REAL, sapm REAL,
      td_acc REAL, td_def REAL, ko_pct REAL, sub_pct REAL, dec_pct REAL,
      ranked_wins REAL, championship_wins REAL, total_fights REAL, avg_fight_time_min REAL,
      recency_index REAL, avg_opp_rank_last5 REAL,
      recent_win_pct5 REAL, recent_finish_pct5 REAL, recent_ko_pct5 REAL, recent_sub_pct5 REAL,
      recent_avg_rank5 REAL, recent_title_bouts5 REAL, recent_streak REAL, recent_ema_form5 REAL,
      avg_dec_margin5 REAL,
      PRIMARY KEY(elo_run_id, fight_key, fighter_id)
    );
    CREATE TABLE IF NOT EXISTS elo_standings(
      elo_run_id INTEGER NOT NULL,
      fighter_id INTEGER NOT NULL,
      weight_class INTEGER NOT NULL,
      elo REAL NOT NULL,
      PRIMARY KEY(elo_run_id, fighter_id, weight_class)
    );
    """)
    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_rates_run_fid ON elo_fight_ratings(elo_run_id, fighter_id);
    CREATE INDEX IF NOT EXISTS idx_rates_run_key ON elo_fight_ratings(elo_run_id, fight_key);
    """)

    # After the executescript blocks in create_schema(...)
    cols = [r[1] for r in con.execute("PRAGMA table_info(features_by_fight)")]
    if "avg_dec_margin5" not in cols:
        con.execute("ALTER TABLE features_by_fight ADD COLUMN avg_dec_margin5 REAL")

def norm(s: str) -> str:
    return (s or "").encode("ascii","ignore").decode().strip().lower()

def upsert_fighter(con, name: str, default_wc=None):
    nn = norm(name)
    row = con.execute("SELECT fighter_id FROM fighters WHERE name_norm=?", (nn,)).fetchone()
    if row: return row[0]
    cur = con.execute("INSERT INTO fighters(name,name_norm,default_wc) VALUES(?,?,?)",
                      (name, nn, default_wc))
    return cur.lastrowid

def start_elo_run(con, rules_version: str, params_dict: dict, source_bytes: bytes):
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    cur = con.execute(
        "INSERT INTO elo_runs(rules_version,params_json,source_hash) VALUES(?,?,?)",
        (rules_version, json.dumps(params_dict, separators=(',',':')), source_hash)
    )
    return cur.lastrowid

def insert_row(con, table: str, **kw):
    cols = ",".join(kw.keys())
    qs = ",".join("?" for _ in kw)
    con.execute(f"INSERT OR REPLACE INTO {table}({cols}) VALUES({qs})", tuple(kw.values()))
