#!/usr/bin/env python3
"""
materialize_training.py
Build a leak-free, fight-level training CSV from data/elo.db.

• One row per bout (not per side).
• Uses pre-fight Elo for both sides from elo_fight_ratings.
• Joins frozen features_by_fight (computed strictly before that fight's year).
• Emits Δ features (A − B) + a label y = 1 if A won else 0.
• Adds convenience fields: elo_diff, same_division, wc_gap, fight_year, is_title.
• NEW: optional reach/height/sub_avg/stance and avg_dec_margin5 if present in features_by_fight.
"""

import os
import csv
import argparse
import sqlite3
from typing import Optional, List, Dict

HERE = os.path.dirname(__file__)
DB_PATH_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "data", "elo.db"))
OUT_CSV_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "data", "training_fights.csv"))

# Base numeric features that usually exist
BASE_NUMERIC_FEATURES = [
    "sig_strike_acc", "sig_strike_def", "sapm",
    "td_acc", "td_def",
    "ko_pct", "sub_pct", "dec_pct",
    "ranked_wins", "championship_wins",
    "total_fights", "avg_fight_time_min",
    "recency_index", "avg_opp_rank_last5",
    # recency block (last 5 frozen pre-bout)
    "recent_win_pct5", "recent_finish_pct5", "recent_ko_pct5",
    "recent_sub_pct5", "recent_avg_rank5", "recent_title_bouts5",
    "recent_streak", "recent_ema_form5",
]

# Optional numeric features we’ll include only if present
OPTIONAL_NUMERIC_FEATURES = [
    "reach_in",        # reach in inches
    "height_in",       # height in inches
    "sub_avg",         # submissions attempted per 15
    "avg_dec_margin5", # average judges' decision margin over last 5 (frozen pre-bout)
]

# Optional categorical/string features (handled into engineered columns)
OPTIONAL_CATEGORICAL_FEATURES = [
    "stance",  # expected: "Orthodox" / "Southpaw" / "Switch" (case-insensitive)
]

def safe(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def rget(row: sqlite3.Row, key: str, default=None):
    """sqlite3.Row doesn't have .get(); use this to read safely with a default."""
    try:
        val = row[key]
        return default if val is None else val
    except Exception:
        return default

def get_latest_run_id(con: sqlite3.Connection) -> Optional[int]:
    row = con.execute("SELECT MAX(elo_run_id) FROM elo_runs").fetchone()
    return int(row[0]) if row and row[0] is not None else None

def choose_run_id(con: sqlite3.Connection, arg_value: str) -> int:
    if arg_value == "latest":
        rid = get_latest_run_id(con)
        if rid is None:
            raise RuntimeError("elo_runs is empty. Run compute_elo.py first.")
        return rid
    try:
        rid = int(arg_value)
    except ValueError:
        raise RuntimeError(f"--elo-run must be an integer or 'latest', got: {arg_value}")
    row = con.execute("SELECT 1 FROM elo_runs WHERE elo_run_id=?", (rid,)).fetchone()
    if not row:
        raise RuntimeError(f"elo_run_id {rid} does not exist in elo_runs.")
    return rid

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cols = []
    for row in con.execute(f"PRAGMA table_info({table})"):
        cols.append(row[1])
    return cols

def norm_stance(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    if s.startswith("orth"): return "orthodox"
    if s.startswith("south"): return "southpaw"
    if "switch" in s: return "switch"
    return "unknown"

def main():
    ap = argparse.ArgumentParser(description="Export fight-level training data from elo.db")
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help=f"Path to elo.db (default: {DB_PATH_DEFAULT})")
    ap.add_argument("--elo-run", default="latest", help="elo_run_id to export (integer) or 'latest' (default)")
    ap.add_argument("--since", type=int, default=None, help="Keep fights with fight_year >= since")
    ap.add_argument("--until", type=int, default=None, help="Keep fights with fight_year <= until")
    ap.add_argument("--min-fights", type=int, default=1,
                    help="Require both sides have at least this many total_fights (default: 1)")
    ap.add_argument("--outfile", default=OUT_CSV_DEFAULT, help=f"Output CSV path (default: {OUT_CSV_DEFAULT})")
    args = ap.parse_args()

    db_path = os.path.normpath(args.db)
    out_csv = os.path.normpath(args.outfile)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Helpful indexes (safe to run repeatedly)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ratings_run_key ON elo_fight_ratings (elo_run_id, fight_key)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ratings_ids     ON elo_fight_ratings (elo_run_id, fighter_id, opponent_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_features_trip   ON features_by_fight (elo_run_id, fight_key, fighter_id)")
    con.commit()

    elo_run_id = choose_run_id(con, str(args.elo_run))

    # Determine which optional columns exist
    fbf_cols = set(table_columns(con, "features_by_fight"))
    numeric_feats = list(BASE_NUMERIC_FEATURES)
    for c in OPTIONAL_NUMERIC_FEATURES:
        if c in fbf_cols:
            numeric_feats.append(c)
    has_stance = ("stance" in fbf_cols)

    # --- Use aliases that won't collide case-insensitively with fighters fA/fB ---
    featA_alias = "ffa"
    featB_alias = "ffb"

    # Build dynamic SQL selecting A_*/B_* aliases for all features we want
    a_select = ",\n      ".join(
        [f"{featA_alias}.{c} AS A_{c}" for c in numeric_feats] +
        ([f"{featA_alias}.stance AS A_stance"] if has_stance else [])
    )
    b_select = ",\n      ".join(
        [f"{featB_alias}.{c} AS B_{c}" for c in numeric_feats] +
        ([f"{featB_alias}.stance AS B_stance"] if has_stance else [])
    )

    sql = f"""
    WITH pairs AS (
      SELECT
        rA.elo_run_id,
        rA.fight_key,
        rA.fight_year,
        rA.is_title,
        rA.weight_class AS wcA,
        rB.weight_class AS wcB,
        rA.fighter_id AS A_id,
        rB.fighter_id AS B_id,
        rA.opponent_id AS A_opp_id,
        rB.opponent_id AS B_opp_id,
        rA.pre_elo AS pre_elo_A,
        rB.pre_elo AS pre_elo_B,
        rA.result  AS result_A
      FROM elo_fight_ratings rA
      JOIN elo_fight_ratings rB
        ON rA.elo_run_id = rB.elo_run_id
       AND rA.fight_key  = rB.fight_key
       AND rA.fighter_id = rB.opponent_id
       AND rA.opponent_id= rB.fighter_id
      WHERE rA.elo_run_id = ?
        AND rA.fighter_id < rB.fighter_id
    )
    SELECT
      p.elo_run_id, p.fight_key, p.fight_year, p.is_title,
      p.wcA, p.wcB,
      p.A_id, p.B_id, p.pre_elo_A, p.pre_elo_B, p.result_A,
      fA.name AS A_name, fB.name AS B_name,
      {a_select},
      {b_select}
    FROM pairs p
    JOIN features_by_fight {featA_alias}
      ON {featA_alias}.elo_run_id = p.elo_run_id
     AND {featA_alias}.fight_key  = p.fight_key
     AND {featA_alias}.fighter_id = p.A_id
    JOIN features_by_fight {featB_alias}
      ON {featB_alias}.elo_run_id = p.elo_run_id
     AND {featB_alias}.fight_key  = p.fight_key
     AND {featB_alias}.fighter_id = p.B_id
    LEFT JOIN fighters fA ON fA.fighter_id = p.A_id
    LEFT JOIN fighters fB ON fB.fighter_id = p.B_id
    ORDER BY p.fight_year, p.fight_key;
    """

    rows = con.execute(sql, (elo_run_id,)).fetchall()

    # Prepare output
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_cols = [
        # meta
        "elo_run_id", "fight_key", "fight_year", "is_title",
        "A_id", "B_id", "A_name", "B_name",
        "wcA", "wcB", "same_division", "wc_gap",
        # priors
        "pre_elo_A", "pre_elo_B", "elo_diff",
        # label
        "y",
        # Δ core features (A − B)
        "d_sig_strike_acc", "d_sig_strike_def", "d_sapm_neg",
        "d_td_acc", "d_td_def",
        "d_ko_pct", "d_sub_pct", "d_dec_pct",
        "d_ranked_wins", "d_championship_wins",
        "d_total_fights", "d_avg_fight_time_min",
        "d_recency_index", "d_recent_win_pct5", "d_recent_finish_pct5",
        "d_recent_ko_pct5", "d_recent_sub_pct5",
        "d_recent_title_bouts5", "d_recent_streak", "d_recent_ema_form5",
        # opponent quality (lower rank better) — better schedule => positive value
        "d_opp_rank5_neg", "d_recent_avg_rank5_neg",
    ]

    # Add optional Δ columns if present
    if "reach_in" in numeric_feats:        out_cols.append("d_reach_in")
    if "height_in" in numeric_feats:       out_cols.append("d_height_in")
    if "sub_avg" in numeric_feats:         out_cols.append("d_sub_avg")
    if "avg_dec_margin5" in numeric_feats: out_cols.append("d_avg_dec_margin5")

    # Engineered stance features (binary)
    has_stance_cols = has_stance
    if has_stance_cols:
        out_cols += ["same_stance", "open_stance", "southpaw_adv", "switch_adv"]

    kept = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()

        for r in rows:
            yr = int(r["fight_year"]) if r["fight_year"] is not None else None
            if args.since is not None and yr is not None and yr < args.since:
                continue
            if args.until is not None and yr is not None and yr > args.until:
                continue

            wcA = r["wcA"]; wcB = r["wcB"]
            same_div = int(wcA == wcB) if (wcA is not None and wcB is not None) else 0
            wc_gap = (safe(wcA, 0) - safe(wcB, 0))

            preA = safe(r["pre_elo_A"], 1500.0)
            preB = safe(r["pre_elo_B"], 1500.0)

            y = 1 if str(r["result_A"]).lower() == "win" else 0

            # Gather sides into dicts
            A: Dict[str, float] = {}
            B: Dict[str, float] = {}
            for c in numeric_feats:
                A[c] = safe(rget(r, f"A_{c}"), 0.0)
                B[c] = safe(rget(r, f"B_{c}"), 0.0)

            # Optional novice filter
            if args.min_fights > 1:
                if (A.get("total_fights", 0.0) < args.min_fights) or (B.get("total_fights", 0.0) < args.min_fights):
                    continue

            # Core Δs
            d = {
                "d_sig_strike_acc": A.get("sig_strike_acc",0)-B.get("sig_strike_acc",0),
                "d_sig_strike_def": A.get("sig_strike_def",0)-B.get("sig_strike_def",0),
                "d_sapm_neg": -(A.get("sapm",0)-B.get("sapm",0)),  # lower SAPM is better
                "d_td_acc": A.get("td_acc",0)-B.get("td_acc",0),
                "d_td_def": A.get("td_def",0)-B.get("td_def",0),
                "d_ko_pct": A.get("ko_pct",0)-B.get("ko_pct",0),
                "d_sub_pct": A.get("sub_pct",0)-B.get("sub_pct",0),
                "d_dec_pct": A.get("dec_pct",0)-B.get("dec_pct",0),
                "d_ranked_wins": A.get("ranked_wins",0)-B.get("ranked_wins",0),
                "d_championship_wins": A.get("championship_wins",0)-B.get("championship_wins",0),
                "d_total_fights": A.get("total_fights",0)-B.get("total_fights",0),
                "d_avg_fight_time_min": A.get("avg_fight_time_min",0)-B.get("avg_fight_time_min",0),
                "d_recency_index": A.get("recency_index",0)-B.get("recency_index",0),
                "d_recent_win_pct5": A.get("recent_win_pct5",0)-B.get("recent_win_pct5",0),
                "d_recent_finish_pct5": A.get("recent_finish_pct5",0)-B.get("recent_finish_pct5",0),
                "d_recent_ko_pct5": A.get("recent_ko_pct5",0)-B.get("recent_ko_pct5",0),
                "d_recent_sub_pct5": A.get("recent_sub_pct5",0)-B.get("recent_sub_pct5",0),
                "d_recent_title_bouts5": A.get("recent_title_bouts5",0)-B.get("recent_title_bouts5",0),
                "d_recent_streak": A.get("recent_streak",0)-B.get("recent_streak",0),
                "d_recent_ema_form5": A.get("recent_ema_form5",0)-B.get("recent_ema_form5",0),
                # Opponent quality: lower rank number = tougher (negate so tougher A => positive)
                "d_opp_rank5_neg": -(A.get("avg_opp_rank_last5",0)-B.get("avg_opp_rank_last5",0)),
                "d_recent_avg_rank5_neg": -(A.get("recent_avg_rank5",0)-B.get("recent_avg_rank5",0)),
            }

            # Optional Δs
            if "reach_in" in numeric_feats:
                d["d_reach_in"]  = A.get("reach_in",0)  - B.get("reach_in",0)
            if "height_in" in numeric_feats:
                d["d_height_in"] = A.get("height_in",0) - B.get("height_in",0)
            if "sub_avg" in numeric_feats:
                d["d_sub_avg"]   = A.get("sub_avg",0)   - B.get("sub_avg",0)
            if "avg_dec_margin5" in numeric_feats:
                d["d_avg_dec_margin5"] = A.get("avg_dec_margin5",0) - B.get("avg_dec_margin5",0)

            row_out = {
                "elo_run_id": elo_run_id,
                "fight_key": r["fight_key"],
                "fight_year": r["fight_year"],
                "is_title": int(r["is_title"] or 0),
                "A_id": r["A_id"],
                "B_id": r["B_id"],
                "A_name": r["A_name"] or "",
                "B_name": r["B_name"] or "",
                "wcA": wcA,
                "wcB": wcB,
                "same_division": same_div,
                "wc_gap": wc_gap,
                "pre_elo_A": preA,
                "pre_elo_B": preB,
                "elo_diff": preA - preB,
                "y": y,
                **d,
            }

            # Stance engineered binaries
            if has_stance_cols:
                sa = norm_stance(rget(r, "A_stance", ""))
                sb = norm_stance(rget(r, "B_stance", ""))
                same_stance   = int(sa != "unknown" and sb != "unknown" and sa == sb)
                open_stance   = int({sa, sb} == {"orthodox", "southpaw"})
                southpaw_adv  = int(sa == "southpaw" and sb != "southpaw")
                switch_adv    = int(sa == "switch"   and sb != "switch")
                row_out.update({
                    "same_stance": same_stance,
                    "open_stance": open_stance,
                    "southpaw_adv": southpaw_adv,
                    "switch_adv": switch_adv,
                })

            w.writerow(row_out)
            kept += 1

    con.close()
    print(f"[OK] Wrote {kept} rows to {out_csv}")
    if kept == 0:
        print("WARNING: 0 rows written (check your --since/--until/--min-fights filters).")

if __name__ == "__main__":
    main()
