#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_logit.py — fit stats-only logistic with Elo as an offset; write weights to JSON (and optionally DB).

NEW: Supports Δreach/Δheight/Δsub_avg and stance matchup binaries (open_stance, southpaw_adv, switch_adv, same_stance)
     if present in the training CSV. Everything is optional/robust.
"""

import os, json, math, csv, argparse, datetime, sqlite3
from typing import List, Dict, Tuple, Optional
from statistics import mean, pstdev

# Default paths
HERE = os.path.dirname(__file__)
CSV_DEFAULT  = os.path.normpath(os.path.join(HERE, "..", "data", "training_fights.csv"))
JSON_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "data", "model_weights.json"))
DB_DEFAULT   = os.path.normpath(os.path.join(HERE, "..", "data", "elo.db"))

# ---------- Baseline Δ columns (from materialize_training.py) ----------
BASE_DELTA_COLS = [
    "d_sig_strike_acc","d_sig_strike_def","d_sapm_neg",
    "d_td_acc","d_td_def",
    "d_ko_pct","d_sub_pct","d_dec_pct",
    "d_ranked_wins","d_championship_wins",
    "d_total_fights","d_avg_fight_time_min",
    "d_recency_index","d_recent_win_pct5","d_recent_finish_pct5",
    "d_recent_ko_pct5","d_recent_sub_pct5",
    "d_recent_title_bouts5","d_recent_streak","d_recent_ema_form5",
    "d_opp_rank5_neg","d_recent_avg_rank5_neg",
    # last-3 block (if present)
    "d_recent_win_pct3","d_recent_avg_rank3_neg",
    # context
    "same_division","wc_gap",
    # optional numerics
    "d_reach_in","d_height_in","d_sub_avg",
    "d_avg_dec_margin5",     # <— add this
    # stance binaries
    "open_stance","southpaw_adv","switch_adv","same_stance",
]


# ---------- Engineered delta columns ----------
ENGINEERED_COLS = [
    # power/durability
    "d_ko_rate","d_sub_rate","d_finish_rate",
    "d_ko_loss_adv","d_sub_loss_adv",
    # career opponent quality (if present): lower rank = tougher, so we store NEG diff
    "d_career_avg_rank_neg",
]

def logit(p): return math.log(p/(1-p))
def sigmoid(z): return 1.0/(1.0+math.exp(-z))

def load_rows(csv_path: str) -> List[Dict[str,str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

def stdz_fit(X: List[Dict[str, float]], cols: List[str]) -> Tuple[Dict[str,float], Dict[str,float]]:
    mu = {c: mean([float(x.get(c, 0.0)) for x in X]) for c in cols}
    sd = {}
    for c in cols:
        vals = [float(x.get(c, 0.0)) for x in X]
        s = pstdev(vals)
        sd[c] = (s if s > 1e-8 else 1.0)
    return mu, sd

def stdz_apply(x: Dict[str, float], mu, sd, cols: List[str]) -> Dict[str, float]:
    return {c: (float(x.get(c, 0.0)) - mu[c]) / sd[c] for c in cols}

# ---------- Robust getters ----------
def _get_num(row: Dict[str,str], key_variants: List[str]) -> Optional[float]:
    for k in key_variants:
        if k in row and row[k] not in (None, "", "NA"):
            try:
                return float(row[k])
            except Exception:
                pass
    return None

def _ab(row: Dict[str,str], side: str, metric: str) -> Optional[float]:
    side = side.upper()  # "A" or "B"
    metric = metric.lower()
    cand = [
        f"{side}_{metric}", f"{side.lower()}_{metric}",
        f"{metric}_{side}", f"{metric}_{side.lower()}",
    ]
    return _get_num(row, cand)

def _rate(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return float(num)/float(den) * 100.0  # percentage 0..100

def derive_power_durability(row: Dict[str,str]) -> Dict[str, float]:
    out: Dict[str,float] = {}
    # wins
    A_ko_w = _ab(row, "A", "ko_wins");   B_ko_w = _ab(row, "B", "ko_wins")
    A_sub_w= _ab(row, "A", "sub_wins");  B_sub_w= _ab(row, "B", "sub_wins")
    # losses
    A_ko_l = _ab(row, "A", "ko_losses"); B_ko_l = _ab(row, "B", "ko_losses")
    A_sub_l= _ab(row, "A", "sub_losses");B_sub_l= _ab(row, "B", "sub_losses")
    # totals
    A_tot  = _ab(row, "A", "total_fights"); B_tot = _ab(row, "B", "total_fights")

    # win rates
    A_ko_rate  = _rate(A_ko_w,  A_tot);  B_ko_rate  = _rate(B_ko_w,  B_tot)
    A_sub_rate = _rate(A_sub_w, A_tot);  B_sub_rate = _rate(B_sub_w, B_tot)
    if A_ko_rate is not None and B_ko_rate is not None:
        out["d_ko_rate"] = A_ko_rate - B_ko_rate
    if A_sub_rate is not None and B_sub_rate is not None:
        out["d_sub_rate"] = A_sub_rate - B_sub_rate
    if (A_ko_rate is not None and A_sub_rate is not None and
        B_ko_rate is not None and B_sub_rate is not None):
        out["d_finish_rate"] = (A_ko_rate + A_sub_rate) - (B_ko_rate + B_sub_rate)

    # durability (loss rates): advantage defined so higher = safer for A
    A_ko_loss_rate  = _rate(A_ko_l,  A_tot);  B_ko_loss_rate  = _rate(B_ko_l,  B_tot)
    A_sub_loss_rate = _rate(A_sub_l, A_tot);  B_sub_loss_rate = _rate(B_sub_l, B_tot)
    if A_ko_loss_rate is not None and B_ko_loss_rate is not None:
        out["d_ko_loss_adv"]  = B_ko_loss_rate  - A_ko_loss_rate
    if A_sub_loss_rate is not None and B_ko_loss_rate is not None:
        out["d_sub_loss_adv"] = B_ko_loss_rate  - A_sub_loss_rate
    return out

def derive_career_opp_rank(row: Dict[str,str]) -> Dict[str, float]:
    out: Dict[str,float] = {}
    a_keys = ["A_career_avg_rank","A_avg_opp_rank_all","A_avg_opp_rank","A_opp_rank_avg_career"]
    b_keys = ["B_career_avg_rank","B_avg_opp_rank_all","B_avg_opp_rank","B_opp_rank_avg_career"]
    A = _get_num(row, a_keys); B = _get_num(row, b_keys)
    if A is not None and B is not None:
        out["d_career_avg_rank_neg"] = A - B
    return out

def main():
    ap = argparse.ArgumentParser(description="Train stats-only logistic with Elo offset; write weights JSON.")
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=JSON_DEFAULT)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--model-name", default="logit_v3_reach_stance_subavg")
    # Align with runtime defaults
    ap.add_argument("--ELO_K", type=float, default=1/800)
    ap.add_argument("--W_ELO", type=float, default=0.20)
    ap.add_argument("--welo_min", type=float, default=None)
    ap.add_argument("--welo_max", type=float, default=None)
    ap.add_argument("--since", type=int, default=None)
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--include-legacy-finishes", action="store_true",
                    help="Keep d_ko_pct/d_sub_pct/d_dec_pct")
    ap.add_argument("--stats_temp", type=float, default=0.35)
    ap.add_argument("--clip_z", type=float, default=4.0)
    ap.add_argument("--combine", choices=["additive","blend"], default="additive",
                    help="How inference should combine Elo and stats; 'additive' recommended with Elo offset training")
    # Guardrails / emphasis
    ap.add_argument("--enforce_monotonic", action="store_true",
                    help="Clamp sign so KO/SUB/finish rates and durability advantages are ≥0")
    ap.add_argument("--min_beta_ko", type=float, default=None)
    ap.add_argument("--min_beta_dur", type=float, default=None)
    ap.add_argument("--scale_dur", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import statsmodels.api as sm
        import numpy as np
    except Exception:
        print("ERROR: statsmodels (and numpy) not installed. Please `pip install statsmodels numpy` and rerun.")
        return

    rows = load_rows(args.csv)
    if args.since is not None:
        rows = [r for r in rows if r.get("fight_year") and int(r["fight_year"]) >= args.since]
    if args.until is not None:
        rows = [r for r in rows if r.get("fight_year") and int(r["fight_year"]) <= args.until]
    if not rows:
        print("ERROR: no rows after filtering.")
        return

    # Targets + Elo offset
    y = [int(r["y"]) for r in rows]
    elo_diff = [float(r["elo_diff"]) for r in rows]
    p_elo = [1.0/(1.0 + 10**(-args.ELO_K * d)) for d in elo_diff]
    z_off = [logit(max(min(p, 1-1e-8), 1e-8)) for p in p_elo]

    # Assemble X (only keep columns actually present in the CSV header)
    header_cols = set(rows[0].keys())
    base_cols = [c for c in BASE_DELTA_COLS if c in header_cols]
    if not args.include_legacy_finishes:
        for c in ("d_ko_pct","d_sub_pct","d_dec_pct"):
            if c in base_cols: base_cols.remove(c)

    Xraw: List[Dict[str,float]] = []
    eng_present_counts = {k: 0 for k in ENGINEERED_COLS}
    for r in rows:
        # baseline deltas present in CSV
        x = {c: float(r.get(c, 0.0) or 0.0) for c in base_cols}

        # engineered: power/durability (from totals and win/loss counts if present)
        pd = derive_power_durability(r)
        for k,v in pd.items():
            x[k] = float(v); eng_present_counts[k] += 1

        # engineered: career opponent rank (optional)
        cr = derive_career_opp_rank(r)
        for k,v in cr.items():
            x[k] = float(v); eng_present_counts[k] += 1

        Xraw.append(x)

    # Final design
    eng_cols_kept = [k for k in ENGINEERED_COLS if eng_present_counts[k] > 0]
    DELTA_COLS: List[str] = base_cols + eng_cols_kept
    if not DELTA_COLS:
        print("ERROR: No features found to train on. Check CSV and flags.")
        return

    # Standardize and fit GLM with offset
    mu, sd = stdz_fit(Xraw, DELTA_COLS)
    Xstd = [stdz_apply(x, mu, sd, DELTA_COLS) for x in Xraw]
    Xmat = np.array([[x.get(c, 0.0) for c in DELTA_COLS] for x in Xstd], dtype=float)
    yvec = np.array(y, dtype=float)
    offset = np.array(z_off, dtype=float)

    Xmat_with_intercept = np.column_stack([np.ones(len(Xmat)), Xmat])
    try:
        model = sm.GLM(yvec, Xmat_with_intercept, family=sm.families.Binomial(), offset=offset)
        res = model.fit(maxiter=300)
    except Exception as e:
        print(f"ERROR: GLM fit failed: {e}")
        return

    # Params
    params = res.params
    B0 = float(params[0])
    B = {DELTA_COLS[i]: float(params[i+1]) for i in range(len(DELTA_COLS))}

    # Guardrails / emphasis
    notes = []
    if args.enforce_monotonic:
        for k in ["d_ko_rate","d_sub_rate","d_finish_rate","d_ko_loss_adv","d_sub_loss_adv"]:
            if k in B and B[k] < 0.0:
                B[k] = 0.0
                notes.append(f"clamped {k} to ≥0")
    if args.min_beta_ko is not None:
        for k in ["d_ko_rate","d_sub_rate","d_finish_rate"]:
            if k in B and B[k] < args.min_beta_ko:
                B[k] = args.min_beta_ko
                notes.append(f"floored {k} to {args.min_beta_ko}")
    if args.min_beta_dur is not None:
        for k in ["d_ko_loss_adv","d_sub_loss_adv"]:
            if k in B and B[k] < args.min_beta_dur:
                B[k] = args.min_beta_dur
                notes.append(f"floored {k} to {args.min_beta_dur}")
    if args.scale_dur and abs(args.scale_dur-1.0) > 1e-9:
        for k in ["d_ko_loss_adv","d_sub_loss_adv"]:
            if k in B: B[k] *= args.scale_dur
        notes.append(f"scaled durability betas by ×{args.scale_dur}")

    # Output JSON
    out = {
        "version": "v3_reach_stance_subavg",
        "created": datetime.datetime.utcnow().isoformat() + "Z",
        "ELO_K": args.ELO_K,
        "W_ELO": args.W_ELO,
        "B0": B0,
        "B": B,
        "scaler": {
            "feature_order": DELTA_COLS,
            "mean": {k: mu[k] for k in DELTA_COLS},
            "std":  {k: sd[k] for k in DELTA_COLS},
        },
        "fit_stats": {
            "n": int(len(Xmat)),
            "llf": float(res.llf),
            "aic": float(res.aic),
            "bic": float(getattr(res, "bic", float("nan"))),
        },
        "config": {
            "ELO_K": args.ELO_K,
            "W_ELO": args.W_ELO,
            "W_ELO_MIN": args.welo_min,
            "W_ELO_MAX": args.welo_max,
            "stats_temp": args.stats_temp,
            "clip_z": args.clip_z,
            "combine": args.combine,
        },
        "engineered_presence": eng_present_counts,
        "training_notes": notes,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote weights to {os.path.abspath(args.out)}")
    if eng_present_counts:
        derived = ", ".join(f"{k}:{v}" for k,v in eng_present_counts.items())
        print(f"[info] engineered features present counts → {derived}")
    if notes:
        print(f"[notes] {'; '.join(notes)}")

    # Optional DB logging
    if args.write_db:
        con = sqlite3.connect(args.db)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
          model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at   TEXT,
          model_name   TEXT,
          elo_k        REAL,
          w_elo        REAL,
          notes        TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_weights (
          model_run_id INTEGER,
          feature_name TEXT,
          weight       REAL,
          FOREIGN KEY(model_run_id) REFERENCES model_runs(model_run_id)
        );
        """)
        cur.execute("INSERT INTO model_runs(created_at,model_name,elo_k,w_elo,notes) VALUES(?,?,?,?,?)",
                    (out["created"], args.model_name, args.ELO_K, args.W_ELO,
                     "GLM with Elo offset; additive combine; +reach/height/sub_avg + stance binaries"))
        mrid = cur.lastrowid
        cur.execute("INSERT INTO model_weights(model_run_id,feature_name,weight) VALUES(?,?,?)",
                    (mrid, "__INTERCEPT__", B0))
        for k,v in B.items():
            cur.execute("INSERT INTO model_weights(model_run_id,feature_name,weight) VALUES(?,?,?)",
                        (mrid, k, v))
        con.commit(); con.close()
        print(f"[OK] logged model into {os.path.abspath(args.db)} (model_run_id={mrid})")

if __name__ == "__main__":
    main()
