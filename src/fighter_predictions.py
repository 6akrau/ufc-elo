#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fighter_predictions.py

Elo-anchored fight predictor with:
- GLM/Logistic "training logits" loaded from model_weights.json (B0/B and optional scaler)
- Random Forest stats model support; **enabled by default** if ../data/rf_model.pkl exists
  (or pass --rf_model). Blend weight configurable via --w_rf (default 0.35). Use --no_rf to disable.
- RF is gracefully skipped if NumPy is unavailable or the model path is missing.
- Percent-to-fraction fallback for 0..100% features (--assume_pct01)
- Temperature to shrink stats logit (--stats_temp)
- Elo shaping: temperature (--elo_temp) + cap (--elo_cap)
- Combine modes: additive / additive_dyn / blend for Elo+stats
- New features: stance (orthodox/southpaw/switch) + engineered binaries (same_stance, open_stance, southpaw_adv, switch_adv)
- New features: reach_in, height_in, sub_avg (per 15)
- Decision-quality signals (dec_dom5, sd_tilt5, ud_rate5)
- Rich CLI with top drivers and optional full profile

Notes:
• If an RF model is available, we blend the GLM stats logit and the RF stats logit:
  z_stats_combo = (1 - w_rf)*z_stats_glm + w_rf*z_stats_rf  (default w_rf=0.35).
"""

import json, math, unicodedata, argparse, os, sys, re, statistics, pickle
from difflib import get_close_matches
from typing import Optional, Dict, Tuple, List
# numpy/sklearn are optional, only used when an RF model is loaded
try:
    import numpy as np
except Exception:
    np = None
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# ========================== Defaults (used if no weights file) ==========================
ELO_K = 1/900
W_ELO = 0.2
USE_DYNAMIC_ELO = True  # dynamic Elo weighting based on rank/fights/recency

# New tuning defaults (can be overridden via JSON config or CLI)
COMBINE_DEFAULT = "blend"     # "additive", "additive_dyn", or "blend"
ELO_TEMP_DEFAULT = 0.55       # shrink Elo logit; 1.0 = no shrink
ELO_CAP_DEFAULT  = 0.90       # cap |z_elo| so Elo can’t run away
ELO_ACTIVITY_BETA_DEFAULT = 0.30  # reserved for optional directional activity nudge (unused here)

# If an RF is supplied, blend it into the stats logit with this weight (0..1)
RF_STATS_WEIGHT_DEFAULT = 0.35

# Fallback intercept & weights (only if model_weights.json absent)
B0 = 0.0
B_DEFAULT = {
    # --- Core striking/grappling skill deltas ---
    "ssa": 0.020,    # Sig. strike accuracy
    "ssd": 0.018,    # Sig. strike defense
    "sapm": 0.025,   # Lower absorbed/min is better (already signed as advantage)

    "tda": 0.008,    # TD accuracy
    "tdd": 0.020,    # TD defense

    # --- Legacy finish profiles (kept ~zero to avoid double counting with rates below) ---
    "ko":  0.000,
    "sub": 0.000,
    "dec": 0.000,

    # --- Résumé/meta ---
    "ranked_wins": 0.025,
    "champ_wins":  0.035,
    "exp":         0.001,
    "aft":         0.001,
    "recency":     0.100,
    "opp_rank5":   0.015,
    "same_div":    0.050,
    "wc_gap":     -0.020,

    # --- Recent form (light touch so they complement, not dominate) ---
    "r_win5":      0.030,
    "r_ema5":      0.040,
    "r_streak":    0.020,
    "r_avg_rank5": 0.015,

    "r_win3":      0.040,
    "r_ema3":      0.050,
    "r_streak3":   0.020,
    "r_avg_rank3": 0.015,

    # --- Power & durability signals ---
    "ko_rate":      0.300,  # KO wins / total fights (%)
    "sub_rate":     0.100,  # SUB wins / total fights (%)
    "finish_rate":  0.100,  # (KO+SUB) wins / total fights (%)
    "ko_loss_adv":  0.450,  # durability edges: higher is better
    "sub_loss_adv": 0.350,

    # --- Decision quality (last 5 decisions) ---
    "dec_dom5": 0.080,
    "sd_tilt5": 0.060,
    "ud_rate5": 0.050,

    # --- NEW Anthropometrics / stance (modest priors, adjust to taste) ---
    "reach":        0.010,  # per inch reach edge
    "height":       0.006,  # per inch height edge
    "sub_avg":      0.015,  # submissions attempted per 15 (A-B)
    "same_stance":  0.015,  # small bonus for familiarity
    "open_stance":  0.030,  # orthodox vs southpaw creates asymmetry; slight pull
    "southpaw_adv": 0.040,  # southpaw often underpriced
    "switch_adv":   0.015,  # switch can create tricky looks
}
# ========================================================================================

# Trainer→Runtime feature name map (if your model_weights.json uses training CSV column names).
# Supports BOTH v1 and v2 trainer column names for robustness.
TRAIN_TO_SHORT = {
    # core deltas
    "d_sig_strike_acc": "ssa",
    "d_sig_strike_def": "ssd",
    "d_sapm_neg":       "sapm",
    "d_td_acc":         "tda",
    "d_td_def":         "tdd",
    "d_ko_pct":         "ko",
    "d_sub_pct":        "sub",
    "d_dec_pct":        "dec",
    "d_ranked_wins":    "ranked_wins",
    "d_championship_wins": "champ_wins",
    "d_total_fights":   "exp",
    "d_avg_fight_time_min": "aft",
    "d_recency_index":  "recency",
    "d_opp_rank5_neg":  "opp_rank5",     # negative diff in training → positive means tougher A
    "same_division":    "same_div",
    "wc_gap":           "wc_gap",

    # recent 5 features
    "d_recent_win_pct5":     "r_win5",
    "d_recent_finish_pct5":  "r_fin5",
    "d_recent_ko_pct5":      "r_ko5",
    "d_recent_sub_pct5":     "r_sub5",
    "d_recent_title_bouts5": "r_title5",
    "d_recent_streak":       "r_streak",
    "d_recent_ema_form5":    "r_ema5",
    "d_recent_avg_rank5_neg":"r_avg_rank5",

    # recent 3 features
    "d_recent_win_pct3":     "r_win3",
    "d_recent_avg_rank3_neg":"r_avg_rank3",

    # power & vulnerability (v2 names)
    "d_ko_rate":        "ko_rate",
    "d_sub_rate":       "sub_rate",
    "d_finish_rate":    "finish_rate",
    "d_ko_loss_adv":    "ko_loss_adv",
    "d_sub_loss_adv":   "sub_loss_adv",

    # v1 names (compat)
    "d_ko_win_rate":       "ko_rate",
    "d_sub_win_rate":      "sub_rate",
    "d_finish_win_rate":   "finish_rate",
    "d_ko_loss_rate_neg":  "ko_loss_adv",
    "d_sub_loss_rate_neg": "sub_loss_adv",

    # decision
    "d_dec_dom5": "dec_dom5",
    "d_sd_tilt5": "sd_tilt5",
    "d_ud_rate5": "ud_rate5",

    # NEW anthropometrics + stance (from materialize_training)
    "d_reach_in":   "reach",
    "d_height_in":  "height",
    "d_sub_avg":    "sub_avg",
    "same_stance":  "same_stance",
    "open_stance":  "open_stance",
    "southpaw_adv": "southpaw_adv",
    "switch_adv":   "switch_adv",

    "d_recent_ko_loss_pct5": "ko_loss_recent5",   # recent chin fragility (fraction 0..1)
    "d_ko_powervschin": "ko_powervschin",        # matchup: opp KO power vs my chin

}
SHORT_FROM_TRAIN = {v: k for k, v in TRAIN_TO_SHORT.items()}

# --------------------------- Name normalization ---------------------------
def normalize_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().strip().lower()

# --------------------------- Division snapping ---------------------------
def snap_div(w):
    if w is None: return None
    try: w = float(w)
    except: return None
    if w <= 124: return 125
    if w <= 140: return 135
    if w <= 150: return 145
    if w <= 160: return 155
    if w <= 177: return 170
    if w <= 197: return 185
    if w <= 215: return 205
    return 265

# --------------------------- Data loaders ---------------------------
def load_rankings(path):
    by_wc = json.load(open(path, "r", encoding="utf-8"))
    m = {}
    for wc, rows in by_wc.items():
        for r in rows:
            m[normalize_name(r["name"])] = (int(wc), float(r["elo"]))
    return m

def load_stats(path):
    arr = json.load(open(path, "r", encoding="utf-8"))
    return { normalize_name(d["name"]): d for d in arr }

def load_weights(path):
    """
    Supports:
      {
        "B0": float,
        "B": { "<feat>": weight, ... },          # keys can be training names or short names
        "config": { "ELO_K": ..., "W_ELO": ..., "stats_temp": ..., "clip_z": ...,
                    "combine": "additive|additive_dyn|blend",
                    "elo_temp": ..., "elo_cap": ..., "W_ELO_MIN": ..., "W_ELO_MAX": ... },
        "scaler": {
          "means"/"mean": { "<feat>": m, ... },  # supports plural or singular
          "stds"/"std":  { "<feat>": s, ... }
        }
      }
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except Exception as e:
        print(f"[warn] failed to read weights from {path}: {e}", file=sys.stderr)
        return None

def _recent_ko_loss_frac(recs, N=5) -> float:
    last = [fr for fr in recs if fr.get("year", 0) > 0 and fr.get("result") in ("win","loss")]
    last = sorted(enumerate(last), key=lambda t: (t[1]["year"], t[0]))[-N:]
    last = [fr for _, fr in last]
    if not last:
        return 0.0
    def is_ko_loss(fr):
        if str(fr.get("result","")).lower() != "loss":
            return False
        m = str(fr.get("method","")).upper()
        return ("KO" in m) or ("TKO" in m)
    return sum(is_ko_loss(fr) for fr in last) / float(len(last))

# --------------------------- Activity / Elo helpers ---------------------------
def activity_index(fights) -> float:
    if not fights: return 0.0
    years = []
    for fr in fights:
        y = fr.get("year")
        if isinstance(y, (int, float)) and y > 0:
            years.append(int(y))
    if not years:
        return 0.0
    cur = max(years)
    recent = sum(1 for y in years if y >= cur - 1)
    return max(0.0, min(1.0, recent / 3.0))

def shape_elo_logit(z_elo: float, temp: float, cap: Optional[float]) -> Tuple[float, float]:
    if temp is None:
        temp = 1.0
    z = z_elo * float(temp)
    scale_used = float(temp)
    if cap is not None:
        try:
            c = abs(float(cap))
            if z > c:
                z = c
                scale_used = (z / z_elo) if z_elo != 0 else 0.0
            elif z < -c:
                z = -c
                scale_used = (z / z_elo) if z_elo != 0 else 0.0
        except Exception:
            pass
    return z, scale_used

# --------------------------- Feature engineering ---------------------------
def recency_index(fights):
    years = [fr["year"] for fr in fights if fr.get("year",0)>0 and fr.get("result") in ("win","loss")]
    if not years: return 0.0
    y = max(years)
    if y <= 2016: return 0.0
    if y <= 2019: return 0.2
    if y <= 2021: return 0.35
    if y == 2022: return 0.5
    if y == 2023: return 0.75
    return 1.0

def avg_opp_rank_last5(fights):
    ranks=[]
    for fr in fights:
        r = fr.get("rank")
        if isinstance(r, str) and r.isdigit(): ranks.append(int(r))
        elif isinstance(r, int): ranks.append(r)
    ranks = [x for x in ranks if 1<=x<=15][-5:]
    return sum(ranks)/len(ranks) if ranks else 20.0

def norm_stance(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    if s.startswith("orth"): return "orthodox"
    if s.startswith("south"): return "southpaw"
    if "switch" in s: return "switch"
    return "unknown"

def _recent_block(recs, N=5):
    last = [fr for fr in recs if fr.get("year",0)>0 and fr.get("result") in ("win","loss")]
    last = sorted(enumerate(last), key=lambda t: (t[1]["year"], t[0]))[-N:]
    last = [fr for _, fr in last]
    if not last:
        return dict(
            r_win5=0.0, r_fin5=0.0, r_ko5=0.0, r_sub5=0.0,
            r_avg_rank5=20.0, r_title5=0.0, r_streak=0.0, r_ema5=0.0
        )

    wins = [fr for fr in last if fr["result"]=="win"]
    def _pct(cond): return 100.0*sum(cond(fr) for fr in wins)/len(wins) if wins else 0.0
    M = lambda fr: str(fr.get("method","")).upper()
    ko_pct = _pct(lambda fr: ("KO" in M(fr)) or ("TKO" in M(fr)))
    sub_pct= _pct(lambda fr: ("SUB" in M(fr)))
    fin_pct= _pct(lambda fr: ("KO" in M(fr)) or ("TKO" in M(fr)) or ("SUB" in M(fr)))
    ranks=[]
    for fr in last:
        r=fr.get("rank")
        if isinstance(r,str) and r.isdigit(): ranks.append(int(r))
        elif isinstance(r,int): ranks.append(r)
    ranks=[x for x in ranks if 1<=x<=15]
    avg_rank = sum(ranks)/len(ranks) if ranks else 20.0
    title_bouts = sum(bool(fr.get("is_title")) for fr in last)
    streak=0
    for fr in reversed(last):
        if fr["result"]=="win":
            if streak>=0: streak+=1
            else: break
        else:
            if streak<=0: streak-=1
            else: break
    alpha=0.6; ema=None
    for fr in last:
        x=1.0 if fr["result"]=="win" else 0.0
        ema = x if ema is None else alpha*x+(1-alpha)*ema
    return dict(r_win5=100.0*len(wins)/len(last),
                r_fin5=fin_pct, r_ko5=ko_pct, r_sub5=sub_pct,
                r_avg_rank5=avg_rank, r_title5=float(title_bouts),
                r_streak=float(streak), r_ema5=float(ema if ema is not None else 0.0))

def _recent_block_n(recs, N=3):
    last = [fr for fr in recs if fr.get("year",0)>0 and fr.get("result") in ("win","loss")]
    last = sorted(enumerate(last), key=lambda t: (t[1]["year"], t[0]))[-N:]
    last = [fr for _, fr in last]
    if not last:
        return dict(r_win3=0.0, r_avg_rank3=20.0, r_streak3=0.0, r_ema3=0.0)
    wins = sum(fr["result"]=="win" for fr in last)
    ranks=[]
    for fr in last:
        r=fr.get("rank")
        if isinstance(r,str) and r.isdigit(): ranks.append(int(r))
        elif isinstance(r,int): ranks.append(r)
    ranks=[x for x in ranks if 1<=x<=15]
    avg_rank = sum(ranks)/len(ranks) if ranks else 20.0
    streak=0
    for fr in reversed(last):
        if fr["result"]=="win":
            if streak>=0: streak+=1
            else: break
        else:
            if streak<=0: streak-=1
            else: break
    alpha=0.6; ema=None
    for fr in last:
        x=1.0 if fr["result"]=="win" else 0.0
        ema = x if ema is None else alpha*x+(1-alpha)*ema
    return dict(
        r_win3=100.0*wins/len(last),
        r_avg_rank3=avg_rank,
        r_streak3=float(streak),
        r_ema3=float(ema if ema is not None else 0.0),
    )

def _first(d, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

def feature_vector(d):
    recs = d.get("fight_records", [])
    wc_raw = d.get("fight_weight_class") or d.get("bout_weight_class") or d.get("weight_class") or d.get("wc") or d.get("division") or d.get("weight")

    # raw counts if available (robust to naming)
    tot = _first(d, "total_fights", "fights", "record_total")
    ko_w = _first(d, "ko_wins", "wins_by_ko", "wins_ko")
    sub_w= _first(d, "sub_wins", "wins_by_sub", "wins_sub")
    ko_l = _first(d, "ko_losses", "losses_by_ko", "loss_ko")
    sub_l= _first(d, "sub_losses", "losses_by_sub", "loss_sub")

    def _pct(num, den):
        try:
            n = float(num or 0.0); dden = float(den or 0.0)
            return (100.0 * n / dden) if dden > 0 else 0.0
        except Exception:
            return 0.0

    ko_rate       = _pct(ko_w,  tot)     # %
    sub_rate      = _pct(sub_w, tot)     # %
    finish_rate   = ko_rate + sub_rate   # %
    ko_loss_rate  = _pct(ko_l,  tot)     # %
    sub_loss_rate = _pct(sub_l, tot)     # %

    base = {
        "ssa": float(d.get("sig_strike_acc") or 0.0),
        "ssd": float(d.get("sig_strike_def") or 0.0),
        "sapm": float(d.get("sapm") or 0.0),
        "tda": float(d.get("td_acc") or 0.0),
        "tdd": float(d.get("td_def") or 0.0),

        # legacy profiles (back-compat; trainer may ignore)
        "ko":  float(d.get("ko_pct") or 0.0),
        "sub": float(d.get("sub_pct") or 0.0),
        "dec": float(d.get("dec_pct") or 0.0),

        "ranked_wins": float(d.get("ranked_wins") or 0.0),
        "champ_wins":  float(d.get("championship_wins") or 0.0),
        "exp": float(d.get("total_fights") or 0.0),
        "aft": float((d.get("avg_fight_time_sec") or 0.0))/60.0,
        "recency": recency_index(recs),
        "opp_rank5": avg_opp_rank_last5(recs),
        "wc_snap": snap_div(wc_raw),

        # per-side rates
        "ko_rate": ko_rate,
        "sub_rate": sub_rate,
        "finish_rate": finish_rate,
        "ko_loss_rate": ko_loss_rate,
        "sub_loss_rate": sub_loss_rate,

        # NEW per-side anthropometrics & stance
        "reach": float(d.get("reach_in") or 0.0),
        "height": float(d.get("height_in") or 0.0),
        "sub_avg": float(d.get("sub_avg") or 0.0),
        "stance": norm_stance(d.get("stance")),
    }
    base.update(_recent_block(recs, N=5))
    base.update(_recent_block_n(recs, N=3))
    base.update(_decision_block(recs, N=5))
    # recent chin fragility (fraction 0..1)
    base["ko_loss_recent5"] = _recent_ko_loss_frac(recs, N=5)

    return base

def delta_feats(fA, fB):
    d = {
        "ssa": fA["ssa"]-fB["ssa"],
        "ssd": fA["ssd"]-fB["ssd"],
        "sapm": -(fA["sapm"]-fB["sapm"]),   # lower better
        "tda": fA["tda"]-fB["tda"],
        "tdd": fA["tdd"]-fB["tdd"],
        "ko":  fA["ko"]-fB["ko"],
        "sub": fA["sub"]-fB["sub"],
        "dec": fA["dec"]-fB["dec"],
        "ranked_wins": fA["ranked_wins"]-fB["ranked_wins"],
        "champ_wins":  fA["champ_wins"]-fB["champ_wins"],
        "exp": fA["exp"]-fB["exp"],
        "aft": fA["aft"]-fB["aft"],
        "recency": fA["recency"]-fB["recency"],
        "opp_rank5": -(fA["opp_rank5"]-fB["opp_rank5"]),
        "same_div": 1.0 if fA["wc_snap"]==fB["wc_snap"] else 0.0,
        "wc_gap": float((fA["wc_snap"] or 0) - (fB["wc_snap"] or 0)),

        # recent-5
        "r_win5":   fA["r_win5"]   - fB["r_win5"],
        "r_fin5":   fA["r_fin5"]   - fB["r_fin5"],
        "r_ko5":    fA["r_ko5"]    - fB["r_ko5"],
        "r_sub5":   fA["r_sub5"]   - fB["r_sub5"],
        "r_title5": fA["r_title5"] - fB["r_title5"],
        "r_streak": fA["r_streak"] - fB["r_streak"],
        "r_ema5":   fA["r_ema5"]   - fB["r_ema5"],
        "r_avg_rank5": -(fA["r_avg_rank5"]-fB["r_avg_rank5"]),

        # recent-3
        "r_win3":   fA["r_win3"]   - fB["r_win3"],
        "r_streak3":fA["r_streak3"]- fB["r_streak3"],
        "r_ema3":   fA["r_ema3"]   - fB["r_ema3"],
        "r_avg_rank3": -(fA["r_avg_rank3"]-fB["r_avg_rank3"]),

        # power & durability
        "ko_rate":     fA.get("ko_rate",0.0)     - fB.get("ko_rate",0.0),
        "sub_rate":    fA.get("sub_rate",0.0)    - fB.get("sub_rate",0.0),
        "finish_rate": fA.get("finish_rate",0.0) - fB.get("finish_rate",0.0),
        "ko_loss_adv": fB.get("ko_loss_rate",0.0)  - fA.get("ko_loss_rate",0.0),
        "sub_loss_adv":fB.get("sub_loss_rate",0.0) - fA.get("sub_loss_rate",0.0),

        # decision quality (last 5)
        "dec_dom5":  fA.get("dec_dom5", 0.0)  - fB.get("dec_dom5", 0.0),
        "sd_tilt5":  fA.get("sd_tilt5", 0.0)  - fB.get("sd_tilt5", 0.0),
        "ud_rate5":  fA.get("ud_rate5", 0.0)  - fB.get("ud_rate5", 0.0),

        # NEW anthropometrics & stance
        "reach":   fA.get("reach",0.0)   - fB.get("reach",0.0),
        "height":  fA.get("height",0.0)  - fB.get("height",0.0),
        "sub_avg": fA.get("sub_avg",0.0) - fB.get("sub_avg",0.0),
    }

    # stance-engineered binaries
    sa = fA.get("stance","unknown"); sb = fB.get("stance","unknown")
    same_stance   = int(sa != "unknown" and sb != "unknown" and sa == sb)
    open_stance   = int({sa, sb} == {"orthodox", "southpaw"})
    southpaw_adv  = int(sa == "southpaw" and sb != "southpaw")
    switch_adv    = int(sa == "switch"   and sb != "switch")

    # --- Matchup: KO power vs chin (fractions 0..1) ---
    a_ko_r = (fA.get("ko_rate", 0.0) or 0.0) / 100.0
    b_ko_r = (fB.get("ko_rate", 0.0) or 0.0) / 100.0
    a_ko_l = (fA.get("ko_loss_rate", 0.0) or 0.0) / 100.0
    b_ko_l = (fB.get("ko_loss_rate", 0.0) or 0.0) / 100.0

    # "Danger" to A is B's power against A's chin, and vice versa
    dangerA = b_ko_r * a_ko_l
    dangerB = a_ko_r * b_ko_l

    d.update({
        "ko_powervschin": dangerB - dangerA,   # positive → favors A
        "ko_loss_recent5": (fB.get("ko_loss_recent5", 0.0) or 0.0)
                        - (fA.get("ko_loss_recent5", 0.0) or 0.0),
        "same_stance": same_stance,
        "open_stance": open_stance,
        "southpaw_adv": southpaw_adv,
        "switch_adv": switch_adv,
    })

    return d

# --------------------------- Decision scorecard dominance helpers ---------------------------
_SCORE_RE = re.compile(r'(\d{2})\s*[-–]\s*(\d{2})')

def _extract_score_pairs(raw):
    pairs = []
    if raw is None:
        return pairs
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for j in raw:
            a = j.get("a") or j.get("A") or j.get("total_a") or j.get("fighter_a") or j.get("left")
            b = j.get("b") or j.get("B") or j.get("total_b") or j.get("fighter_b") or j.get("right")
            try:
                a = int(a); b = int(b)
                pairs.append((a, b))
            except Exception:
                pass
        return pairs
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        for s in raw:
            m = _SCORE_RE.search(str(s))
            if m:
                pairs.append((int(m.group(1)), int(m.group(2))))
        return pairs
    if isinstance(raw, str):
        for m in _SCORE_RE.finditer(raw):
            pairs.append((int(m.group(1)), int(m.group(2))))
        return pairs
    return pairs

def _infer_rounds_from_pairs(pairs):
    for a, b in pairs:
        if max(a, b) >= 50:
            return 5
    return 3

def _fight_decision_dominance(fr):
    method = str(fr.get("method", "")).upper()
    if not ("DEC" in method or "DECISION" in method):
        return None, False, False

    res = str(fr.get("result", "")).lower()
    score_field = fr.get("scorecards") or fr.get("scorecard") or fr.get("judges") or fr.get("cards")
    pairs = _extract_score_pairs(score_field)

    is_ud = "UNAN" in method
    is_sd = "SPLIT" in method
    is_md = "MAJ" in method
    is_draw = "DRAW" in method or res == "draw"

    if not pairs:
        base = 0.0
        if is_draw:
            base = 0.0
        elif is_ud:
            base = 0.70
        elif is_md:
            base = 0.40
        elif is_sd:
            base = 0.25
        else:
            base = 0.50
        if res == "loss":
            base = -base
        return base, is_ud, is_sd

    R = _infer_rounds_from_pairs(pairs)
    margins = [abs(a - b) / float(R) for (a, b) in pairs]

    if res == "win":
        if is_sd and len(margins) >= 3:
            idx_min = margins.index(min(margins))
            signed = [m if i != idx_min else -m for i, m in enumerate(margins)]
        else:
            signed = margins[:]
    elif res == "loss":
        if is_sd and len(margins) >= 3:
            idx_min = margins.index(min(margins))
            signed = [-m if i != idx_min else m for i, m in enumerate(margins)]
        else:
            signed = [-m for m in margins]
    else:
        signed = [0.0 for _ in margins]

    dom = sum(signed) / len(signed) if signed else 0.0
    dom = max(-1.0, min(1.0, dom))
    return dom, is_ud, is_sd

def _decision_block(recs, N=5):
    decs = [fr for fr in recs if fr.get("year", 0) > 0 and ("DEC" in str(fr.get("method","")).upper() or "DECISION" in str(fr.get("method","")).upper())]
    decs = sorted(enumerate(decs), key=lambda t: (t[1]["year"], t[0]))[-N:]
    decs = [fr for _, fr in decs]

    if not decs:
        return dict(dec_dom5=0.0, sd_tilt5=0.0, ud_rate5=0.0)

    doms = []
    sd_w = sd_l = sd_tot = 0
    ud_w = dec_wins = 0

    for fr in decs:
        dom, is_ud, is_sd = _fight_decision_dominance(fr)
        if dom is not None:
            doms.append(dom)
        res = str(fr.get("result","")).lower()
        if "win" in res:
            dec_wins += 1
            if is_ud:
                ud_w += 1
            if is_sd:
                sd_w += 1; sd_tot += 1
        elif "loss" in res:
            if is_sd:
                sd_l += 1; sd_tot += 1

    dec_dom5 = statistics.mean(doms) if doms else 0.0
    sd_tilt5 = ((sd_w - sd_l) / sd_tot) if sd_tot > 0 else 0.0
    ud_rate5 = (ud_w / dec_wins) if dec_wins > 0 else 0.0
    return dict(dec_dom5=dec_dom5, sd_tilt5=sd_tilt5, ud_rate5=ud_rate5)

# --------------------------- Percent fallback helpers ---------------------------
PCT_KEYS = {
    "ssa","ssd","tda","tdd","ko","sub","dec",
    "r_win5","r_fin5","r_ko5","r_sub5","r_win3",
    # new % features
    "ko_rate","sub_rate","finish_rate","ko_loss_adv","sub_loss_adv"
}

def _percent01_fallback(X: dict) -> dict:
    Z = dict(X)
    for k in PCT_KEYS:
        if k in Z:
            Z[k] = Z[k] / 100.0
    return Z

# --------------------------- Math helpers ---------------------------
def logit(p): return math.log(p/(1-p))
def sigmoid(z): return 1/(1+math.exp(-z))
def elo_prior(eloA, eloB, K=ELO_K): return 1/(1 + 10**(-K*(eloA-eloB)))

def elo_confidence_weight(dA, dB):
    fa = float(dA.get("total_fights", 0) or 0)
    fb = float(dB.get("total_fights", 0) or 0)
    c_f = max(0.0, min(1.0, min(fa, fb)/12.0))
    ra = recency_index(dA.get("fight_records", []))
    rb = recency_index(dB.get("fight_records", []))
    c_r = min(ra, rb)
    base = 0.6*c_f + 0.4*c_r
    return W_ELO * max(0.1, min(1.0, base))

def dynamic_elo_weight(
    dA, dB, na: str, nb: str, base: float,
    rank_meta: Optional[dict],
    wmin: Optional[float], wmax: Optional[float],
    eloA: float, eloB: float,
) -> tuple[float, dict]:
    """
    Return the share for Elo in [wmin, wmax], driven by CONFIDENCE that Elo should matter:
      • rank_quality: higher for top-ranked fighters (we trust Elo more there)
      • c_fights, c_recency: more data → trust Elo more
      • gap_pct: large Elo gap → trust Elo more
    """
    # fights / recency confidence in [0,1]
    fa = float(dA.get("total_fights") or 0.0)
    fb = float(dB.get("total_fights") or 0.0)
    c_fights   = max(0.0, min(1.0, min(fa, fb) / 12.0))
    ra = recency_index(dA.get("fight_records", []))
    rb = recency_index(dB.get("fight_records", []))
    c_recency  = min(ra, rb)

    # rank_quality: 1.0 for #1 fighters, 0.0 for bottom of division
    if rank_meta and (na in rank_meta) and (nb in rank_meta):
        rank_pct = max(rank_meta[na]["rank_pct"], rank_meta[nb]["rank_pct"])  # 0 best … 1 worst
        rank_quality = 1.0 - rank_pct
    else:
        rank_quality = 0.5

    # Elo gap strength (≈1 for 600+ Elo diff)
    gap_pct = max(0.0, min(1.0, abs(float(eloA - eloB)) / 600.0))

    # CONFIDENCE that Elo should carry more weight
    confidence = 0.40*rank_quality + 0.30*c_fights + 0.20*c_recency + 0.10*gap_pct

    if wmin is None: wmin = max(0.0, min(1.0, base * 0.75))  # e.g., base=0.20 → 0.15
    if wmax is None: wmax = 0.85
    w = wmin + (wmax - wmin) * confidence
    w = max(0.0, min(0.95, w))

    dbg = {
        "rank_quality": rank_quality, "c_fights": c_fights, "c_recency": c_recency,
        "gap_pct": gap_pct, "confidence": confidence, "wmin": wmin, "wmax": wmax
    }
    return w, dbg

def prob_to_decimal(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return 1.0/p
def decimal_to_american(d):
    return int(round((d-1)*100)) if d>=2.0 else int(round(-100/(d-1)))
def prob_to_american(p): return decimal_to_american(prob_to_decimal(p))
def apply_overround(p, overround=0.04):
    m = 1.0 + max(0.0, overround)
    return p*m, (1-p)*m

def load_rank_meta(path):
    by_wc = json.load(open(path, "r", encoding="utf-8"))
    meta = {}
    for wc, rows in by_wc.items():
        sorted_rows = sorted(rows, key=lambda r: float(r.get("elo", 0.0)), reverse=True)
        n = len(sorted_rows) if sorted_rows else 1
        for idx, r in enumerate(sorted_rows):
            nm = normalize_name(r["name"])
            rank = idx + 1
            pct = (rank - 1) / (n - 1) if n > 1 else 0.0
            meta[nm] = {"wc": int(wc), "rank": rank, "n_in_wc": n, "rank_pct": pct}
    return meta

def _rate(num, denom):
    try:
        n = float(num or 0.0); d = float(denom or 0.0)
    except Exception:
        return 0.0
    return (n / d) if d > 0 else 0.0

# --------------------------- Weights + scaler plumbing ---------------------------
def _map_keys_to_short(d):
    out = {}
    for k, v in d.items():
        out[TRAIN_TO_SHORT.get(k, k)] = v
    return out

def _standardize_if_available(X_short: dict, scaler: Optional[dict]) -> dict:
    if not scaler:
        return X_short
    means_raw = scaler.get("means") or scaler.get("mean") or {}
    stds_raw  = scaler.get("stds")  or scaler.get("std")  or {}
    means = _map_keys_to_short(means_raw)
    stds  = _map_keys_to_short(stds_raw)
    Z = {}
    for k, v in X_short.items():
        m = means.get(k, 0.0)
        s = stds.get(k, 1.0)
        try:
            s_val = float(s)
        except Exception:
            s_val = 1.0
        Z[k] = (v - m) if abs(s_val) < 1e-12 else (v - m) / s_val
    return Z

def _select_weights_for_available_features(B_learned: dict, X_keys: set) -> dict:
    return {k: v for k, v in B_learned.items() if k in X_keys}

# --------------------------- Runtime overrides ---------------------------
def _parse_beta_overrides(s: Optional[str]) -> Dict[str, float]:
    if not s: return {}
    out = {}
    parts = [p for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except Exception:
            pass
    return out

def _parse_zero_list(s: Optional[str]) -> set:
    if not s: return set()
    return {p.strip() for p in s.split(",") if p.strip()}

# --------------------------- RF helpers ---------------------------
def _load_rf_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RF model not found: {path}")
    # Prefer joblib if available
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    # Fallback to pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def _safe_logit(p):
    p = max(1e-9, min(1-1e-9, float(p)))
    return math.log(p/(1-p))

def _build_training_row_for_rf(X_short: dict,
                               eloA: float, eloB: float,
                               wcA: Optional[int], wcB: Optional[int],
                               is_title: bool) -> Dict[str, float]:
    """
    Create a dict keyed by *training* column names that your RF likely expects,
    based on materialize_training.py (elo_diff, same_division, wc_gap + Δ cols).
    """
    # meta/prior-esque
    T = {
        "elo_diff": float(eloA - eloB),
        "same_division": 1.0 if (wcA is not None and wcB is not None and wcA == wcB) else 0.0,
        "wc_gap": float((wcA or 0) - (wcB or 0)),
        "is_title": 1.0 if is_title else 0.0,
    }
    # List of Δ columns from materialize_training.py (including new ones)
    delta_cols = [
        "d_sig_strike_acc", "d_sig_strike_def", "d_sapm_neg",
        "d_td_acc", "d_td_def",
        "d_ko_pct", "d_sub_pct", "d_dec_pct",
        "d_ranked_wins", "d_championship_wins",
        "d_total_fights", "d_avg_fight_time_min",
        "d_recency_index",
        "d_recent_win_pct5", "d_recent_finish_pct5", "d_recent_ko_pct5", "d_recent_sub_pct5",
        "d_recent_title_bouts5", "d_recent_streak", "d_recent_ema_form5",
        "d_opp_rank5_neg", "d_recent_avg_rank5_neg",
        # optional anthropometrics & stance
        "d_reach_in", "d_height_in", "d_sub_avg",
        "same_stance", "open_stance", "southpaw_adv", "switch_adv",
    ]
    for tn in delta_cols:
        sn = TRAIN_TO_SHORT.get(tn, None)
        if sn is None:
            continue
        T[tn] = float(X_short.get(sn, 0.0))
    return T

# --------------------------- Core prediction ---------------------------
def predict(nameA, nameB, rankings, stats, is_title=False, overround=0.0,
            weights_path=None, welo_override=None, k_override=None,
            stats_temp: float | None = None, clip_z: float | None = None, assume_pct01: bool = False,
            beta_overrides: Optional[Dict[str, float]] = None, zero_features: Optional[set] = None,
            rank_meta: Optional[dict] = None, welo_min: Optional[float] = None,
            welo_max: Optional[float] = None, inspect_blend: bool = False,
            combine_override: Optional[str] = None,
            elo_temp: float | None = None, elo_cap: float | None = None,
            rf_model_path: Optional[str] = None, w_rf: Optional[float] = None):
    learned = load_weights(weights_path) if weights_path else None
    global_cfg = learned.get("config", {}) if learned else {}

    stats_temp_eff = float(stats_temp) if stats_temp is not None else float(global_cfg.get("stats_temp", 1.0))
    clip_z_eff     = float(clip_z)     if clip_z is not None     else global_cfg.get("clip_z", None)
    if clip_z_eff is not None:
        try: clip_z_eff = float(clip_z_eff)
        except Exception: clip_z_eff = None

    ELO_K_eff = float(k_override if k_override is not None else global_cfg.get("ELO_K", ELO_K))
    W_ELO_eff = float(welo_override if welo_override is not None else global_cfg.get("W_ELO", W_ELO))
    if welo_min is None: welo_min = global_cfg.get("W_ELO_MIN", None)
    if welo_max is None: welo_max = global_cfg.get("W_ELO_MAX", None)

    # Elo shaping params
    ELO_TEMP_eff = float(elo_temp) if elo_temp is not None else float(global_cfg.get("elo_temp", ELO_TEMP_DEFAULT))
    ELO_CAP_eff  = float(elo_cap)  if elo_cap  is not None else float(global_cfg.get("elo_cap", ELO_CAP_DEFAULT))
    try:
        ELO_CAP_eff = None if (ELO_CAP_eff is None) else float(ELO_CAP_eff)
    except Exception:
        ELO_CAP_eff = ELO_CAP_DEFAULT

    scaler = learned.get("scaler") if learned else None
    used_scaler = bool(scaler)

    B0_eff = float(learned.get("B0", B0)) if learned else B0
    B_learned = _map_keys_to_short(learned.get("B", {})) if learned else {}
    B_eff_full = B_learned if B_learned else dict(B_DEFAULT)

    # --- overrides & mutes ---
    overrides = _map_keys_to_short(beta_overrides or {})
    for k, v in overrides.items():
        B_eff_full[k] = float(v)
    for k in (zero_features or set()):
        kk = TRAIN_TO_SHORT.get(k, k)
        if kk in B_eff_full:
            del B_eff_full[kk]

    # --- Fetch fighters ---
    na, nb = normalize_name(nameA), normalize_name(nameB)
    if na not in rankings or nb not in rankings:
        missing = [n for n in [nameA, nameB] if normalize_name(n) not in rankings]
        raise KeyError(f"Missing from elo_rankings_by_weight.json: {missing}")

    (wcA, eloA), (wcB, eloB) = rankings[na], rankings[nb]
    dA, dB = stats.get(na), stats.get(nb)
    if dA is None or dB is None:
        missing = [n for n in [nameA, nameB] if normalize_name(n) not in stats]
        raise KeyError(f"Missing from fighter_stats.json: {missing}")

    # --- Build features (per-side, then deltas) ---
    fA, fB = feature_vector(dA), feature_vector(dB)
    X_raw = delta_feats(fA, fB)
    X0 = _percent01_fallback(X_raw) if (assume_pct01 and not used_scaler) else dict(X_raw)

    X_keys = set(X0.keys())
    B_eff = _select_weights_for_available_features(B_eff_full, X_keys)

    X_std = _standardize_if_available(X0, scaler)
    X_weighted = {k: X_std[k] for k in B_eff.keys()}

    # --- GLM Stats logit (temperature affects stats only) ---
    z_stats_glm = B0_eff + sum(B_eff[k]*X_weighted[k] for k in B_eff)
    if is_title and ("aft" in X_raw):
        z_stats_glm += 0.05 * X_raw["aft"]
    z_stats_glm *= stats_temp_eff

    if (abs(z_stats_glm) > 10.0) and (not used_scaler):
        print("[warn] Very large stats logit and no scaler found. "
              "Consider --assume_pct01 or ensure your model_weights.json includes a scaler.",
              file=sys.stderr)

    # --- Optional Random Forest stats model (prob -> logit) ---
    z_stats_rf = None
    rf_meta = None
    if rf_model_path:
        if np is None:
            print("[warn] NumPy not available; skipping RF model.", file=sys.stderr)
        elif not os.path.exists(rf_model_path):
            print(f"[warn] RF model not found at {rf_model_path}; skipping RF.", file=sys.stderr)
        else:
            rf_clf = _load_rf_model(rf_model_path)
            T = _build_training_row_for_rf(X0, eloA, eloB, wcA, wcB, is_title)
            if hasattr(rf_clf, "feature_names_in_"):
                feat_names: List[str] = [str(x) for x in rf_clf.feature_names_in_]
            else:
                # fallback: use keys from T in deterministic order
                feat_names = sorted(T.keys())
            X_vec = np.array([[float(T.get(k, 0.0)) for k in feat_names]], dtype=float)
            try:
                proba = rf_clf.predict_proba(X_vec)[0]
                # class index for "1" (A wins) if available
                if hasattr(rf_clf, "classes_") and 1 in getattr(rf_clf, "classes_"):
                    idx = list(rf_clf.classes_).index(1)
                else:
                    idx = 1 if len(proba) > 1 else 0
                p_rf = float(proba[idx])
            except Exception:
                # Some pipelines expose decision_function; fall back to that if needed
                if hasattr(rf_clf, "decision_function"):
                    z_raw = float(rf_clf.decision_function(X_vec)[0])
                    # squish to prob assuming symmetric link
                    p_rf = 1/(1+math.exp(-z_raw))
                else:
                    raise
            z_stats_rf = _safe_logit(p_rf)
            rf_meta = {"feature_names": feat_names, "row": T, "p_rf": p_rf}

    # --- Blend GLM stats with RF stats (if provided) ---
    if z_stats_rf is not None:
        w_rf_eff = RF_STATS_WEIGHT_DEFAULT if (w_rf is None) else max(0.0, min(1.0, float(w_rf)))
        z_stats = (1.0 - w_rf_eff) * z_stats_glm + w_rf_eff * z_stats_rf
        stats_components = {"z_stats_glm": z_stats_glm, "z_stats_rf": z_stats_rf, "w_rf": w_rf_eff}
    else:
        z_stats = z_stats_glm
        stats_components = {"z_stats_glm": z_stats_glm, "z_stats_rf": None, "w_rf": 0.0}

    # --- Elo prior (offset-style), then shape (temp + cap) ---
    p_elo = 1/(1 + 10**(-ELO_K_eff*(eloA-eloB)))
    z_elo_raw = logit(p_elo)
    z_elo_shaped, elo_scale_used = shape_elo_logit(z_elo_raw, ELO_TEMP_eff, ELO_CAP_eff)

    # Dynamic weights (for display and for 'blend' mode; ignored in 'additive')
    if USE_DYNAMIC_ELO:
        w_elo, blend_dbg = dynamic_elo_weight(
    dA, dB, na, nb, W_ELO_eff, rank_meta, welo_min, welo_max, eloA, eloB)
    else:
        w_elo, blend_dbg = W_ELO_eff, None
    w_stats = 1.0 - w_elo

    # Decide how to combine
    combine_cfg = global_cfg.get("combine") or COMBINE_DEFAULT
    combine = combine_override or combine_cfg

    if combine == "additive":
        z = z_elo_shaped + z_stats
        elo_scale_for_print = 1.0
    elif combine == "additive_dyn":
        z = (w_elo * z_elo_shaped) + z_stats
        elo_scale_for_print = w_elo
    else:
        z = w_elo * z_elo_shaped + w_stats * z_stats
        elo_scale_for_print = w_elo

    # Optional clipping of the final logit
    if clip_z_eff is not None:
        cz = float(clip_z_eff)
        if z >  cz: z =  cz
        if z < -cz: z = -cz

    pA = sigmoid(z); pB = 1.0 - pA

    fair = {
        "A_prob": pA, "B_prob": pB,
        "A_decimal": prob_to_decimal(pA), "B_decimal": prob_to_decimal(pB),
        "A_american": prob_to_american(pA), "B_american": prob_to_american(pB),
    }
    book = None
    if overround > 0:
        pA_imp, pB_imp = apply_overround(pA, overround)
        book = {
            "A_implied_prob": pA_imp, "B_implied_prob": pB_imp,
            "A_decimal": 1.0/pA_imp, "B_decimal": 1.0/pB_imp,
            "A_american": decimal_to_american(1.0/pA_imp),
            "B_american": decimal_to_american(1.0/pB_imp),
            "overround_sum": pA_imp + pB_imp,
        }

    components = {
        "z_elo": z_elo_shaped,
        "z_elo_raw": z_elo_raw,
        "elo_scale": elo_scale_for_print,
        "z_stats": z_stats,
        "z_final": z,
        "combine": combine,
        **stats_components
    }

    return {
        "fighters": {"A": nameA, "B": nameB},
        "elos": {"A": eloA, "B": eloB},
        "prior_pA": p_elo,
        "weights": {"w_elo": w_elo, "w_stats": w_stats},
        "components": components,
        "features_used": X_raw,
        "features_used_std": X_weighted,
        "per_side_feats": {"A": fA, "B": fB},
        "fair": fair,
        "book": book,
        "config_used": {
            "ELO_K": ELO_K_eff, "W_ELO": W_ELO_eff,
            "stats_temp": stats_temp_eff, "clip_z": clip_z_eff,
            "elo_temp": ELO_TEMP_eff, "elo_cap": ELO_CAP_eff,
            "used_scaler": used_scaler, "assume_pct01": assume_pct01,
            "combine": combine
        },
        "betas": B_eff,
        "intercept": B0_eff,
        "weights_path": weights_path,
        "blend_debug": blend_dbg,
        "rank_meta_used": bool(rank_meta),
        "inspect_blend": bool(inspect_blend),
        "elo_scale_used": elo_scale_used,
        "rf_meta": rf_meta,
    }

# --------------------------- Pretty output helpers ---------------------------
def fmt_american(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)
def pct(p: float) -> str:
    return f"{p*100:.2f}%"

_FRIENDLY = {
    "ssa": "Sig. strike ACC",
    "ssd": "Sig. strike DEF",
    "sapm": "Strikes absorbed/min (lower better)",
    "tda": "TD accuracy",
    "tdd": "TD defense",
    "ko": "KO% profile",
    "sub": "SUB% profile",
    "dec": "DEC% profile",
    "ranked_wins": "Ranked wins",
    "champ_wins": "Championship wins",
    "exp": "Total fights",
    "aft": "Avg fight time (min)",
    "recency": "Recency index",
    "opp_rank5": "Recent opp. quality",
    "same_div": "Same division",
    "wc_gap": "Weight-class gap",

    # recent-5
    "r_win5": "Win% (last 5)",
    "r_fin5": "Finish% (last 5)",
    "r_ko5": "KO% (last 5 wins)",
    "r_sub5": "SUB% (last 5 wins)",
    "r_title5": "Title bouts (last 5)",
    "r_streak": "Current streak (±)",
    "r_ema5": "EMA form (last 5)",
    "r_avg_rank5": "Opp avg rank (last 5, lower=tougher)",

    # recent-3
    "r_win3": "Win% (last 3)",
    "r_streak3": "Current streak (last 3, ±)",
    "r_ema3": "EMA form (last 3)",
    "r_avg_rank3": "Opp avg rank (last 3, lower=tougher)",

    # power/durability
    "ko_rate": "KO win rate",
    "sub_rate": "SUB win rate",
    "finish_rate": "Finish win rate",
    "ko_loss_adv": "KO durability edge",
    "sub_loss_adv": "SUB durability edge",

    "dec_dom5": "Decision dominance (last 5)",
    "sd_tilt5": "Split-decision tilt (last 5)",
    "ud_rate5": "UD rate (decision wins, last 5)",

    # NEW anthropometrics / stance
    "reach": "Reach edge (in)",
    "height": "Height edge (in)",
    "sub_avg": "Sub attempts /15 edge",
    "same_stance": "Same stance",
    "open_stance": "Open stance (O vs S)",
    "southpaw_adv": "Southpaw advantage",
    "switch_adv": "Switch stance advantage",

    "ko_loss_recent5": "Recent KO-loss fragility (last 5)",
    "ko_powervschin": "KO power vs chin (matchup edge)",
}

def _top_drivers(features_std: dict, betas: dict, features_raw: dict | None = None, topn=6):
    contrib = {k: betas.get(k, 0.0) * features_std.get(k, 0.0) for k in betas}
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topn]
    lines = []
    for k, v in items:
        lbl = _FRIENDLY.get(k, k)
        arrow = "→ favors A" if v > 0 else "→ favors B"
        raw = None if features_raw is None else features_raw.get(k, None)
        raw_txt = "" if raw is None else f"  Δ={raw:.3f}"
        lines.append(f"• {lbl}:{raw_txt}  ({'+' if v>=0 else ''}{v:.3f} logit) {arrow}")
    return lines

def pretty_print(result: dict, topn: int = 6):
    A = result["fighters"]["A"]; Bn = result["fighters"]["B"]
    eloA = result["elos"]["A"];   eloB = result["elos"]["B"]
    prior = result["prior_pA"];   pA = result["fair"]["A_prob"]; pB = result["fair"]["B_prob"]
    fairA_dec = result["fair"]["A_decimal"]; fairB_dec = result["fair"]["B_decimal"]
    fairA_amer = fmt_american(result["fair"]["A_american"])
    fairB_amer = fmt_american(result["fair"]["B_american"])
    delta_pp = (pA - prior) * 100
    w = result.get("weights", {}); c = result.get("components", {})
    mode = (c or {}).get("combine", "blend")
    elo_scale = (c or {}).get("elo_scale", 1.0)
    cfg = result.get("config_used", {})

    print("\n================= UFC Fight Prediction =================")
    print(f"Matchup: {A} vs {Bn}")
    print(f"Elo: {eloA:.1f} vs {eloB:.1f}  (Δ = {eloA-eloB:+.1f} for {A})")
    print(f"Elo prior (A): {pct(prior)}")
    print(f"Model win probability: {A} {pct(pA)} | {Bn} {pct(pB)}  (stats shift: {delta_pp:+.2f} pp)")

    print("\nBlend / Config:")
    if w and c:
        if mode == "additive":
            print("  Combine: additive (z = z_elo + z_stats)")
        elif mode == "additive_dyn":
            print(f"  Combine: additive_dyn (z = {elo_scale:.2f}·z_elo + z_stats)")
            print(f"  Dyn weights → Elo: {w['w_elo']:.2f} | Stats: {w['w_stats']:.2f}")
        else:
            print("  Combine: blend (z = w_elo·z_elo + (1-w_elo)·z_stats)")
            print(f"  Weights → Elo: {w['w_elo']:.2f} | Stats: {w['w_stats']:.2f}")
        # Show components with GLM and (if present) RF breakdown
        z_glm = c.get("z_stats_glm")
        z_rf  = c.get("z_stats_rf")
        if z_rf is not None:
            print(f"  Stats logit parts → GLM: {z_glm:+.3f} | RF: {z_rf:+.3f} (w_rf={c.get('w_rf'):.2f}) | Combined: {c.get('z_stats'):+.3f}")
            # Optional: also show RF probability for A
            rf_meta = result.get("rf_meta")
            if rf_meta and isinstance(rf_meta.get("p_rf"), float):
                print(f"  RF(A) prob: {rf_meta['p_rf']:.3f}")
        else:
            print(f"  Stats logit → {c.get('z_stats'):+.3f}")
        print(f"  Logit parts → Elo: {c['z_elo']:+.3f} | Final: {c['z_final']:+.3f}")
    if cfg:
        print(f"  ELO_K: {cfg.get('ELO_K')}  |  W_ELO: {cfg.get('W_ELO')}"
              f"  | stats_temp: {cfg.get('stats_temp')}  | clip_z: {cfg.get('clip_z')}"
              f"  | elo_temp: {cfg.get('elo_temp')}  | elo_cap: {cfg.get('elo_cap')}"
              f"  | used_scaler: {cfg.get('used_scaler')}  | assume_pct01: {cfg.get('assume_pct01')}")

    if result.get("blend_debug") and result.get("inspect_blend"):
        bd = result["blend_debug"]
        print("  Dynamic blend detail → "
              f"rank_pct: {bd.get('rank_pct'):.2f}, "
              f"fights_conf: {bd.get('c_fights'):.2f}, "
              f"recency_conf: {bd.get('c_recency'):.2f}, "
              f"uncertainty: {bd.get('uncertainty'):.2f}, "
              f"range: [{bd.get('wmin'):.2f}, {bd.get('wmax'):.2f}]")

    print("\nFair odds (no vig):")
    print(f"  {A}: {fairA_amer}  (dec {fairA_dec:.3f})")
    print(f"  {Bn}: {fairB_amer}  (dec {fairB_dec:.3f})")

    if result.get("book"):
        b = result["book"]
        print("\nBook-style odds (with margin):")
        print(f"  Overround sum: {b['overround_sum']:.3f}")
        print(f"  {A}: {fmt_american(b['A_american'])}  (dec {b['A_decimal']:.3f}, implied {pct(b['A_implied_prob'])})")
        print(f"  {Bn}: {fmt_american(b['B_american'])}  (dec {b['B_decimal']:.3f}, implied {pct(b['B_implied_prob'])})")

    betas = result.get("betas", {})
    feats_std = result.get("features_used_std", result.get("features_used", {}))
    feats_raw = result.get("features_used", {})
    print("\nTop stat drivers (GLM):")
    for line in _top_drivers(feats_std, betas, feats_raw, topn=topn):
        print(" ", line)
    print("========================================================\n")

# --------------------------- Full profile table ---------------------------
def print_full_profile(result: dict, order: str = "contrib"):
    Aname = result["fighters"]["A"]; Bname = result["fighters"]["B"]
    betas = result.get("betas", {})
    X_raw = result.get("features_used", {})
    X_std = result.get("features_used_std", {})
    side = result.get("per_side_feats", {})
    fA = side.get("A", {})
    fB = side.get("B", {})

    rows = []
    for k, beta in betas.items():
        a = fA.get(k, None)
        b = fB.get(k, None)
        raw = X_raw.get(k, None)
        z   = X_std.get(k, None)
        contrib = (beta * z) if (beta is not None and z is not None) else None
        fav = "A" if (contrib is not None and contrib > 0) else ("B" if (contrib is not None and contrib < 0) else "-")
        rows.append((k, a, b, raw, z, beta, contrib, fav))

    if order == "logical":
        order_list = [
            "ssa","ssd","sapm","tda","tdd","ko","sub","dec",
            "recency","opp_rank5","r_avg_rank5",
            "r_win5","r_fin5","r_ko5","r_sub5","r_streak","r_ema5","r_title5",
            "r_win3","r_avg_rank3","r_streak3","r_ema3",
            "same_div","wc_gap","ranked_wins","champ_wins","exp","aft",
            # power/durability
            "ko_rate","sub_rate","finish_rate","ko_loss_adv","sub_loss_adv",
            # anthropometrics & stance
            "reach","height","sub_avg","same_stance","open_stance","southpaw_adv","switch_adv",
        ]
        pos = {k:i for i,k in enumerate(order_list)}
        rows.sort(key=lambda r: pos.get(r[0], 999))
    else:
        rows.sort(key=lambda r: 0.0 if r[6] is None else -abs(r[6]))

    print("\nFull statistical profile (per-feature):")
    hdr = f"{'Feature':<30}{Aname:>12}{Bname:>12}{'Δ raw':>12}{'zΔ':>10}{'β':>10}{'β·zΔ':>12}{'Fav':>6}"
    print(hdr)
    print("-" * len(hdr))

    def fmt(x, w):
        if x is None: return f"{'-':>{w}}"
        try:
            return f"{float(x):>{w}.3f}"
        except Exception:
            return f"{str(x):>{w}}"

    for k, a, b, raw, z, beta, contrib, fav in rows:
        lbl = _FRIENDLY.get(k, k)
        print(f"{lbl:<30}{fmt(a,12)}{fmt(b,12)}{fmt(raw,12)}{fmt(z,10)}{fmt(beta,10)}{fmt(contrib,12)}{fav:>6}")

# --------------------------- CLI helpers ---------------------------
def default_data(path):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), path))

def ensure_in(mapping, raw_label, mapping_name):
    key = normalize_name(raw_label)
    if key in mapping:
        return key
    choices = list(mapping.keys())
    cand = get_close_matches(key, choices, n=6, cutoff=0.7)
    msg = f"{mapping_name} name not found: '{raw_label}'."
    if cand:
        msg += f" Did you mean: {', '.join(c.title() for c in cand)}?"
    print(msg, file=sys.stderr)
    sys.exit(2)

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict UFC fight using Elo + stats (GLM + optional RF)")
    ap.add_argument("--elo_json",   default=default_data("../data/elo_rankings_by_weight.json"),
                    help="Path to elo_ranking_by_weight.json")
    ap.add_argument("--stats_json", default=default_data("../data/fighter_stats.json"),
                    help="Path to fighter_stats.json")
    ap.add_argument("--weights_json", default=default_data("../data/model_weights.json"),
                    help="Path to model_weights.json (learned B0/B and optional scaler)")
    ap.add_argument("--A", help="Fighter A name")
    ap.add_argument("--B", help="Fighter B name")
    ap.add_argument("A_pos", nargs="?", help="Fighter A name (positional)")
    ap.add_argument("B_pos", nargs="?", help="Fighter B name (positional)")
    ap.add_argument("--title", action="store_true", help="Set if 5-round title bout")
    ap.add_argument("--overround", type=float, default=0.0, help="Book margin, e.g., 0.05 for 5%%")
    ap.add_argument("--welo", type=float, default=None, help="Override W_ELO at runtime (e.g., 0.75)")
    ap.add_argument("--k",    type=float, default=None, help="Override ELO_K at runtime (e.g., 0.0025)")
    ap.add_argument("--stats_temp", type=float, default=None,
                    help="Multiply the GLM stats logit (e.g., 0.20 to shrink)")
    ap.add_argument("--clip_z", type=float, default=None,
                    help="Clip final blended logit to ±this value (e.g., 4.0)")
    ap.add_argument("--assume_pct01", action="store_true",
                    help="If no scaler in weights, assume % deltas are 0..100 and scale to 0..1")
    ap.add_argument("--set_beta", type=str, default=None,
                    help='Comma list of beta overrides, e.g. "exp=0,tdd=0.30,ssa=0.006,r_ema3=0.45,r_win3=0.02"')
    ap.add_argument("--zero", type=str, default=None,
                    help='Comma list of GLM features to drop, e.g. "exp,ko"')
    # New printing controls
    ap.add_argument("--print_profile", action="store_true",
                    help="Print full per-feature profile table (A, B, Δ, zΔ, β, β·zΔ)")
    ap.add_argument("--profile_order", choices=["contrib","logical"], default="contrib",
                    help="Order for full profile table (default: contrib)")
    ap.add_argument("--topn", type=int, default=6,
                    help="How many items to show in the Top drivers list (default: 6)")
    # Dynamic blend controls
    ap.add_argument("--welo_min", type=float, default=None,
                    help="Lower bound for Elo share when dynamic blend is on")
    ap.add_argument("--welo_max", type=float, default=None,
                    help="Upper bound for Elo share when dynamic blend is on")
    ap.add_argument("--inspect_blend", action="store_true",
                    help="Print rank/fights/recency components of the dynamic blend")
    # Combine + Elo shaping
    ap.add_argument("--combine", choices=["additive","additive_dyn","blend"], default=None,
                    help="Override combine mode (default from weights.json config or 'blend')")
    ap.add_argument("--elo_temp", type=float, default=None,
                    help="Multiply Elo logit before combining (e.g., 0.55 to soften Elo)")
    ap.add_argument("--elo_cap", type=float, default=None,
                    help="Clip Elo logit to ±this value before combining (e.g., 0.9)")

    # Random Forest integration (auto-on if ../data/rf_model.pkl exists)
    ap.add_argument("--rf_model", type=str, default=None,
                help="Path to a pickled/joblib sklearn classifier. If omitted, ../data/rf_model.pkl is used if present.")
    ap.add_argument("--w_rf", type=float, default=None,
                help=f"Weight (0..1) for blending RF stats logit with GLM stats logit (default {RF_STATS_WEIGHT_DEFAULT})")
    ap.add_argument("--no_rf", action="store_true",
                help="Disable RF blending even if a model file is found.")

    args = ap.parse_args()

    nameA = args.A or args.A_pos or (input("Enter Fighter A: ").strip())
    nameB = args.B or args.B_pos or (input("Enter Fighter B: ").strip())

    rankings = load_rankings(args.elo_json)
    stats    = load_stats(args.stats_json)
    rank_meta = load_rank_meta(args.elo_json)

    _ = ensure_in(rankings, nameA, "Elo")
    _ = ensure_in(rankings, nameB, "Elo")
    _ = ensure_in(stats, nameA, "Stats")
    _ = ensure_in(stats, nameB, "Stats")

    # Decide RF model path (auto-enable if default file exists)
    rf_path_default = default_data("../data/rf_model.pkl")
    rf_path = None
    if not args.no_rf:
        if args.rf_model:
            if os.path.exists(args.rf_model):
                rf_path = args.rf_model
            else:
                print(f"[warn] RF model not found at {args.rf_model}; using GLM-only.", file=sys.stderr)
        else:
            if os.path.exists(rf_path_default):
                rf_path = rf_path_default

    out = predict(
        nameA, nameB,
        rankings, stats,
        is_title=args.title,
        overround=args.overround,
        weights_path=args.weights_json,
        welo_override=args.welo,
        k_override=args.k,
        stats_temp=args.stats_temp,
        clip_z=args.clip_z,
        assume_pct01=args.assume_pct01,
        beta_overrides=_parse_beta_overrides(args.set_beta),
        zero_features=_parse_zero_list(args.zero),
        rank_meta=rank_meta,
        welo_min=args.welo_min,
        welo_max=args.welo_max,
        inspect_blend=args.inspect_blend,
        combine_override=args.combine,
        elo_temp=args.elo_temp,
        elo_cap=args.elo_cap,
        rf_model_path=rf_path,
        w_rf=args.w_rf
    )
    pretty_print(out, topn=args.topn)
    if args.print_profile:
        print_full_profile(out, order=args.profile_order)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
