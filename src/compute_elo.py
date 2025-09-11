#!/usr/bin/env python3
# compute_elo.py — Global Elo with opponent-Elo difficulty, rank-aware loss,
# annual decay, and win-streak bonus. Draw/NC do not affect streaks.
# NEW: (a) sorting-only recent-title tiebreak
# NEW (2025-08): enforce head-to-head ordering ONLY if the winner beat the opponent
#                within their last H2H_RECENT_WINDOW fights AND the two are within
#                ELO_PROX_THRESHOLD Elo.

import os
import json
import unicodedata
import re
from datetime import date
from typing import Dict, Any, List, Tuple, Optional
# DB logging (optional but enabled in main())
from EloDB import db, create_schema, upsert_fighter, start_elo_run, insert_row, norm as norm_db


# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
INPUT_FILE = os.path.join(DATA_DIR, "fighter_stats.json")
OUTPUT_JSON_DETAILED = os.path.join(DATA_DIR, "elo_ratings.json")
OUTPUT_JSON_SIMPLE   = os.path.join(DATA_DIR, "elo_rankings_by_weight.json")

# ─── Parameters ────────────────────────────────────────────────────────────────
BASE_ELO = 1500
STATIC_ELO_EXTERNAL = 1500  # opponents not in DB stay fixed here (no updates)

# Rank bonus on WINS: #1 → +100 down to #15 → +10 (linear). Champ “C” → +50
RANK_TOP_BONUS     = 100
RANK_BOTTOM_BONUS  = 10
CHAMP_BONUS        = 50

# Titles
TITLE_WIN_FLAT     = 25
UNDISPUTED_MULT    = 1.6   # multiplier ONLY on that fight (no additive)

# Win/Loss shaping
WIN_FLAT_BONUS     = 5     # +5 on every win
LOSS_STREAK_PER    = -20   # extra per loss after 2nd (penalty magnitude increases)

# Method multipliers (wins & losses)
METHOD_KO_SUB      = 1.5
METHOD_UD          = 1.1
METHOD_DEFAULT     = 1.0

# Scorecards: ±(avg judge margin × 5)
SCORECARD_FACTOR   = 5.0

# Recency bins (multiplies per-fight delta)
TODAY              = date(2025, 8, 11)
def recency_weight(fight_year: int) -> float:
    if not fight_year or fight_year <= 0:   return 1.0
    if fight_year <= 2016:                  return 0.40
    if 2017 <= fight_year <= 2019:          return 0.55
    if 2020 <= fight_year <= 2021:          return 0.65
    if fight_year == 2022:                  return 0.75
    if fight_year == 2023:                  return 0.90
    if fight_year == 2024:                  return 1.15
    if fight_year >= 2025:                  return 1.25
    return 1.0

# Annual decay of rating offset toward BASE_ELO between calendar years
ANNUAL_DECAY       = 0.75   # steep to reward activity

# Opponent-Elo difficulty (global sim)
GAMMA              = 0.25
OPP_MULT_MIN       = 0.80
OPP_MULT_MAX       = 1.25

# Division snapping (men’s bins)
DIVISION_BINS      = [125, 135, 145, 155, 170, 185, 205, 265]

# Sorting-only knobs
TITLE_SORT_BONUS   = 40    # add for ordering if last title win in-division was in 2024/2025
TITLE_SORT_SINCE   = 2024

# NEW constraint rule knobs
H2H_RECENT_WINDOW  = 3      # winner must have beaten loser within last N fights
ELO_PROX_THRESHOLD = 100.0  # and |ΔElo| must be ≤ this threshold to enforce winner-above-loser

MAX_CONSTRAINT_PASSES = 20

# Opponent field keys accepted inside fight_records
OPP_NAME_KEYS = (
    "opponent", "opponent_name", "opponent_canon", "opponent_canonical",
    "opp_name", "opp", "vs", "opponent_full_name"
)

# Potential per-fight weight-class keys
FIGHT_WC_KEYS      = ("fight_weight_class", "bout_weight_class", "weight_class", "wc", "division", "weight")

# ─── Canonicalization ──────────────────────────────────────────────────────────
def canonical_name(name: str) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[^a-z\s]", "", n)
    return re.sub(r"\s+", " ", n).strip()

# ─── Helpers ───────────────────────────────────────────────────────────────────
def method_multiplier(method: str) -> float:
    m = (method or "").upper()
    if "KO/TKO" in m or "KNOCKOUT" in m or m.strip() == "KO":
        return METHOD_KO_SUB
    if "SUB" in m:
        return METHOD_KO_SUB
    if "DECISION - UNANIMOUS" in m or "U-DEC" in m:
        return METHOD_UD
    return METHOD_DEFAULT

def rank_bonus(rank_value: Any) -> int:
    if rank_value in (None, "", "NR"):
        return 0
    if rank_value == "C":
        return CHAMP_BONUS
    try:
        r = int(rank_value)
    except Exception:
        return 0
    span = (RANK_TOP_BONUS - RANK_BOTTOM_BONUS) / 14.0
    val = round(RANK_TOP_BONUS - (r - 1) * span)
    return max(RANK_BOTTOM_BONUS, min(RANK_TOP_BONUS, val))

# rank-aware LOSS penalty magnitude (positive number; minus sign added later)
def loss_penalty_mag_by_rank(rank_value: Any) -> float:
    # NR ⇒ 25; C ⇒ 2; #1..#15 linearly 5..22
    if rank_value in (None, "", "NR"):
        return 25.0
    if rank_value == "C":
        return 2.0
    try:
        r = int(rank_value)
    except Exception:
        return 25.0
    r = max(1, min(15, r))
    return 5.0 + (r - 1) * ((22.0 - 5.0) / 14.0)

def avg_score_margin(scores_block: List[List[str]]) -> float:
    if not scores_block:
        return 0.0
    vals: List[int] = []
    for triple in scores_block:
        for s in triple:
            s = s.replace("–", "-").strip()
            parts = s.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]), int(parts[1])
                vals.append(abs(a - b))
    return (sum(vals) / len(vals)) if vals else 0.0

def signed_decision_margin_one(f: dict) -> float:
    """
    Signed judge-margin for a single fight if it was a decision.
    +margin for wins, -margin for losses; 0 if not a decision or no scores.
    """
    method = str(f.get("method", "")).upper()
    if "DECISION" not in method:
        return 0.0
    m = avg_score_margin(f.get("scores", []))
    if not m:
        return 0.0
    res = (f.get("result") or "").lower()
    if res == "win":
        return float(m)
    if res == "loss":
        return float(-m)
    return 0.0

def snap_division_lbs(wc: Optional[int]) -> Optional[int]:
    if wc is None: return None
    if wc < 125:   return 125
    if wc > 205:   return 265
    return min(DIVISION_BINS[:-1], key=lambda b: abs(b - wc))

def sort_fights_local(fights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Drops "next", "nc", and "draw"
    with_idx = list(enumerate(fights))
    with_idx = [t for t in with_idx if (t[1].get("result") not in ("next", "nc", "draw"))]
    with_idx = [t for t in with_idx if t[1].get("year", 0)]
    with_idx.sort(key=lambda t: (t[1].get("year", 0), t[0]))
    return [f for _, f in with_idx]

def get_most_recent_weight_class_from_records(fighter: Dict[str, Any]) -> Optional[int]:
    fights = sort_fights_local(fighter.get("fight_records", []))
    if fights:
        last = fights[-1]
        for k in FIGHT_WC_KEYS:
            if k in last and isinstance(last[k], (int, float)) and last[k]:
                return snap_division_lbs(int(last[k]))
    wc = fighter.get("weight_class")
    return snap_division_lbs(int(wc)) if isinstance(wc, (int, float)) and wc else None

def _debug_event_coverage(fighters: List[Dict[str, Any]]) -> None:
    total = 0
    with_opp = 0
    for ft in fighters:
        for fr in ft.get("fight_records", []):
            if fr.get("result") in ("win","loss") and fr.get("year", 0):
                total += 1
                if any(k in fr and isinstance(fr[k], str) and fr[k].strip() for k in OPP_NAME_KEYS):
                    with_opp += 1
    if total:
        pct = 100.0 * with_opp / total
        print(f"[info] fights usable for GLOBAL sim (have opponent name): {with_opp}/{total} ({pct:.1f}%)")
    else:
        print("[info] no fights found with year + (win/loss).")


def fight_key(year: int, name_a: str, name_b: str, seq: int) -> str:
    """Deterministic key per bout (no dates in data, so add a yearly sequence)."""
    a, b = sorted([canonical_name(name_a), canonical_name(name_b)])
    return f"{year}-{seq:03d}-{a}--{b}"

def _recency_index_from_records(recs: List[Dict[str, Any]]) -> float:
    yrs = [fr.get("year", 0) for fr in recs if fr.get("year",0)>0 and fr.get("result") in ("win","loss")]
    if not yrs: return 0.0
    y = max(yrs)
    if y <= 2016: return 0.0
    if y <= 2019: return 0.2
    if y <= 2021: return 0.35
    if y == 2022: return 0.5
    if y == 2023: return 0.75
    return 1.0

def _avg_opp_rank_last5(recs: List[Dict[str, Any]]) -> float:
    ranks = []
    for fr in recs:
        r = fr.get("rank")
        if isinstance(r, str) and r.isdigit(): ranks.append(int(r))
        elif isinstance(r, int): ranks.append(r)
    ranks = [x for x in ranks if 1 <= x <= 15][-5:]
    return sum(ranks)/len(ranks) if ranks else 20.0

def recent_block(recs: List[Dict[str, Any]], N=5) -> Dict[str, float]:
    last = [fr for fr in recs if fr.get("year",0)>0 and fr.get("result") in ("win","loss")]
    last = sorted(enumerate(last), key=lambda t: (t[1]["year"], t[0]))[-N:]
    last = [fr for _, fr in last]
    if not last:
        return dict(recent_win_pct5=0.0, recent_finish_pct5=0.0, recent_ko_pct5=0.0,
                    recent_sub_pct5=0.0, recent_avg_rank5=20.0, recent_title_bouts5=0.0,
                    recent_streak=0.0, recent_ema_form5=0.0)
    wins = sum(fr["result"]=="win" for fr in last)
    win_pct = 100.0*wins/len(last)
    wins_list = [fr for fr in last if fr["result"]=="win"]
    M = lambda fr: str(fr.get("method","")).upper()
    def _pct(cond): return 100.0*sum(cond(fr) for fr in wins_list)/len(wins_list) if wins_list else 0.0
    ko_pct = _pct(lambda fr: ("KO" in M(fr)) or ("TKO" in M(fr)))
    sub_pct= _pct(lambda fr: ("SUB" in M(fr)))
    finish_pct = _pct(lambda fr: ("KO" in M(fr)) or ("TKO" in M(fr)) or ("SUB" in M(fr)))
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
    ema = 0.0 if ema is None else ema
    return dict(recent_win_pct5=win_pct, recent_finish_pct5=finish_pct, recent_ko_pct5=ko_pct,
                recent_sub_pct5=sub_pct, recent_avg_rank5=avg_rank, recent_title_bouts5=float(title_bouts),
                recent_streak=float(streak), recent_ema_form5=float(ema))

def features_as_of_year(fighter_obj: Dict[str,Any], cutoff_year: int) -> Dict[str,float]:
    """Freeze features using fights strictly before `cutoff_year` (year-granular; no dates available)."""
    recs = [fr for fr in fighter_obj.get("fight_records", []) 
            if fr.get("year",0)>0 and fr["year"] < cutoff_year and fr.get("result") in ("win","loss")]

    # --- NEW: signed decision dominance over last 5 decision fights ---
    dec_only = [fr for fr in recs if "DECISION" in str(fr.get("method","")).upper()]
    dec_only = sorted(enumerate(dec_only), key=lambda t: (t[1]["year"], t[0]))[-5:]
    dec_only = [fr for _, fr in dec_only]
    if dec_only:
        margins = [signed_decision_margin_one(fr) for fr in dec_only]
        avg_dec_margin5 = float(sum(margins)/len(margins)) if margins else 0.0
    else:
        avg_dec_margin5 = 0.0
    # -----------------------------------------------------------------

    base = {
        "sig_strike_acc": float(fighter_obj.get("sig_strike_acc") or 0.0),
        "sig_strike_def": float(fighter_obj.get("sig_strike_def") or 0.0),
        "sapm": float(fighter_obj.get("sapm") or 0.0),
        "td_acc": float(fighter_obj.get("td_acc") or 0.0),
        "td_def": float(fighter_obj.get("td_def") or 0.0),
        "ko_pct": float(fighter_obj.get("ko_pct") or 0.0),
        "sub_pct": float(fighter_obj.get("sub_pct") or 0.0),
        "dec_pct": float(fighter_obj.get("dec_pct") or 0.0),
        "ranked_wins": float(fighter_obj.get("ranked_wins") or 0.0),
        "championship_wins": float(fighter_obj.get("championship_wins") or 0.0),
        "total_fights": float(fighter_obj.get("total_fights") or 0.0),
        "avg_fight_time_min": float((fighter_obj.get("avg_fight_time_sec") or 0.0))/60.0,
        "recency_index": _recency_index_from_records(recs),
        "avg_opp_rank_last5": _avg_opp_rank_last5(recs),
        "avg_dec_margin5": avg_dec_margin5,
    }
    base.update(recent_block(recs, N=5))
    return base

# ─── Win-streak bonus ─────────────────────────────────────────────────────────
def win_streak_bonus(consec_win_before: int) -> float:
    """
    Additive bonus BEFORE multipliers for WINs.
    3rd straight win: +10  (before=2), 4th:+20, 5th:+30, ...
    """
    return float(max(0, (consec_win_before - 1) * 5))

# ─── Per-fight delta (no opponent-Elo yet) ─────────────────────────────────────
def local_delta(result: str, method: str, opp_rank: Any, is_title: bool, scores: List[List[str]],
                year: int, consec_loss_before: int, consec_win_before: int,
                is_champion: bool, undisputed_trigger: bool) -> float:
    delta = 0.0
    m_upper = (method or "").upper()
    dq_win  = (result == "win"  and "DQ" in m_upper)
    dq_loss = (result == "loss" and "DQ" in m_upper)  # keep if you also want DQ losses to skip multipliers

    if result == "win":
        # Additives always allowed
        delta += rank_bonus(opp_rank)
        if is_title:
            delta += TITLE_WIN_FLAT
        delta += WIN_FLAT_BONUS
        delta += win_streak_bonus(consec_win_before)

        if dq_win:
            # DQ WIN: no multipliers, no scorecard margin
            return delta

        # Multipliers (non-DQ only)
        delta *= method_multiplier(method)
        delta *= recency_weight(year)
        if undisputed_trigger:
            delta *= UNDIDPUTED_MULT  # typo-guard resolved in __main__

        # Scorecards bonus/penalty for decisions
        if "DECISION" in m_upper:
            delta += avg_score_margin(scores) * SCORECARD_FACTOR

    elif result == "loss":
        base_mag = loss_penalty_mag_by_rank(opp_rank)
        next_consec_loss = consec_loss_before + 1
        if next_consec_loss >= 2:
            base_mag += (-LOSS_STREAK_PER) * (next_consec_loss - 2)
        delta = -base_mag

        if dq_loss:
            # DQ LOSS: no multipliers, no scorecard margin
            return delta

        # Multipliers (non-DQ only)
        delta *= method_multiplier(method)
        delta *= recency_weight(year)

        if "DECISION" in m_upper:
            delta += -avg_score_margin(scores) * SCORECARD_FACTOR

    return delta

# ─── Annual decay helper ───────────────────────────────────────────────────────
def apply_annual_decay(current_elo: float, current_year: int, last_year: Optional[int]) -> Tuple[float, int]:
    if last_year is None:
        return current_elo, current_year
    years_gap = max(0, int(current_year) - int(last_year))
    if years_gap == 0:
        return current_elo, current_year
    offset = current_elo - BASE_ELO
    decayed = BASE_ELO + offset * (ANNUAL_DECAY ** years_gap)
    return decayed, current_year

# ─── Local (fallback) Elo for a single fighter ────────────────────────────────
def compute_fighter_local_elo(fighter: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    fights = sort_fights_local(fighter.get("fight_records", []))
    elo = BASE_ELO
    trace: List[Dict[str, Any]] = []
    consec_loss = 0
    consec_win = 0
    is_champion = False
    last_year: Optional[int] = None

    for f in fights:
        year     = int(f.get("year", 0) or 0)
        result   = (f.get("result") or "").lower()
        method   = f.get("method") or ""
        opp_rk   = f.get("rank", "NR")
        is_title = bool(f.get("is_title", False))
        scores   = f.get("scores", [])

        elo, last_year = apply_annual_decay(elo, year, last_year)

        undisputed = (result == "win" and is_title and opp_rk == "C" and not is_champion)
        delta = local_delta(result, method, opp_rk, is_title, scores, year,
                            consec_loss, consec_win, is_champion, undisputed)
        if result == "win" and undisputed:
            delta *= UNDISPUTED_MULT

        elo += delta

        if result == "win":
            consec_win  += 1
            consec_loss  = 0
            if undisputed:
                is_champion = True
        elif result == "loss":
            consec_loss += 1
            consec_win   = 0
            is_champion  = False

        trace.append({
            "year": year, "result": result, "method": method, "opp_rank": opp_rk, "title": is_title,
            "delta": round(delta, 2), "elo_after": round(elo, 2)
        })

    return round(elo, 2), trace

# ─── Global simulation with opponent-Elo difficulty ───────────────────────────
class FighterState:
    __slots__ = ("name", "canon", "elo", "recent_wc", "loss_streak", "win_streak",
                 "champions", "external", "last_year", "last_title_win_year")
    def __init__(self, name: str, wc: Optional[int], external: bool = False):
        self.name = name
        self.canon = canonical_name(name)
        self.elo = float(STATIC_ELO_EXTERNAL if external else BASE_ELO)
        self.recent_wc = wc
        self.loss_streak = 0
        self.win_streak = 0
        self.champions: Dict[int, bool] = {}
        self.external = external
        self.last_year: Optional[int] = None
        self.last_title_win_year: Dict[int, int] = {}  # division -> most recent title-win year

def expected_score(a_elo: float, b_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b_elo - a_elo) / 400.0))

def clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x

def build_global_events(fighters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    def get_opp_name(fr: Dict[str, Any]) -> Optional[str]:
        for k in OPP_NAME_KEYS:
            if k in fr and isinstance(fr[k], str) and fr[k].strip():
                return fr[k].strip()
        return None

    # Pass 1: from WIN records (loser rank known)
    for f in fighters:
        subj_name = f.get("name") or ""
        subj_canon = canonical_name(subj_name)
        for fr in sort_fights_local(f.get("fight_records", [])):
            if (fr.get("result") or "").lower() != "win":
                continue
            opp = get_opp_name(fr)
            if not opp:
                continue
            opp_canon = canonical_name(opp)
            year = int(fr.get("year", 0) or 0)
            if not year:
                continue
            a, b = sorted([subj_canon, opp_canon])
            key = (year, a, b)
            if key not in events:
                events[key] = {
                    "year": year,
                    "winner": subj_canon,
                    "loser": opp_canon,
                    "winner_rank": None,
                    "loser_rank": fr.get("rank", "NR"),
                    "method": fr.get("method") or "",
                    "is_title": bool(fr.get("is_title", False)),
                    "scores": fr.get("scores", [])
                }

    # Pass 2: from LOSS records (winner rank known)
    for f in fighters:
        subj_name = f.get("name") or ""
        subj_canon = canonical_name(subj_name)
        for fr in sort_fights_local(f.get("fight_records", [])):
            if (fr.get("result") or "").lower() != "loss":
                continue
            opp = get_opp_name(fr)
            if not opp:
                continue
            opp_canon = canonical_name(opp)
            year = int(fr.get("year", 0) or 0)
            if not year:
                continue
            a, b = sorted([subj_canon, opp_canon])
            key = (year, a, b)
            if key in events:
                if events[key]["winner"] == opp_canon:
                    events[key]["winner_rank"] = fr.get("rank", "NR")
            else:
                events[key] = {
                    "year": year,
                    "winner": opp_canon,
                    "loser": subj_canon,
                    "winner_rank": fr.get("rank", "NR"),
                    "loser_rank": None,
                    "method": fr.get("method") or "",
                    "is_title": bool(fr.get("is_title", False)),
                    "scores": fr.get("scores", [])
                }

    # Return sorted by (year, pair)
    return [events[k] for k in sorted(events.keys(), key=lambda x: (x[0], x[1], x[2]))]

def run_global_sim(fighters_raw: List[Dict[str, Any]], db_con=None, elo_run_id: Optional[int]=None) -> Tuple[Dict[str, 'FighterState'], List[Dict[str, Any]]]:
    states: Dict[str, FighterState] = {}
    for f in fighters_raw:
        nm = f.get("name") or ""
        cn = canonical_name(nm)
        wc = get_most_recent_weight_class_from_records(f)
        states[cn] = FighterState(nm, wc, external=False)

    # Map canonical -> full fighter dict (for features freeze)
    canon_to_fighter = {canonical_name(f.get("name") or ""): f for f in fighters_raw}

    events = build_global_events(fighters_raw)

    for seq, ev in enumerate(events, start=1):
        year       = ev["year"]
        w_canon    = ev["winner"]
        l_canon    = ev["loser"]
        method     = ev["method"] or ""
        is_title   = bool(ev["is_title"])
        winner_rk  = ev.get("winner_rank", None)
        loser_rk   = ev.get("loser_rank", None)
        scores     = ev.get("scores", [])

        if w_canon not in states: states[w_canon] = FighterState(w_canon, None, external=True)
        if l_canon not in states: states[l_canon] = FighterState(l_canon, None, external=True)

        ws = states[w_canon]
        ls = states[l_canon]

        # PRE: decay and pre-elos
        ws.elo, ws.last_year = apply_annual_decay(ws.elo, year, ws.last_year)
        ls.elo, ls.last_year = apply_annual_decay(ls.elo, year, ls.last_year)
        pre_w, pre_l = ws.elo, ls.elo

        w_div = ws.recent_wc if ws.recent_wc is not None else 185
        l_div = ls.recent_wc if ls.recent_wc is not None else 185
        w_is_champ_before = bool(ws.champions.get(w_div, False))
        undisputed = (is_title and (loser_rk == "C") and not w_is_champ_before)

        w_delta = local_delta("win",  method, loser_rk,  is_title, scores, year,
                              ls.loss_streak, ws.win_streak, w_is_champ_before, undisputed)
        l_delta = local_delta("loss", method, winner_rk, is_title, scores, year,
                              ls.loss_streak, ls.win_streak, False,           False)
        if undisputed:
            w_delta *= UNDISPUTED_MULT

        Ew = expected_score(ws.elo, ls.elo)
        mult_w = clamp(1.0 + GAMMA * (1.0 - 2.0 * Ew), OPP_MULT_MIN, OPP_MULT_MAX)
        mult_l = clamp(1.0 + GAMMA * (2.0 * (1.0 - Ew) - 1.0),       OPP_MULT_MIN, OPP_MULT_MAX)
        w_delta *= mult_w
        l_delta *= mult_l

        # POST: apply to internal (non-external) states
        if not ws.external:
            ws.elo += w_delta
            ws.loss_streak = 0
            ws.win_streak  += 1
            if undisputed or (is_title and (winner_rk != "C")):
                ws.last_title_win_year[w_div] = max(ws.last_title_win_year.get(w_div, 0), year)
            if undisputed:
                ws.champions[w_div] = True
        if not ls.external:
            ls.elo += l_delta
            ls.loss_streak += 1
            ls.win_streak   = 0
            ls.champions[l_div] = False

        # ---- DB LOGGING (optional) ----
        if db_con is not None and elo_run_id is not None:
            fkey = fight_key(year, ws.name if hasattr(ws, 'name') else w_canon,
                                   ls.name if hasattr(ls, 'name') else l_canon,
                             seq)
            rec_w = recency_weight(year)
            # Freeze features as-of this YEAR (no dates available)
            fa = canon_to_fighter.get(w_canon, {})
            fb = canon_to_fighter.get(l_canon, {})
            feats_w = features_as_of_year(fa, year)
            feats_l = features_as_of_year(fb, year)

            # upsert fighters to get IDs
            w_id = upsert_fighter(db_con, fa.get("name") or w_canon, default_wc=w_div)
            l_id = upsert_fighter(db_con, fb.get("name") or l_canon, default_wc=l_div)

            # features per side
            insert_row(db_con, "features_by_fight",
                       elo_run_id=elo_run_id, fight_key=fkey, fighter_id=w_id, **feats_w)
            insert_row(db_con, "features_by_fight",
                       elo_run_id=elo_run_id, fight_key=fkey, fighter_id=l_id, **feats_l)

            # ratings rows (WIN side)
            insert_row(db_con, "elo_fight_ratings",
                elo_run_id=elo_run_id, fight_key=fkey, fight_date=None, fight_year=year,
                fighter_id=w_id, opponent_id=l_id, opp_name_norm=norm_db(fb.get("name") or l_canon),
                weight_class=w_div, is_title=int(is_title), rank_opp=str(loser_rk),
                method=method, result="win",
                pre_elo=pre_w, post_elo=ws.elo, delta_elo=w_delta, recency_weight=rec_w
            )
            # ratings rows (LOSS side)
            insert_row(db_con, "elo_fight_ratings",
                elo_run_id=elo_run_id, fight_key=fkey, fight_date=None, fight_year=year,
                fighter_id=l_id, opponent_id=w_id, opp_name_norm=norm_db(fa.get("name") or w_canon),
                weight_class=l_div, is_title=int(is_title), rank_opp=str(winner_rk),
                method=method, result="loss",
                pre_elo=pre_l, post_elo=ls.elo, delta_elo=l_delta, recency_weight=rec_w
            )

    return states, events

# ─── Constraint enforcement helpers ───────────────────────────────────────────
def build_recent_h2h_edges(fighters: List[Dict[str, Any]], window: int = H2H_RECENT_WINDOW) -> Dict[Tuple[str, str], int]:
    """
    Build edges (winner -> loser) only from **each winner's last `window` fights**.
    Returns {(winner_canon, loser_canon): year_of_win}.
    """
    edges: Dict[Tuple[str, str], int] = {}

    def get_opp_name(fr: Dict[str, Any]) -> Optional[str]:
        for k in OPP_NAME_KEYS:
            v = fr.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    for f in fighters:
        w_name = f.get("name") or ""
        w_canon = canonical_name(w_name)
        recs = sort_fights_local(f.get("fight_records", []))
        if not recs:
            continue
        last = recs[-window:]
        for fr in last:
            if (fr.get("result") or "").lower() != "win":
                continue
            opp = get_opp_name(fr)
            if not opp:
                continue
            l_canon = canonical_name(opp)
            yr = int(fr.get("year", 0) or 0)
            key = (w_canon, l_canon)
            # keep the most recent year if multiple entries
            if (key not in edges) or (yr > edges[key]):
                edges[key] = yr
    return edges

def title_sort_boost(state: 'FighterState', division: Optional[int]) -> float:
    if division is None:
        return 0.0
    yr = state.last_title_win_year.get(division, 0)
    if yr >= TITLE_SORT_SINCE:
        return float(TITLE_SORT_BONUS)
    return 0.0

def enforce_h2h_constraints(order: List[Dict[str, Any]],
                            name_to_state: Dict[str, 'FighterState'],
                            edges: Dict[Tuple[str, str], int]) -> None:
    """
    Reorder 'order' in-place to satisfy winner-above-loser constraints.
    'order' is a list of rows: {"weight_class", "name", "elo"} (one division).
    Edges must already be filtered (recency + Elo proximity + same division).
    """
    # map name->index for quick lookup
    def canon_of(row): return canonical_name(row["name"])
    changed = True
    passes = 0
    while changed and passes < MAX_CONSTRAINT_PASSES:
        changed = False
        passes += 1
        index = {canon_of(row): i for i, row in enumerate(order)}
        for (w, l), _yr in edges.items():
            if w not in index or l not in index:
                continue
            iw = index[w]; il = index[l]
            if iw > il:
                # move winner just above loser (minimal movement)
                row_w = order.pop(iw)
                order.insert(il, row_w)
                changed = True
                break  # restart pass to rebuild index

# ─── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        fighters: List[Dict[str, Any]] = json.load(f)

    # Can we do the global sim (need opponent names)?
    can_global = any(
        any(k in fr and isinstance(fr[k], str) and fr[k].strip()
            for k in OPP_NAME_KEYS)
        for ft in fighters
        for fr in ft.get("fight_records", [])
    )

    results: List[Dict[str, Any]] = []
    by_division: Dict[str, List[Dict[str, Any]]] = {}

    states: Optional[Dict[str, FighterState]] = None
    events: List[Dict[str, Any]] = []

    # --- DB run header (enables per-fight logging) ---
    rules_version = "elo_rules_final_2025-08-12"
    params_dict = {
        "BASE_ELO": BASE_ELO, "STATIC_ELO_EXTERNAL": STATIC_ELO_EXTERNAL,
        "RANK_TOP_BONUS": RANK_TOP_BONUS, "RANK_BOTTOM_BONUS": RANK_BOTTOM_BONUS, "CHAMP_BONUS": CHAMP_BONUS,
        "TITLE_WIN_FLAT": TITLE_WIN_FLAT, "UNDISPUTED_MULT": UNDISPUTED_MULT,
        "WIN_FLAT_BONUS": WIN_FLAT_BONUS, "LOSS_STREAK_PER": LOSS_STREAK_PER,
        "METHOD": {"KO_SUB": METHOD_KO_SUB, "UD": METHOD_UD, "DEFAULT": METHOD_DEFAULT},
        "SCORECARD_FACTOR": SCORECARD_FACTOR, "ANNUAL_DECAY": ANNUAL_DECAY,
        "OPP_ELO": {"GAMMA": GAMMA, "MIN": OPP_MULT_MIN, "MAX": OPP_MULT_MAX},
        "DIVISION_BINS": DIVISION_BINS,
        "SORTING_ONLY": {
            "TITLE_SORT_BONUS": TITLE_SORT_BONUS,
            "TITLE_SORT_SINCE": TITLE_SORT_SINCE,
            # New constraint rule description & knobs:
            "H2H_RULE": "winner above loser only if win ∈ winner's last N fights and |ΔElo| ≤ threshold",
            "H2H_RECENT_WINDOW": H2H_RECENT_WINDOW,
            "ELO_PROX_THRESHOLD": ELO_PROX_THRESHOLD
        }
    }
    with open(INPUT_FILE, "rb") as _srcf:
        source_bytes = _srcf.read()

    elo_run_id: Optional[int] = None

    if can_global:
        # Try to log this run to the DB; fall back to non-DB if anything fails
        try:
            with db() as con:
                create_schema(con)  # idempotent
                elo_run_id = start_elo_run(con, rules_version, params_dict, source_bytes)
                states, events = run_global_sim(fighters, db_con=con, elo_run_id=elo_run_id)
        except Exception as e:
            print(f"[warn] DB logging disabled for this run: {e}")
            states, events = run_global_sim(fighters)
    # else: we'll use local fallback below (no DB logging / no events)

    if can_global:
        input_canons = {canonical_name(ft.get("name") or ""): ft for ft in fighters}
        # Build recent H2H edges from last N fights of each winner
        recent_h2h = build_recent_h2h_edges(fighters, H2H_RECENT_WINDOW)

        for cn, st in states.items():
            if cn not in input_canons:
                continue
            recent_wc = get_most_recent_weight_class_from_records(input_canons[cn])
            sort_bonus = title_sort_boost(st, recent_wc)
            sort_score = round(st.elo + sort_bonus, 6)

            results.append({
                "name": st.name,
                "weight_class": recent_wc,
                "final_elo": round(st.elo, 2),
                "total_fights": input_canons[cn].get("total_fights"),
                "wins": input_canons[cn].get("wins"),
                "losses": input_canons[cn].get("losses"),
                "trace_note": "Global simulation; externals "
                              f"{STATIC_ELO_EXTERNAL}; rank-aware loss; annual decay; win-streak bonus",
                "_sort_score": sort_score,
            })
            key = str(recent_wc) if recent_wc is not None else "unknown"
            by_division.setdefault(key, []).append({
                "weight_class": recent_wc,
                "name": st.name,
                "elo": round(st.elo, 2),
                "_sort_score": sort_score,
            })

        # Sort + enforce in-division H2H constraints (RECENT + Elo-proximity)
        for div, rows in by_division.items():
            rows.sort(key=lambda r: r.get("_sort_score", r["elo"]), reverse=True)
            # maps
            name_to_state = {r["name"]: states[canonical_name(r["name"])] for r in rows
                             if canonical_name(r["name"]) in states}
            cn_in_div = {canonical_name(r["name"]) for r in rows}
            div_int = None if div == "unknown" else int(div)

            # Filter edges to this division, ensure both in division, same division,
            # and Elo proximity condition holds.
            relevant_edges: Dict[Tuple[str, str], int] = {}
            for (w, l), yr in recent_h2h.items():
                if (w not in cn_in_div) or (l not in cn_in_div):
                    continue
                sw = states.get(w)
                sl = states.get(l)
                if not sw or not sl:
                    continue
                if sw.recent_wc != sl.recent_wc or sw.recent_wc != div_int:
                    continue
                if abs(sw.elo - sl.elo) <= ELO_PROX_THRESHOLD:
                    relevant_edges[(w, l)] = yr

            enforce_h2h_constraints(rows, name_to_state, relevant_edges)
            for r in rows:
                r.pop("_sort_score", None)

        # Drop the helper key from results as well
        for r in results:
            r.pop("_sort_score", None)

    else:
        # Local fallback (no global events / opponent-Elo)
        for ft in fighters:
            final_elo, _ = compute_fighter_local_elo(ft)
            recent_wc = get_most_recent_weight_class_from_records(ft)
            results.append({
                "name": ft.get("name"),
                "weight_class": recent_wc,
                "final_elo": final_elo,
                "total_fights": ft.get("total_fights"),
                "wins": ft.get("wins"),
                "losses": ft.get("losses"),
                "trace_note": "Local per-fighter; rank-aware loss; annual decay; win-streak bonus"
            })
            key = str(recent_wc) if recent_wc is not None else "unknown"
            by_division.setdefault(key, []).append({
                "weight_class": recent_wc,
                "name": ft.get("name"),
                "elo": final_elo
            })
        for div, rows in by_division.items():
            rows.sort(key=lambda r: r["elo"], reverse=True)

    # Top 15 overall (post-constraints)
    overall = [r for rows in by_division.values() for r in rows]
    overall.sort(key=lambda r: r["elo"], reverse=True)
    top15_overall = overall[:15]

    payload = {
        "scenario": "global_opp_elo + title_sort_bump + H2H_constraints_recent&prox" if can_global else "local_fallback",
        "params": {
            "BASE_ELO": BASE_ELO, "STATIC_ELO_EXTERNAL": STATIC_ELO_EXTERNAL,
            "RANK_TOP_BONUS": RANK_TOP_BONUS, "RANK_BOTTOM_BONUS": RANK_BOTTOM_BONUS, "CHAMP_BONUS": CHAMP_BONUS,
            "TITLE_WIN_FLAT": TITLE_WIN_FLAT, "WIN_FLAT_BONUS": WIN_FLAT_BONUS, "LOSS_STREAK_PER": LOSS_STREAK_PER,
            "SCORECARD_FACTOR": SCORECARD_FACTOR,
            "RECENCY_BINS": {"<=2016":0.40,"2017-2019":0.55,"2020-2021":0.65,"2022":0.75,"2023":0.90,"2024":1.15,"2025+":1.25},
            "ANNUAL_DECAY": ANNUAL_DECAY, "DIVISION_BINS": DIVISION_BINS, "UNDISPUTED_MULTIPLIER": UNDISPUTED_MULT,
            "OPP_ELO": {"GAMMA": GAMMA, "MIN": OPP_MULT_MIN, "MAX": OPP_MULT_MAX},
            "LOSS_PENALTY_MAG": {"NR":25.0, "C":2.0, "rank_1":5.0, "rank_15":22.0},
            "WIN_STREAK_BONUS": "additive: +10 × (consecutive_wins_before - 1) if >= 2",
            "SORTING_ONLY": {
                "TITLE_SORT_BONUS": TITLE_SORT_BONUS,
                "TITLE_SORT_SINCE": TITLE_SORT_SINCE,
                "H2H_RULE": "winner above loser only if win ∈ winner's last N fights and |ΔElo| ≤ threshold",
                "H2H_RECENT_WINDOW": H2H_RECENT_WINDOW,
                "ELO_PROX_THRESHOLD": ELO_PROX_THRESHOLD
            }
        },
        "generated_on": TODAY.isoformat(),
        "fighters": results,
        "leaderboards_by_division": by_division,
        "top15_overall": top15_overall
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    _abs = lambda p: os.path.abspath(os.path.normpath(p))

    with open(OUTPUT_JSON_DETAILED, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote detailed Elo to {_abs(OUTPUT_JSON_DETAILED)}")

    with open(OUTPUT_JSON_SIMPLE, "w", encoding="utf-8") as f:
        json.dump(by_division, f, indent=2)
    print(f"[OK] Wrote grouped rankings to {_abs(OUTPUT_JSON_SIMPLE)}")

    # Optional: standings snapshot to DB (non-fatal)
    try:
        if can_global and states is not None and elo_run_id is not None:
            with db() as con2:
                for cn, st in states.items():
                    fid = upsert_fighter(con2, getattr(st, "name", cn), default_wc=st.recent_wc)
                    if st.recent_wc is not None:
                        insert_row(con2, "elo_standings",
                                   elo_run_id=elo_run_id,
                                   fighter_id=fid,
                                   weight_class=int(st.recent_wc),
                                   elo=float(round(st.elo, 2)))
                con2.commit()
    except Exception as e:
        print(f"[warn] standings snapshot skipped: {e}")

    print("\nTop 15 overall (all divisions):")
    for i, row in enumerate(top15_overall, 1):
        print(f"{i:>2}. {row['name']}  ({row['weight_class']})  Elo {row['elo']:.2f}")

if __name__ == "__main__":
    # Typo guard (legacy): avoid NameError if local_delta used UNDIDPUTED_MULT
    try:
        UNDIDPUTED_MULT  # noqa
    except NameError:
        UNDIDPUTED_MULT = UNDISPUTED_MULT
    main()
