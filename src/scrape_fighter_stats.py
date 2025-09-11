import os
import json as _json
import re
import difflib
import unicodedata
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

# ─── PLAYWRIGHT SETUP ──────────────────────────────────────────────────────────
_pw = sync_playwright().start()
_browser = _pw.chromium.launch(headless=True)
# ────────────────────────────────────────────────────────────────────────────────

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
ACTIVE_FILE = os.path.join(DATA_DIR, 'active_fighters.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'fighter_stats.json')

os.makedirs(DATA_DIR, exist_ok=True)

with open(ACTIVE_FILE, 'r', encoding='utf-8') as f:
    active_map = _json.load(f)

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

WEIGHT_TOKENS = {
    'catchweight', 'heavyweight', 'light heavyweight', 'middleweight',
    'welterweight', 'lightweight', 'featherweight', 'bantamweight', 'flyweight'
}

def canonical_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()

WEIGHT_TOKENS_CANON = {canonical_name(t) for t in WEIGHT_TOKENS}

def normalize_stance(s):
    """Map various UFCStats stance strings to a tidy canonical form."""
    if not s:
        return None
    s = s.strip().lower()
    if s in ('--', 'n/a', 'na', ''):
        return None
    if 'orth' in s:     return 'Orthodox'
    if 'south' in s:    return 'Southpaw'
    if 'switch' in s:   return 'Switch'
    if 'open' in s:     return 'Open'
    return s.title()

def normalize_date(date_str: str) -> str:
    if not date_str:
        return date_str
    s = str(date_str).strip().replace('\xa0', ' ').replace('.', '')
    m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', s)
    if m:
        return m.group(1)
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%b %d,%Y', '%B %d,%Y'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return s

def extract_decision_scorecards(detail_url):
    try:
        res = session.get(detail_url, timeout=10)
        res.raise_for_status()
        text = BeautifulSoup(res.text, 'html.parser').get_text(separator=' ', strip=True)
        is_title = 'title bout' in text.lower()
        matches = re.findall(r"([A-Z][a-zA-Z\s\.'-]+?)\s+(\d{2})\s*[-–]\s*(\d{2})", text)
        scorecards, temp = [], []
        for _, s1, s2 in matches:
            temp.append(f"{s1}-{s2}")
            if len(temp) == 3:
                scorecards.append(tuple(temp))
                temp = []
        return scorecards, is_title
    except Exception:
        return [], False

# --- Robustly get the two fighter lines in a roster.watch block ----------------
def _two_name_age(block_text: str):
    """
    Return [name1, name2] by scanning for lines that contain an '(<age>)' pattern.
    Skips lines containing 'ufc rank' or 'elo', and ignores weight tokens.
    """
    out = []
    for raw_ln in block_text.splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if 'ufc rank' in low or 'elo:' in low or 'elo •' in low:
            continue
        # Require a pure age like '(35)' (exactly 1–2 digits inside parens)
        if not re.search(r'\(\d{1,2}\)', ln):
            continue
        # Take the text before the '(' as the name area
        left = ln.split('(', 1)[0].strip()
        nm_can = canonical_name(left)
        if not nm_can or nm_can in WEIGHT_TOKENS_CANON:
            continue
        out.append(left)
        if len(out) == 2:
            break
    return out if len(out) == 2 else None

def _parse_rw_block(block_text: str, subj_canon: str):
    """returns list of [(date, opponent, rank)] for the subject in this block"""
    out = []
    m = re.search(r'([A-Za-z]{3,}\.?\s*\d{1,2},\s*\d{4})|(\d{4}-\d{2}-\d{2})', block_text)
    if not m:
        return out
    fight_date = normalize_date(m.group(1) or m.group(2))

    fighters = _two_name_age(block_text)
    if not fighters:
        return out
    name1 = canonical_name(fighters[0])
    name2 = canonical_name(fighters[1])

    # collect ranks (can be 'Champ' or '#<num>')
    raw_ranks_iter = list(re.finditer(r'UFC\s*rank[:\s]*#?(\d+|champ)', block_text, flags=re.IGNORECASE))
    ranks = [('C' if rm.group(1).lower() == 'champ' else rm.group(1)) for rm in raw_ranks_iter]

    if len(ranks) == 1:
        if name1 == subj_canon and name2 != subj_canon:
            opponent, rk = name2, ranks[0]
        elif name2 == subj_canon and name1 != subj_canon:
            opponent, rk = name1, ranks[0]
        else:
            if subj_canon in name1 and subj_canon not in name2:
                opponent, rk = name2, ranks[0]
            elif subj_canon in name2 and subj_canon not in name1:
                opponent, rk = name1, ranks[0]
            else:
                return out
    else:
        r1 = ranks[0] if len(ranks) >= 1 else 'NR'
        r2 = ranks[1] if len(ranks) >= 2 else 'NR'
        if name1 == subj_canon and name2 != subj_canon:
            opponent, rk = name2, r2
        elif name2 == subj_canon and name1 != subj_canon:
            opponent, rk = name1, r1
        else:
            if subj_canon in name1 and subj_canon not in name2:
                opponent, rk = name2, r2
            elif subj_canon in name2 and subj_canon not in name1:
                opponent, rk = name1, r1
            else:
                return out

    opponent = canonical_name(opponent)
    if opponent in WEIGHT_TOKENS_CANON or not opponent:
        return out

    out.append((fight_date, opponent, rk))
    return out

def extract_opponent_from_row(tr, subj_canon: str):
    """
    Return (opponent_name_canonical, opponent_url_or_None) for a UFCStats fight row.
    Prefers anchors that link to /fighter-details/; falls back to the opponent cell text.
    """
    anchors = tr.select('a[href*="/fighter-details/"]')
    seen = {}
    for a in anchors:
        raw = a.get_text(strip=True)
        nm = canonical_name(raw)
        if not nm:
            continue
        href = a.get('href') or ''
        if href and not href.startswith("http"):
            href = urljoin("http://ufcstats.com", href)
        seen.setdefault(nm, href or None)

    for nm, href in seen.items():
        if nm != subj_canon:
            return nm, href

    # Fallback: parse the second table cell text (opponent column) if anchors fail
    cols = tr.select('td')
    cell = cols[1].get_text(' ', strip=True) if len(cols) > 1 else ''
    cand = canonical_name(cell)
    subj_tokens = set(subj_canon.split())
    remain = [t for t in cand.split() if t not in subj_tokens]
    nm = " ".join(remain).strip() or cand
    return nm, None

def date_obj(iso_str: str):
    try:
        return datetime.strptime(iso_str, "%Y-%m-%d").date()
    except Exception:
        return None

def lookup_rank(bout_rank_map, fight_date: str, opp: str) -> str:
    # 1) exact
    rank = bout_rank_map.get((fight_date, opp))
    if rank:
        return rank
    # 2) same-date fuzzy
    same_date = [((d, o), r) for ((d, o), r) in bout_rank_map.items() if d == fight_date]
    if same_date:
        best_pair = max(same_date, key=lambda item: difflib.SequenceMatcher(None, opp, item[0][1]).ratio())
        if difflib.SequenceMatcher(None, opp, best_pair[0][1]).ratio() >= 0.60:
            return best_pair[1]
    # 3/4) ±2 days
    d0 = date_obj(fight_date)
    if d0:
        window = [((d, o), r) for ((d, o), r) in bout_rank_map.items()
                  if date_obj(d) and abs((date_obj(d) - d0).days) <= 2]
        for (d, o), r in window:
            if o == opp:
                return r
        if window:
            best_pair = max(window, key=lambda item: difflib.SequenceMatcher(None, opp, item[0][1]).ratio())
            if difflib.SequenceMatcher(None, opp, best_pair[0][1]).ratio() >= 0.70:
                return best_pair[1]
    return 'NR'

def _parse_rw_text_fallback(body_text: str, subj_canon: str) -> dict:
    lines = [ln.strip() for ln in body_text.splitlines()]
    date_re = re.compile(r'^(?:[A-Za-z]{3,}\.?\s*\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})$')
    idxs = [i for i, ln in enumerate(lines) if date_re.match(ln)]
    bout_rank_map = {}
    for j, start in enumerate(idxs):
        end = idxs[j+1] if j+1 < len(idxs) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        for (dt, opp, rk) in _parse_rw_block(block, subj_canon):
            bout_rank_map[(dt, opp)] = rk
    return bout_rank_map

def build_bout_rank_map(rw_url: str, subj_canon: str) -> dict:
    if not rw_url:
        return {}
    page = _browser.new_page()
    try:
        page.goto(rw_url, wait_until='domcontentloaded')
        # Warm-up scroll until any row mounts (up to ~10s)
        for _ in range(40):
            if page.locator('div.rt-tr-group').count() > 0:
                break
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(250)

        # Parse while scrolling to trigger virtualization
        seen_keys = set()
        bout_rank_map = {}
        no_new_cycles = 0
        MAX_CYCLES = 160
        for _ in range(MAX_CYCLES):
            groups = page.locator('div.rt-tr-group')
            count_now = groups.count()
            new_in_cycle = 0
            for i in range(count_now):
                block = groups.nth(i).inner_text()
                fighters = _two_name_age(block) or []
                mdate = re.search(r'([A-Za-z]{3,}\.?\s*\d{1,2},\s*\d{4})|(\d{4}-\d{2}-\d{2})', block)
                if not mdate or len(fighters) < 2:
                    continue
                k = (normalize_date(mdate.group(1) or mdate.group(2)),
                     canonical_name(fighters[0]),
                     canonical_name(fighters[1]))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                new_in_cycle += 1
                for (dt, opp, rk) in _parse_rw_block(block, subj_canon):
                    bout_rank_map[(dt, opp)] = rk

            if new_in_cycle == 0:
                no_new_cycles += 1
            else:
                no_new_cycles = 0
            if no_new_cycles >= 10:
                break
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(160)

        # Text fallback if react-table yielded nothing (Reyes/Ulberg cases)
        if not bout_rank_map:
            body_text = page.inner_text('body')
            bout_rank_map = _parse_rw_text_fallback(body_text, subj_canon)

        return bout_rank_map
    finally:
        page.close()

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
fighter_stats = []
for name, (ufc_url, rw_url) in tqdm(active_map.items(), desc='Scraping stats'):
    try:
        res = session.get(ufc_url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        title_tag = soup.select_one('span.b-content__title-highlight')
        fighter_name = title_tag.text.strip() if title_tag else name
        subj_canon = canonical_name(fighter_name)

        bout_rank_map = build_bout_rank_map(rw_url, subj_canon)

        # --- Bio panel (weight, reach, stance) ---
        weight = None
        reach_in = None
        stance = None
        for li in soup.select('ul.b-list__box-list li'):
            txt = li.get_text(' ', strip=True)

            # Weight: 185 lbs.
            if txt.startswith('Weight:'):
                m = re.search(r'(\d+)\s*lbs', txt)
                if m:
                    weight = int(m.group(1))

            # Reach: 75"
            elif txt.startswith('Reach:'):
                val = txt.split(':', 1)[1].strip()
                m = re.search(r'(\d+(?:\.\d+)?)', val)
                if m:
                    try:
                        reach_in = float(m.group(1))
                    except Exception:
                        pass

            # STANCE: Orthodox
            elif txt.upper().startswith('STANCE:'):
                val = txt.split(':', 1)[1].strip()
                stance = normalize_stance(val)

        # --- Career stats (per 15 / percentages) incl. Sub. Avg. ---
        stats_map = {
            'Avg. Fight Time': None,
            'Str. Acc.': None,
            'Str. Def': None,
            'SApM': None,
            'TD Acc.': None,
            'TD Def.': None,
            'Sub. Avg.': None,  # NEW
        }
        for li in soup.select('ul.b-list__box-list li'):
            txt = li.get_text(strip=True)
            for key in stats_map:
                if txt.startswith(key):
                    val = txt.split(':', 1)[1].strip().rstrip('%')
                    if key == 'Avg. Fight Time':
                        parts = val.split(':')
                        stats_map[key] = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else None
                    else:
                        stats_map[key] = float(val) if val not in ('--', 'N/A', '') else None

        wins = losses = draws = 0
        win_methods = {m: 0 for m in ['KO/TKO', 'SUBMISSION', 'DECISION - UNANIMOUS', 'DECISION - SPLIT', 'DECISION - MAJORITY', 'DQ']}
        loss_methods = win_methods.copy()
        ranked_wins = 0
        win_opponent_ranks = []
        all_opponent_ranks = []
        decision_scores = []
        most_recent_year = 0
        championship_wins = championship_losses = championship_fights = 0
        fight_records = []

        for tr in soup.select('table.b-fight-details__table tbody tr'):
            cols = tr.select('td')
            if len(cols) < 8:
                continue

            # Result
            raw = cols[0].get_text(strip=True).upper()
            if raw in ('NC', 'CNC', 'NO CONTEST'):
                result = 'nc'
            elif raw in ('DRAW', 'M-DRAW', 'SPLIT DRAW', 'MAJORITY DRAW'):
                result = 'draw'
            elif raw in ('WIN', 'W'):
                result = 'win'
            elif raw in ('LOSS', 'L'):
                result = 'loss'
            else:
                result = raw.lower()

            # Opponent (canonical) + URL
            opp, opp_url = extract_opponent_from_row(tr, subj_canon)

            # Date/year
            dm = re.search(r"([A-Za-z]{3,}\.?\s*\d{1,2},\s*\d{4})", cols[6].get_text(' ', strip=True))
            fight_date = normalize_date(dm.group(1)) if dm else ''
            year = int(fight_date.split('-')[0]) if fight_date else 0
            most_recent_year = max(most_recent_year, year)

            # Scores / title (ensure absolute URL)
            link = tr.select_one('a[href*="/fight-details/"]')
            detail_href = link['href'] if link else ''
            if detail_href and not detail_href.startswith('http'):
                detail_href = urljoin("http://ufcstats.com", detail_href)
            scores, is_title = extract_decision_scorecards(detail_href) if detail_href else ([], False)
            if scores:
                decision_scores.extend(scores)
            if is_title:
                championship_fights += 1

            # Method
            lr = cols[7].get_text(' ', strip=True).lower()
            if lr.startswith('u-dec'):
                method = 'DECISION - UNANIMOUS'
            elif lr.startswith('s-dec'):
                method = 'DECISION - SPLIT'
            elif lr.startswith('m-dec'):
                method = 'DECISION - MAJORITY'
            elif 'ko/tko' in lr or 'knockout' in lr:
                method = 'KO/TKO'
            elif 'sub' in lr:
                method = 'SUBMISSION'
            elif lr.startswith('draw'):
                method = 'DRAW'
            elif 'dq' in lr:
                method = 'DQ'
            else:
                method = cols[7].get_text(' ', strip=True).upper()

            # Opponent rank (use canonical opponent string)
            opp_rank = lookup_rank(bout_rank_map, fight_date, opp)
            all_opponent_ranks.append(opp_rank)

            # Tallies
            if is_title:
                if result == 'win':
                    championship_wins += 1
                elif result == 'loss':
                    championship_losses += 1

            if result == 'win':
                wins += 1
                win_methods[method] += 1
                if isinstance(opp_rank, str) and opp_rank.isdigit():
                    ranked_wins += 1
                    win_opponent_ranks.append(int(opp_rank))
                else:
                    win_opponent_ranks.append(opp_rank)
            elif result == 'loss':
                losses += 1
                loss_methods[method] += 1
            elif result == 'draw':
                draws += 1

            # Store fight record WITH opponent name + URL
            fight_records.append({
                'year': year,
                'result': result,
                'method': method,
                'rank': opp_rank,
                'is_title': is_title,
                'scores': scores,
                'opponent': opp,
                'opponent_url': opp_url
            })

        # Totals
        total = wins + losses + draws or 1
        ko = win_methods['KO/TKO']
        sub = win_methods['SUBMISSION']
        dec = sum(win_methods[m] for m in ['DECISION - UNANIMOUS', 'DECISION - SPLIT', 'DECISION - MAJORITY'])
        dq = win_methods['DQ']
        ko_pct = round(ko / wins * 100, 1) if wins else 0.0
        sub_pct = round(sub / wins * 100, 1) if wins else 0.0
        dec_pct = round(dec / wins * 100, 1) if wins else 0.0
        dq_pct = round(dq / wins * 100, 1) if wins else 0.0
        diff = 100.0 - (ko_pct + sub_pct + dec_pct + dq_pct)
        if abs(diff) > 0.01:
            ko_pct += diff

        # Convenience opponent name/url collections
        opponents = []
        opponents_urls = {}
        for fr in fight_records:
            o = fr.get('opponent')
            if o:
                opponents.append(o)
                u = fr.get('opponent_url')
                if u:
                    opponents_urls.setdefault(o, u)

        fighter_stats.append({
            'name': fighter_name,
            'url': [ufc_url, rw_url],
            'weight_class': weight,
            'stance': stance,                                   # NEW
            'reach_in': reach_in,                               # NEW
            'reach_cm': (round(reach_in * 2.54, 1) if reach_in else None),  # NEW
            'total_fights': total,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'avg_fight_time_sec': stats_map['Avg. Fight Time'],
            'sig_strike_acc': stats_map['Str. Acc.'],
            'sig_strike_def': stats_map['Str. Def'],
            'sapm': stats_map['SApM'],
            'td_acc': stats_map['TD Acc.'],
            'td_def': stats_map['TD Def.'],
            'sub_avg15': stats_map['Sub. Avg.'],               # NEW
            'ko_wins': ko,
            'sub_wins': sub,
            'unanimous_dec_wins': win_methods['DECISION - UNANIMOUS'],
            'split_dec_wins': win_methods['DECISION - SPLIT'],
            'majority_dec_wins': win_methods['DECISION - MAJORITY'],
            'dq_wins': dq,
            'ko_losses': loss_methods['KO/TKO'],
            'sub_losses': loss_methods['SUBMISSION'],
            'unanimous_dec_losses': loss_methods['DECISION - UNANIMOUS'],
            'split_dec_losses': loss_methods['DECISION - SPLIT'],
            'majority_dec_losses': loss_methods['DECISION - MAJORITY'],
            'dq_losses': loss_methods['DQ'],
            'ranked_wins': ranked_wins,
            'win_opponent_ranks': win_opponent_ranks,
            'all_opponent_ranks': all_opponent_ranks,
            'ko_pct': ko_pct,
            'sub_pct': sub_pct,
            'dec_pct': dec_pct,
            'decision_scores': decision_scores,
            'championship_wins': championship_wins,
            'championship_losses': championship_losses,
            'championship_fights': championship_fights,
            'most_recent_year': most_recent_year,
            'fight_records': fight_records,

            # NEW convenience fields (names/urls) — do not affect Elo but handy
            'opponents': opponents,                 # list of canonical opponent names
            'opponents_urls': opponents_urls        # {canonical_name: url}
        })

    except Exception as e:
        print(f"[ERROR] {name} ({ufc_url}): {e}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    _json.dump(fighter_stats, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(fighter_stats)} fighters to {OUTPUT_FILE}")

# ─── PLAYWRIGHT TEARDOWN ───────────────────────────────────────────────────────
_browser.close()
_pw.stop()
