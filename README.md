 #ufc-elo
 - An end-to-end, reproducible pipeline for MMA fight predictions that anchors on Elo (prior skill), adds a GLM/Logistic stats model, optionally blends a Random Forest for non-linear matchup patterns, and outputs calibrated win probabilities plus fair/book odds. Includes dynamic Elo weighting, stance/anthro features, and specialized KO power vs. chin matchup signals.
 -Created by Akshay Rau 

 VERSION: 4.0

 Highlights
 - Elo is generated with the intent to fix UFC rankings with an objective scale. This allows us to dynamically see who is currently the best without any bias.
 - Elo prior with temperature + cap to keep priors meaningful but bounded.
 - Stats model (GLM/Logistic Regression) from engineered fight deltas; percent features safely standardized.
 - Optional Random Forest blend (auto if rf_model.pkl present) to capture non-linear interactions.
 - Matchup features: stance, reach/height, sub pace, decision quality, and KO power vs. chin in order to add nuance and deeper insight in matchups
 - Readable CLI output: top drivers, components breakdown (Elo vs. stats vs. RF), and fair odds.
 - Config-driven: all weights/scalers/combination rules live in a single JSON.

 Pipeline 
 *LOAD INPUTS
- elo_rankings_by_weight.json → per-division Elo for each fighter.
- fighter_stats.json → per-fighter stats + recent fights. (Sig. Strike Accuracy, Takedown Defense, KO/SUB/DEC percentage)
- fighter_stats.json → per-fighter stats + recent fights.
- configs/model_weights.json → GLM betas, scaler, and blend config.
*BUILD
- Per-fighter feature vectors (accuracy/defense, SAPM, TD, résumé, recent form, decision quality).
- Anthropometrics/stance (reach, height, stance binaries).
- KO power vs. chin: (A_power × B_chin_frag) – (B_power × A_chin_frag). (recently tuned down)
- Recent chin fragility (share of last-5 KO losses).
*DELTAS
- Convert per-fighter features into matchup deltas (A−B), with sensible signs (e.g., lower Strikes Absorbed Per Minute is good)
*LOGISTIC REGRESSION WITH STATS
- Standardize using provided scaler; apply betas + intercept.
- Apply stats temperature to shrink/expand the stats contribution.
*RANDOM FOREST MODEL (Optional)
- Build training-style row; get predict_proba; convert to logit.
- Blend GLM and RF stats logits with weight w_rf.
*ELO PRIOR
- Convert Elo difference to prior probability → logit
- Apply Elo temperature (soften) and cap (bound)
*COMBINE
- additive: z = z_elo + z_stats
- additive_dyn: z = w_elo·z_elo + z_stats
- blend (default): z = w_elo·z_elo + (1−w_elo)·z_stats
- w_elo can be dynamic (bounded by W_ELO_MIN..MAX) based on rank/recency/fights.
*PROBABLITIES
- p = sigmoid(z), then compute fair odds and optional book odds with overround.

Techniques Used
- Elo Rating System as a Bayesian-style prior for matchup strength.
- Generalized Linear Model (Logistic Regression) for interpretable, calibrated probabilities from engineered features.
- Random Forest Classifier (optional) to learn non-linear interactions the GLM might miss.
- Model Blending/Ensembling (GLM ⨁ RF) at the logit level.
- Dynamic Prior Weighting (uncertainty-aware Elo share).
- Feature Engineering: stance asymmetry (open/orthodox/southpaw), reach/height deltas, decision dominance, KO power vs. chin.
- Regularization by Temperature/Cap: stable behavior across extreme priors or noisy stats.
- Odds Conversion: probability → decimal/US odds; optional overround to simulate a bookmaker margin.
- Config-as-data: all knobs in JSON for fast iteration and reproducibility.

Repository Layout
ufc-elo/
├─ fighter_predictions.py         # CLI + core logic
├─ configs/
│  └─ model_weights.json          # betas, scaler, combine, temps/caps
├─ data/
│  └─ sample/                     # tiny fixtures for tests/examples
├─ models/
│  └─ rf_model.pkl                # optional (git LFS)
├─ tests/
│  └─ test_predict_basic.py
├─ README.md  LICENSE  CHANGELOG.md
└─ .github/workflows/ci.yml       # pytest


Data Files

elo_rankings_by_weight.json
{
  "135": [{"name": "Fighter A", "elo": 2400.5}, ...],
  "145": [{"name": "Fighter B", "elo": 2350.2}, ...]
}


fighter_stats.json
[
  {
    "name": "Fighter A",
    "sig_strike_acc": 42.1, "sig_strike_def": 58.3, "sapm": 3.1,
    "td_acc": 33.0, "td_def": 70.0,
    "ranked_wins": 3, "championship_wins": 1,
    "total_fights": 18, "avg_fight_time_sec": 820,
    "reach_in": 74, "height_in": 72, "stance": "Orthodox",
    "fight_records": [
      {"year": 2024, "result": "win", "method": "UD", "rank": 5, "scorecards": ["50-45","49-46","49-46"]},
      {"year": 2023, "result": "loss", "method": "KO", "rank": 8}
    ],
    "ko_wins": 8, "sub_wins": 2, "ko_losses": 1, "sub_losses": 1,
    "ko_loss_recent5": 0.0
  }
]


Configuration
- B0: intercept
- B: betas keyed by training names (e.g., d_sig_strike_acc) or short names (ssa); both are supported.
- scaler.mean/std: per-feature standardization (percent features often use std = 100).
- config:
  combine: "blend" | "additive" | "additive_dyn"
  ELO_K, W_ELO, W_ELO_MIN, W_ELO_MAX
  elo_temp, elo_cap
  stats_temp
  clip_z
- Example: your v5 weights include KO durability, stance/anthro, and the new d_ko_powervschin / d_recent_ko_loss_pct5.

Command Line Interface Usage
python fighter_predictions.py \
  --elo_json data/elo_rankings_by_weight.json \
  --stats_json data/fighter_stats.json \
  --weights_json configs/model_weights.json \
  --A "Fighter A" --B "Fighter B" \
  --combine blend --overround 0.05 \
  --elo_temp 0.55 --elo_cap 1.2 \
  --stats_temp 0.55 --inspect_blend

Helpful flags
--set_beta "d_ko_powervschin=1.8, d_recent_ko_loss_pct5=0.8" (quick tuning)
--zero "d_total_fights,d_recent_win_pct5" (drop features)
--rf_model models/rf_model.pkl --w_rf 0.35 (enable RF blend)
--assume_pct01 (if no scaler, treat 0..100% as 0..1)
--print_profile --profile_order logical (full per-feature table)

Tuning Guide (what to turn up/down & why)
| Goal                              | Knob                                                    | Effect                                               |
| --------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| Trust **stats** more              | Increase `stats_temp` (→ 1.0)                           | Expands stats logit influence                        |
| Trust **Elo** less (but not zero) | Lower `W_ELO` and/or raise `W_ELO_MIN` prudently        | Reduces prior dominance; dynamic blend stays bounded |
| Prevent runaway Elo               | Lower `elo_temp`, set `elo_cap` (e.g., 1.2)             | Caps prior logit magnitude                           |
| Emphasize KO fragility vs power   | Raise `d_ko_powervschin`, `d_recent_ko_loss_pct5` betas | Improves handling of “chin vs puncher” matchups      |
| Reduce résumé bias                | Decrease `d_total_fights`, `d_recent_win_pct5`, etc.    | Limits streak/volume effects                         |
| Increase non-linear modeling      | Use `--rf_model` and tweak `--w_rf`                     | Lets RF capture interactions GLM misses              |

Data & APIs
- This project pulls all of its data through scarping static & dynamic html files. 
- API integration is the next feature looked to be added to allow for all the elo data and previous statistical data to be easily accessed and used by other programmers

Testing & CL
- Unit tests validate probabilities are in (0,1), both Elo & stats contribute, and key features are directionally monotonic (e.g., more KO power vs weaker chin → higher A probability).
- GitHub Actions runs pytest on PRs and pushes to main.
- Include small sample datasets under data/sample/ for fast tests.

Why Random Forest Helps (Optional Blend)
- Learns thresholds and interactions (EX. “KO power only matters when opponent’s chin fragility and stance create open counters”).
- Complementary to GLM: GLM is interpretable and stable; RF adds capacity where linearity is too limiting.
- We blend at the logit level with w_rf to keep calibration reasonable.

Example Output
================= UFC Fight Prediction =================
Matchup: Fighter A vs Fighter B
Elo: 2400.5 vs 2350.2  (Δ = +50.3 for Fighter A)
Elo prior (A): 57.8%
Model win probability: Fighter A 61.3% | Fighter B 38.7%  (stats shift: +3.5 pp)

Blend / Config:
  Combine: blend (z = w_elo·z_elo + (1-w_elo)·z_stats)
  Weights → Elo: 0.40 | Stats: 0.60
  Stats logit parts → GLM: +0.210 | RF: +0.085 (w_rf=0.35) | Combined: +0.256
  Logit parts → Elo: +0.145 | Final: +0.277

Fair odds (no vig):
  Fighter A: -158  (dec 1.631)
  Fighter B: +158  (dec 2.585)

Top stat drivers (GLM):
  • Sig. strike ACC:  Δ=3.000  (+0.320 logit) → favors A
  • KO power vs chin:  Δ=0.120  (+0.240 logit) → favors A
  • TD defense:  Δ=12.0  (+0.180 logit) → favors A


Acknowledgements
Community datasets (such as ufcstats.com & rosterwatch), public statistics portals, and open-source libraries: NumPy, scikit-learn, pandas, joblib, pytest.
