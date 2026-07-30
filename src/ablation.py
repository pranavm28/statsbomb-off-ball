"""
The honest ablation: what does each LAYER of information actually buy?

WHY THIS EXISTS
An earlier version of this project compared a "base" model against a "+360" model
and reported the difference as the contribution of the freeze-frame. That framing
was wrong, and it is worth stating plainly: the base feature set already contained
`player_x` / `player_y`, which for the pre-pass state IS the 360-inferred run
origin. The comparison therefore measured only what the DEFENSIVE-STRUCTURE
features add on top of positions we could not have known without 360 in the first
place.

There is no true "without 360" version of this project. Without freeze-frames the
run does not exist at all -- no origin, no displacement, no separation. So the
correct question is not "360 or not", it is "what does each successive layer of
information contribute?", which is what this module measures.

    TIER 1  EVENT ONLY        where the ball is, and how dangerous that spot is.
                              This is genuinely all an event feed gives you at the
                              moment the pass is struck.
    TIER 2  + WHERE THE       the runner's own position, and his distance from the
            RUNNER IS         ball. Requires 360 for the pre-pass state.
    TIER 3  + THE DEFENSIVE   nearest defender, defenders goalside, team-mates
            STRUCTURE         ahead. Requires 360 for both states.

Each tier is fitted on identical rows and identical match-grouped folds, so the
increments are comparable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

import config

TIER1_EVENT = ["ball_x", "ball_y", "threat_ball"]
TIER2_RUNNER = TIER1_EVENT + ["player_x", "player_y", "dist_ball_player", "threat_player"]
TIER3_DEFENCE = TIER2_RUNNER + ["near_def_player", "goalside_player", "near_def_ball",
                                "goalside_ball", "mates_ahead"]
# Tier 3 describes pressure only as "how far is the nearest defender", which treats
# one marker and three converging defenders identically. Tier 4 adds density and
# encirclement so being SURROUNDED is distinguishable from being merely marked.
TIER4_COMPACT = TIER3_DEFENCE + ["def_within_10_state", "encirclement_state",
                                 "block_spread_state", "second_nearest_state"]

TIERS = [("1. Event data only", TIER1_EVENT),
         ("2. + where the runner is", TIER2_RUNNER),
         ("3. + defensive structure", TIER3_DEFENCE),
         ("4. + compactness / encirclement", TIER4_COMPACT)]


def run_ablation(states: pd.DataFrame, target: str = "shot_5s") -> dict:
    """Fit each tier on identical rows and folds; return AUC/Brier and increments."""
    df = states.dropna(subset=TIER4_COMPACT + [target]).reset_index(drop=True)
    y = df[target].astype(int).values
    groups = df["match_id"].values

    out, prev = [], None
    for name, feats in TIERS:
        X = df[feats].astype(float).values
        oof = np.zeros(len(df))
        gkf = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
        for tr, te in gkf.split(X, y, groups):
            m = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8,
                                   min_child_samples=60,
                                   random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y, oof))
        row = dict(tier=name, n_features=len(feats), auc=auc,
                   brier=float(brier_score_loss(y, oof)),
                   gain_vs_previous=None if prev is None else auc - prev)
        out.append(row)
        prev = auc

    return dict(target=target, n_rows=int(len(df)), base_rate=float(y.mean()),
                tiers=out,
                total_gain_from_360=out[-1]["auc"] - out[0]["auc"])
