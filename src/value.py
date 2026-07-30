"""
AI-ASSISTED MODULE, written to my specification. I required an explicit fitted
possession-value function rather than an xT subtraction, and set the target to a
5-SECOND window rather than a count of actions. See AI_USAGE.md section 2.7.

The possession-value model -- and the run's value as a CHANGE in it.

THIS IS THE ANSWER TO "what did you actually model?"

    V(state) = P(the team in possession takes a shot within the next 5 SECONDS
                 | ball position, the player's position, and the defensive
                   structure around both)

V is FITTED (LightGBM, out-of-fold, match-grouped), not looked up. It is a
possession-value function in the same family as xT/VAEP, with one difference that
matters: **xT can only see where the ball is; V can see the defenders**, because
360 gives us the freeze-frame.

A run is an action that moves the team between two possession states:

    BEFORE : ball at the passer, the runner standing at his origin,
             defenders as they were when the pass was struck
    AFTER  : ball at the reception point, the runner there, defenders as they
             were when he received

    run value  =  V(after) - V(before)          <- "delta V", the headline metric

That is exactly the brief's framing: value assigned to in-possession game states,
and to the actions that move between them. Run Threat (the xT delta) is kept
alongside as the transparent, explain-in-one-sentence companion -- but it is
arithmetic on a location grid, whereas delta V is a fitted model that has seen
the opposition.

WHY THE 360 SPLIT MATTERS
V is fit twice on identical rows and folds -- with and without the freeze-frame
features -- so the contribution of seeing the defenders is measured, not claimed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

import config

# state features that EVENT DATA ALONE could produce
BASE_STATE = ["ball_x", "ball_y", "player_x", "player_y", "dist_ball_player",
              "threat_ball", "threat_player"]
# what the 360 freeze-frame adds: the opposition
# The compactness columns ARE built into the states table (so the ablation can
# measure them) but are deliberately NOT in the production model: adding them moved
# AUC by +0.0006, which is nothing. Reported as a null rather than carried as
# four extra features nobody can justify.
F360_STATE = ["near_def_player", "goalside_player", "near_def_ball",
              "goalside_ball", "mates_ahead"]
ALL_STATE = BASE_STATE + F360_STATE


def _states_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each run into its BEFORE and AFTER possession states, in one common
    feature schema so a single V can score both.
    """
    from src import threat as threatmod
    keep = ["match_id", "index", "period", "minute", "second", "team", "runner",
            "runner_id", "competition"]
    keep = [c for c in keep if c in runs.columns]

    before = runs[keep].copy()
    before["state"] = "before"
    before["ball_x"] = runs["ball_before_x"]; before["ball_y"] = runs["ball_before_y"]
    before["player_x"] = runs["origin_x"];    before["player_y"] = runs["origin_y"]
    before["near_def_player"] = runs["sep_release"]
    before["goalside_player"] = runs["goalside_origin"]
    before["near_def_ball"] = runs["near_def_ball_before"]
    before["goalside_ball"] = runs["goalside_ball_before"]
    before["mates_ahead"] = runs["mates_ahead_before"]
    before["def_within_10_state"] = runs["def_within_10_origin"]
    before["encirclement_state"] = runs["encirclement_origin"]
    before["block_spread_state"] = runs["block_spread_origin"]
    before["second_nearest_state"] = runs["sep_release"]
    before["t_sec"] = runs["minute"] * 60.0 + runs["second"]

    after = runs[keep].copy()
    after["state"] = "after"
    after["ball_x"] = runs["receipt_x"];  after["ball_y"] = runs["receipt_y"]
    after["player_x"] = runs["receipt_x"]; after["player_y"] = runs["receipt_y"]
    after["near_def_player"] = runs["space_at_receipt"]
    after["goalside_player"] = runs["def_goalside_receipt"]
    after["near_def_ball"] = runs["space_at_receipt"]
    after["goalside_ball"] = runs["def_goalside_receipt"]
    after["mates_ahead"] = runs["mates_ahead_after"]
    after["def_within_10_state"] = runs["def_within_10"]
    after["encirclement_state"] = runs["encirclement"]
    after["block_spread_state"] = runs["block_spread"]
    after["second_nearest_state"] = runs["second_nearest_def"]
    # the AFTER state happens one ball-flight later
    after["t_sec"] = runs["minute"] * 60.0 + runs["second"] + runs["duration"].fillna(1.0)

    st = pd.concat([before, after], ignore_index=True)
    st["dist_ball_player"] = np.hypot(st["ball_x"] - st["player_x"],
                                      st["ball_y"] - st["player_y"])
    return st


def build_states(runs: pd.DataFrame, events: pd.DataFrame, threat_grid) -> pd.DataFrame:
    """BEFORE/AFTER states with threat features and time-window outcomes attached."""
    from src import threat as threatmod
    from src import outcomes as outmod
    st = _states_from_runs(runs)
    st["threat_ball"] = [threatmod.threat_at(threat_grid, x, y)
                         for x, y in zip(st["ball_x"], st["ball_y"])]
    st["threat_player"] = [threatmod.threat_at(threat_grid, x, y)
                           for x, y in zip(st["player_x"], st["player_y"])]
    st = outmod.attach_outcomes(st, events, threat_grid)
    return st


def fit_value(states: pd.DataFrame, target: str = "shot_5s"):
    """
    Fit V on all states. Returns (states with V columns, report).
    Fit twice -- base vs base+360 -- on identical rows and folds.
    """
    df = states.dropna(subset=ALL_STATE + [target]).reset_index(drop=True)
    y = df[target].astype(int).values
    groups = df["match_id"].values

    def _oof(feats):
        X = df[feats].astype(float).values
        oof = np.zeros(len(df))
        gkf = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
        for tr, te in gkf.split(X, y, groups):
            m = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8,
                                   min_child_samples=60, random_state=config.RANDOM_STATE,
                                   n_jobs=-1, verbose=-1)
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        return oof, dict(auc=float(roc_auc_score(y, oof)),
                         brier=float(brier_score_loss(y, oof)),
                         n=int(len(y)), base_rate=float(y.mean()))

    oof_base, m_base = _oof(BASE_STATE)
    oof_all, m_all = _oof(ALL_STATE)
    df["V_base"] = oof_base
    df["V"] = oof_all

    cal = {}
    for name, pred in [("base", oof_base), ("plus_360", oof_all)]:
        fp, mp = calibration_curve(y, pred, n_bins=10, strategy="quantile")
        cal[name] = dict(mean_pred=mp.tolist(), frac_pos=fp.tolist())

    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                               random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
    final.fit(df[ALL_STATE].astype(float).values, y)

    report = dict(target=target, metrics_base=m_base, metrics_plus_360=m_all,
                  auc_lift=m_all["auc"] - m_base["auc"],
                  brier_delta=m_all["brier"] - m_base["brier"],
                  pred_correlation=float(np.corrcoef(oof_base, oof_all)[0, 1]),
                  calibration=cal,
                  feature_importance=dict(zip(ALL_STATE,
                                              final.feature_importances_.tolist())))
    return df, report, final


def delta_v(scored_states: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Join V(before) and V(after) back onto each run and take the difference."""
    piv = scored_states.pivot_table(index=["match_id", "index"], columns="state",
                                    values=["V", "V_base"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    out = runs.merge(piv, on=["match_id", "index"], how="left")
    out["delta_V"] = out["V_after"] - out["V_before"]
    out["delta_V_base"] = out["V_base_after"] - out["V_base_before"]
    # how much of the run's value only the freeze-frame can see
    out["delta_V_360_only"] = out["delta_V"] - out["delta_V_base"]
    return out
