"""
EARLIER EXPLORATION -- retained deliberately, not dead code.

This module produced outputs/reception_report.json, the NULL RESULT that both the
write-up and the application cite: 360 space features add only +0.0026 AUC over a
location-only baseline when predicting whether a reception leads anywhere. That
finding is what redirected the project towards off-ball runs, so the code that
produced it is kept for reproducibility. It is not part of the main build.

Reception Value -- the MODELLED replacement for the old heuristic SpaceVal.

THE PROBLEM WITH THE OLD METRIC
    SpaceVal = Σ control(reception) × (threat(reception) − floor)
was not a model: only `threat` was fitted (xT); `control` was an unvalidated
geometric heuristic; and the metric predicted no outcome. It also had a
degeneracy -- at the moment a pass is completed the receiver is, almost by
definition, in space his team controls -- so it collapsed towards
"volume × pitch location", which event data alone can produce.

WHAT WE DO INSTEAD
Fit a real model with a downstream target, following the validation logic in
Inghilterra's support-creation work (retention / progression / future threat):

    target  y = 1 if, within the next RECEPTION_HORIZON on-ball actions after
                this reception, the receiving team either takes a shot OR
                enters the final third with the ball (i.e. the reception
                *led somewhere*), else 0.

    ReceptionValue = P(y | context)   -- fitted, calibrated, out-of-fold.

THE HEADLINE TEST (the honest part)
Two feature sets on identical rows and identical match-grouped folds:
    BASE  : location / geometry only     (what event data alone can see)
    PLUS  : BASE + 360 space features    (openness, defenders behind, lane, control)
If PLUS does not beat BASE, the 360 space information is NOT adding signal for
this question, and we report that as the finding rather than shipping a metric
that pretends otherwise.

A player's metric is then the sum of value ABOVE EXPECTATION -- crediting the
receiver for getting into positions that beat what a league-average reception in
that location would be worth (this removes the pure volume/possession effect).
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

import config

# what the receiver's situation looks like using EVENT DATA ONLY
BASE_FEATURES = ["end_x", "end_y", "dist_goal_end", "angle_goal_end",
                 "threat_end", "pass_length", "pass_fwd_gain"]
# what the 360 freeze-frame adds about the SPACE he receives in
SPACE_FEATURES = ["receiver_space", "ctrl_end", "def_behind_receiver",
                  "def_within_10", "def_in_lane", "n_def_ahead"]


def _fit_oof(df, feats, groups, n_splits=5):
    X = df[feats].astype(float).values
    y = df["y_progress"].astype(int).values
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=min(n_splits, pd.Series(groups).nunique()))
    for tr, te in gkf.split(X, y, groups):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                               random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, dict(auc=float(roc_auc_score(y, oof)),
                     brier=float(brier_score_loss(y, oof)),
                     base_rate=float(y.mean()), n=int(len(y)))


def build_reception_table(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per completed pass reception, with context + the outcome label."""
    a = actions.sort_values(["match_id", "period", "minute", "second", "index"]).reset_index(drop=True)

    # ---- label: did this reception lead anywhere within the horizon? --------
    K = config.RECEPTION_HORIZON
    y = np.zeros(len(a), dtype=int)
    for _, g in a.groupby(["match_id", "possession"]):
        idx = g.index.to_numpy()
        team = g["team"].to_numpy()
        is_shot = (g["type"] == "Shot").to_numpy()
        ex = g["end_x"].to_numpy()
        sx = g["start_x"].to_numpy()
        for k in range(len(idx)):
            hi = min(len(idx), k + K + 1)
            win = slice(k, hi)
            same = team[win] == team[k]
            shot = (is_shot[win] & same).any()
            # final-third entry: an action that starts outside and ends inside
            entry = ((sx[win] < config.FINAL_THIRD_X) &
                     (ex[win] >= config.FINAL_THIRD_X) & same).any()
            y[idx[k]] = int(shot or entry)
    a["y_progress"] = y

    # ---- receptions = completed passes, credited to the recipient ----------
    r = a[(a["type"] == "Pass") & a["success"] & a["pass_recipient"].notna()
          & a["has_frame"].eq(1)].copy()
    r["dist_goal_end"] = np.hypot(config.GOAL_CENTER[0] - r["end_x"],
                                  config.GOAL_CENTER[1] - r["end_y"])
    gx, gy = config.GOAL_CENTER
    a_side = np.hypot(gx - r["end_x"], (gy - 3.66) - r["end_y"])
    b_side = np.hypot(gx - r["end_x"], (gy + 3.66) - r["end_y"])
    cosang = (a_side**2 + b_side**2 - (2*3.66)**2) / (2*a_side*b_side + 1e-9)
    r["angle_goal_end"] = np.arccos(np.clip(cosang, -1, 1))
    r["pass_length"] = r["length"]
    r["pass_fwd_gain"] = r["fwd_gain"]
    return r.dropna(subset=BASE_FEATURES + SPACE_FEATURES)


def fit_reception_value(receptions: pd.DataFrame):
    """Fit BASE vs BASE+360 on identical rows/folds. Returns (df, report)."""
    df = receptions.reset_index(drop=True)
    groups = df["match_id"].values

    oof_base, m_base = _fit_oof(df, BASE_FEATURES, groups)
    oof_plus, m_plus = _fit_oof(df, BASE_FEATURES + SPACE_FEATURES, groups)
    df["rv_base"] = oof_base
    df["rv"] = oof_plus
    # value ABOVE what location alone predicts -> isolates the space contribution
    df["rv_added"] = df["rv"] - df["rv_base"]

    yb = df["y_progress"].astype(int).values
    cal = {}
    for name, pred in [("base", oof_base), ("plus_360", oof_plus)]:
        fp, mp = calibration_curve(yb, pred, n_bins=10, strategy="quantile")
        cal[name] = dict(mean_pred=mp.tolist(), frac_pos=fp.tolist())

    # a final full-data model for feature importance / app use
    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                               random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
    feats = BASE_FEATURES + SPACE_FEATURES
    final.fit(df[feats].astype(float).values, yb)

    report = dict(metrics_base=m_base, metrics_plus=m_plus,
                  auc_lift=m_plus["auc"] - m_base["auc"],
                  brier_delta=m_plus["brier"] - m_base["brier"],
                  calibration=cal,
                  feature_importance=dict(zip(feats, final.feature_importances_.tolist())),
                  horizon=config.RECEPTION_HORIZON)
    return df, report, final
