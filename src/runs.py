"""
AI-ASSISTED MODULE. The freeze-frame pairing mechanic (a Pass event's frame plus the
Ball Receipt event's own frame, with the receiver flagged `actor`) was suggested by the
assistant and verified against the raw data before being built on. The quality gates,
their thresholds, and the decision to exclude rather than clip bad reconstructions are
mine. See AI_USAGE.md sections 2.5 and 4.1.

Off-ball runs: reconstructing them from 360, and valuing them.

THE PROBLEM
Event data credits the player who touched the ball. The player who *made the run*
-- who timed the movement in behind, who pulled a centre-back across, who found
the pocket -- appears nowhere unless he then scores. Every scout asks "does he
move well without the ball?" and the data has never answered it.

WHAT 360 MAKES POSSIBLE
Two freeze-frames bracket every completed pass:
  * at RELEASE  -- the passer's frame. The future receiver is in it, but as an
                   anonymous team-mate dot.
  * at RECEIPT  -- the Ball Receipt event's own frame, where the receiver is
                   flagged `actor`, so his position is known EXACTLY and by name,
                   along with every defender.
The displacement between the two is the run he made while the ball travelled.
Because we have defenders in both frames, we can also measure whether the run
*gained separation* from his nearest marker -- which no event feed records.

RECONSTRUCTION (and its honest weakness)
We know where the receiver ENDED (actor in the receipt frame). We must infer
which release-frame dot he was. We take the nearest team-mate dot to the
reception point, bounded by what is physically reachable in the ball-flight time
(duration x MAX_RUN_SPEED). We also record an ambiguity score -- the ratio of the
2nd-nearest to the nearest candidate -- so matches that were a coin-flip can be
filtered or reported rather than silently trusted.

VALUING THE RUN
Two complementary values, deliberately:
  1. run_xt        = xT(reception) - xT(origin)    -- the threat the movement
                     itself generated, in the same possession-value currency as
                     the rest of the project. Descriptive and transparent.
  2. run_value     = a FITTED model: P(the possession produces a shot or a
                     final-third entry within RUN_HORIZON actions | run + context).
                     Out-of-fold, match-grouped. This is what makes it a model
                     rather than an arithmetic rearrangement of xT.

SELECTION BIAS -- THE HEADLINE LIMITATION
We only observe runs that RECEIVED the ball. The decoy run that dragged a
defender away and was never picked out is invisible to this method. So this
measures "runs that got the ball", not "all off-ball movement", and it will
under-credit selfless decoy runners. Stated everywhere; it is the first thing to
say out loud, not a footnote.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from src import pitch
from src import threat as threatmod

# lightgbm and scikit-learn are imported lazily inside fit_run_value(). Only the
# training path needs them; the Streamlit app imports this module purely for the
# feature-name constants below, and should not have to install the ML stack.

# what the run itself looks like
RUN_FEATURES = ["run_distance", "run_fwd", "run_lateral", "run_speed",
                "separation_gained", "space_at_receipt", "def_goalside_receipt",
                "defs_broken", "in_behind",
                # Compactness. Counting bodies turned out to be nearly useless
                # (def_within_5/10 and density_escaped ranked 20th-22nd of 23 and were
                # dropped); what carries signal is the GEOMETRY of the defence -- how
                # spread the unit is, whether a second defender is also close, and
                # whether the runner had an escape side.
                "second_nearest_def", "encirclement", "block_spread",
                "encirclement_escaped"]
# where it happened -- the baseline any run metric must beat
CONTEXT_FEATURES = ["origin_x", "origin_y", "receipt_x", "receipt_y",
                    "threat_origin", "threat_receipt", "pass_length"]


def _xy(loc):
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def _nearest(pt, pts):
    if pts is None or len(pts) == 0:
        return np.nan
    return float(np.hypot(pts[:, 0] - pt[0], pts[:, 1] - pt[1]).min())


def build_runs(events: pd.DataFrame, frames: pd.DataFrame, threat_grid) -> pd.DataFrame:
    """One row per reconstructed off-ball run (a completed pass with both frames)."""
    ev = events.sort_values(["match_id", "period", "minute", "second", "index"]).copy()

    # ---- receipts, keyed so we can join a pass to its receipt ---------------
    receipts = ev[ev["type"] == "Ball Receipt*"].copy()
    receipts[["rx", "ry"]] = receipts["location"].apply(lambda l: pd.Series(_xy(l)))
    # a receipt belongs to the pass immediately preceding it in the same possession
    receipts["key"] = receipts["match_id"].astype(str) + "_" + receipts["index"].astype(str)

    passes = ev[(ev["type"] == "Pass") & ev["pass_outcome"].isna()
                & ev["pass_recipient"].notna()].copy()
    passes[["ox", "oy"]] = passes["location"].apply(lambda l: pd.Series(_xy(l)))
    passes[["px", "py"]] = passes["pass_end_location"].apply(lambda l: pd.Series(_xy(l)))

    rows = []
    for mid, mp in passes.groupby("match_id"):
        mf = frames[frames["match_id"] == mid]
        by_id = dict(tuple(mf.groupby("id")))
        mr = receipts[receipts["match_id"] == mid]
        # index receipts by (player, index) for a fast forward lookup
        rec_by_player = {p: g.sort_values("index") for p, g in mr.groupby("player")}

        for _, p in mp.iterrows():
            rel_frame = by_id.get(p["id"])
            if rel_frame is None or len(rel_frame) == 0:
                continue
            # --- the receipt: same player, first one AFTER this pass ---------
            cand = rec_by_player.get(p["pass_recipient"])
            if cand is None:
                continue
            nxt = cand[cand["index"] > p["index"]]
            if len(nxt) == 0:
                continue
            r = nxt.iloc[0]
            if r["index"] - p["index"] > 3:          # not the ball from this pass
                continue
            rec_frame = by_id.get(r["id"])
            if rec_frame is None or len(rec_frame) == 0:
                continue

            # --- exact receiver position at receipt (he is the actor) --------
            act = rec_frame[rec_frame["actor"]]
            if len(act) == 0:
                continue
            recv = np.array(_xy(act["location"].iloc[0]))
            if not np.isfinite(recv).all():
                continue
            rec_locs = np.array([_xy(l) for l in rec_frame["location"].values], float)
            rec_tm = rec_frame["teammate"].values.astype(bool)
            defs_receipt = rec_locs[~rec_tm]

            # --- infer where he started: nearest plausible team-mate dot -----
            rel_locs = np.array([_xy(l) for l in rel_frame["location"].values], float)
            rel_tm = rel_frame["teammate"].values.astype(bool)
            rel_actor = rel_frame["actor"].values.astype(bool)
            mates = rel_locs[rel_tm & ~rel_actor]          # exclude the passer
            if len(mates) < 2:
                continue
            d = np.hypot(mates[:, 0] - recv[0], mates[:, 1] - recv[1])
            order = np.argsort(d)
            origin = mates[order[0]]
            nearest_d, second_d = d[order[0]], d[order[1]]
            ambiguity = float(second_d / (nearest_d + 1e-6))

            dur = float(p["duration"]) if np.isfinite(p.get("duration", np.nan)) else np.nan
            reach = (dur * config.MAX_RUN_SPEED) if np.isfinite(dur) else np.inf
            if nearest_d > max(reach, 12.0):     # not physically reachable -> bad match
                continue

            defs_release = rel_locs[~rel_tm]
            mates_receipt = rec_locs[rec_tm]
            run_vec = recv - origin
            run_distance = float(np.hypot(*run_vec))
            # BEFORE state: ball is at the passer, runner is at his origin
            ball_before = np.array([p["ox"], p["oy"]], float)
            goalside_origin = int((defs_release[:, 0] > origin[0]).sum()) if len(defs_release) else 0
            goalside_ball_before = int((defs_release[:, 0] > ball_before[0]).sum()) if len(defs_release) else 0
            near_def_ball_before = _nearest(ball_before, defs_release)
            mates_ahead_before = int((rel_locs[rel_tm][:, 0] > origin[0]).sum()) if rel_tm.any() else 0
            mates_ahead_after = int((mates_receipt[:, 0] > recv[0]).sum()) if len(mates_receipt) else 0

            sep_release = _nearest(origin, defs_release)
            sep_receipt = _nearest(recv, defs_receipt)
            # --- how SURROUNDED was he, before and after the run? -------------
            c_before = pitch.compactness(origin, defs_release)
            c_after = pitch.compactness(recv, defs_receipt)
            spread_before = pitch.block_spread(defs_release)
            spread_after = pitch.block_spread(defs_receipt)
            goalside = int((defs_receipt[:, 0] > recv[0]).sum()) if len(defs_receipt) else 0
            broke = 0
            if len(defs_release):
                lo, hi = min(origin[0], recv[0]), max(origin[0], recv[0])
                broke = int(((defs_release[:, 0] >= lo) & (defs_release[:, 0] <= hi)).sum())

            rows.append(dict(
                match_id=mid, index=int(p["index"]), minute=int(p["minute"]),
                period=int(p["period"]), possession=int(p["possession"]),
                team=p["team"], passer=p["player"], passer_id=p["player_id"],
                runner=p["pass_recipient"], runner_id=p["pass_recipient_id"],
                play_pattern=p.get("play_pattern"),
                origin_x=float(origin[0]), origin_y=float(origin[1]),
                ball_before_x=float(ball_before[0]), ball_before_y=float(ball_before[1]),
                goalside_origin=goalside_origin,
                goalside_ball_before=goalside_ball_before,
                near_def_ball_before=near_def_ball_before,
                mates_ahead_before=mates_ahead_before,
                mates_ahead_after=mates_ahead_after,
                second=int(p["second"]),
                receipt_x=float(recv[0]), receipt_y=float(recv[1]),
                pass_length=float(np.hypot(p["px"] - p["ox"], p["py"] - p["oy"])),
                duration=dur, ambiguity=ambiguity,
                run_distance=run_distance,
                run_fwd=float(run_vec[0]), run_lateral=float(abs(run_vec[1])),
                run_speed=float(run_distance / dur) if (np.isfinite(dur) and dur > 0.2) else np.nan,
                sep_release=sep_release, space_at_receipt=sep_receipt,
                separation_gained=(sep_receipt - sep_release)
                if np.isfinite(sep_receipt) and np.isfinite(sep_release) else np.nan,
                def_goalside_receipt=goalside, defs_broken=broke,
                # compactness at the moment he received
                def_within_5=c_after["n_within_5"],
                def_within_10=c_after["n_within_10"],
                second_nearest_def=c_after["second_nearest"],
                encirclement=c_after["encirclement"],
                block_spread=spread_after,
                # and where he started, so the RUN can be judged, not just the endpoint
                def_within_10_origin=c_before["n_within_10"],
                encirclement_origin=c_before["encirclement"],
                block_spread_origin=spread_before,
                in_behind=int(goalside <= config.IN_BEHIND_MAX_DEF),
                threat_origin=threatmod.threat_at(threat_grid, origin[0], origin[1]),
                threat_receipt=threatmod.threat_at(threat_grid, recv[0], recv[1]),
                receipt_event_id=r["id"], pass_event_id=p["id"],
            ))

    runs = pd.DataFrame(rows)
    if len(runs) == 0:
        return runs
    runs["run_xt"] = runs["threat_receipt"] - runs["threat_origin"]
    # THE RUN-QUALITY FEATURES. Positive density_escaped means he started in a
    # crowd and received in space -- movement that beat the defensive structure,
    # rather than a player who was simply free the whole time.
    runs["density_escaped"] = runs["def_within_10_origin"] - runs["def_within_10"]
    runs["encirclement_escaped"] = runs["encirclement_origin"] - runs["encirclement"]
    runs["is_run"] = (runs["run_distance"] >= config.MIN_RUN_DISTANCE).astype(int)
    runs["clean_match"] = (runs["ambiguity"] >= config.MATCH_AMBIGUITY).astype(int)
    # A reconstructed run faster than a human can sprint is a MIS-MATCH, not a run:
    # we picked the wrong dot. Flagged (and reported) rather than silently kept.
    runs["plausible"] = ((runs["run_speed"].isna()) |
                         (runs["run_speed"] <= config.MAX_RUN_SPEED)).astype(int)
    runs["usable"] = (runs["clean_match"] & runs["plausible"]).astype(int)
    return runs


def apply_pitch_bounds(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Handle freeze-frame positions that fall outside the pitch.

    Event locations are clamped by StatsBomb; freeze-frame player positions are
    NOT. Since the run origin is inferred from a freeze-frame dot, an off-pitch
    dot silently produces an off-pitch run -- visible as lines starting outside
    the touchline, and quietly wrong in the run vector.

    Small excursions (<= PITCH_TOLERANCE) are real football -- a player steps over
    the line -- so we clip them to the boundary. Larger ones are noise or a
    mis-match, so the run is marked unusable rather than silently plotted.
    """
    r = runs.copy()
    tol = config.PITCH_TOLERANCE
    ox, oy = r["origin_x"], r["origin_y"]
    off = (np.maximum(0, -ox) + np.maximum(0, ox - config.PITCH_LENGTH) +
           np.maximum(0, -oy) + np.maximum(0, oy - config.PITCH_WIDTH))
    r["origin_off_pitch"] = off
    r["on_pitch"] = (off <= tol).astype(int)

    # clip the tolerated excursions so nothing is ever drawn outside the pitch
    r["origin_x"] = ox.clip(0, config.PITCH_LENGTH)
    r["origin_y"] = oy.clip(0, config.PITCH_WIDTH)

    # the run vector must be recomputed from the corrected origin
    dx = r["receipt_x"] - r["origin_x"]
    dy = r["receipt_y"] - r["origin_y"]
    r["run_distance"] = np.hypot(dx, dy)
    r["run_fwd"] = dx
    r["run_lateral"] = dy.abs()
    with np.errstate(invalid="ignore", divide="ignore"):
        r["run_speed"] = np.where(r["duration"].fillna(0) > 0.2,
                                  r["run_distance"] / r["duration"], np.nan)
    r["is_run"] = (r["run_distance"] >= config.MIN_RUN_DISTANCE).astype(int)
    r["plausible"] = ((r["run_speed"].isna()) |
                      (r["run_speed"] <= config.MAX_RUN_SPEED)).astype(int)
    r["usable"] = (r["clean_match"] & r["plausible"] & r["on_pitch"]).astype(int)
    return r


def add_game_state(runs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Score difference for the RUNNER's team at the moment of the run.

    Situational context matters: a run made while chasing a game is a different
    act from one made protecting a lead, and any honest player comparison has to
    show whether a profile is state-dependent.
    """
    goals = events[(events["type"] == "Shot") & (events["shot_outcome"] == "Goal")][
        ["match_id", "period", "minute", "second", "team"]].copy()
    og = events[events["type"] == "Own Goal Against"][
        ["match_id", "period", "minute", "second", "team"]].copy()
    goals = pd.concat([goals, og], ignore_index=True)
    goals["t"] = goals["period"] * 10000 + goals["minute"] * 60 + goals["second"]

    runs = runs.copy()
    runs["t"] = runs["period"] * 10000 + runs["minute"] * 60
    diff = np.zeros(len(runs))
    for mid, g in runs.groupby("match_id"):
        mg = goals[goals["match_id"] == mid]
        if len(mg) == 0:
            continue
        for i, row in g.iterrows():
            before = mg[mg["t"] <= row["t"]]
            if len(before) == 0:
                continue
            f = int((before["team"] == row["team"]).sum())
            a = int((before["team"] != row["team"]).sum())
            diff[runs.index.get_loc(i)] = f - a
    runs["score_diff"] = diff
    runs["game_state"] = pd.cut(runs["score_diff"], bins=[-99, -0.5, 0.5, 99],
                                labels=["Trailing", "Level", "Leading"])
    runs["phase"] = np.where(
        runs["play_pattern"].astype(str).str.contains("Counter", na=False), "Counter",
        np.where(runs["play_pattern"].astype(str).str.contains("Regular", na=False),
                 "Open play", "Set piece / other"))
    runs["zone"] = pd.cut(runs["receipt_x"], bins=[-1, 40, 80, 121],
                          labels=["Own third", "Middle third", "Final third"])
    return runs


def label_outcomes(runs: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """y = 1 if the possession shoots or enters the final third within the horizon."""
    a = actions.sort_values(["match_id", "period", "minute", "second", "index"]).copy()
    K = config.RUN_HORIZON
    lab = {}
    for (mid, poss), g in a.groupby(["match_id", "possession"]):
        idx = g["index"].to_numpy()
        team = g["team"].to_numpy()
        is_shot = (g["type"] == "Shot").to_numpy()
        sx = g["start_x"].to_numpy(); ex = g["end_x"].to_numpy()
        for k in range(len(idx)):
            hi = min(len(idx), k + K + 1)
            same = team[k:hi] == team[k]
            shot = bool((is_shot[k:hi] & same).any())
            entry = bool(((sx[k:hi] < config.FINAL_THIRD_X) &
                          (ex[k:hi] >= config.FINAL_THIRD_X) & same).any())
            lab[(mid, int(idx[k]))] = int(shot or entry)
    runs = runs.copy()
    runs["y_progress"] = [lab.get((m, i), 0) for m, i in zip(runs["match_id"], runs["index"])]
    return runs


def fit_run_value(runs: pd.DataFrame):
    """
    Fit CONTEXT-only vs CONTEXT+RUN on identical rows and folds, so the question
    "do the RUN features add anything over simply where it happened?" is answered
    with evidence rather than assertion.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.calibration import calibration_curve

    df = runs.dropna(subset=CONTEXT_FEATURES + RUN_FEATURES).reset_index(drop=True)
    groups = df["match_id"].values
    y = df["y_progress"].astype(int).values

    def _oof(feats):
        X = df[feats].astype(float).values
        oof = np.zeros(len(df))
        gkf = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
        for tr, te in gkf.split(X, y, groups):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                                   random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        return oof, dict(auc=float(roc_auc_score(y, oof)),
                         brier=float(brier_score_loss(y, oof)), n=int(len(y)),
                         base_rate=float(y.mean()))

    oof_ctx, m_ctx = _oof(CONTEXT_FEATURES)
    oof_all, m_all = _oof(CONTEXT_FEATURES + RUN_FEATURES)
    df["rv_context"] = oof_ctx
    df["run_value"] = oof_all
    df["run_value_added"] = df["run_value"] - df["rv_context"]

    cal = {}
    for name, pred in [("context", oof_ctx), ("plus_run", oof_all)]:
        fp, mp = calibration_curve(y, pred, n_bins=10, strategy="quantile")
        cal[name] = dict(mean_pred=mp.tolist(), frac_pos=fp.tolist())

    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
                               random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
    feats = CONTEXT_FEATURES + RUN_FEATURES
    final.fit(df[feats].astype(float).values, y)

    report = dict(metrics_context=m_ctx, metrics_plus_run=m_all,
                  auc_lift=m_all["auc"] - m_ctx["auc"],
                  brier_delta=m_all["brier"] - m_ctx["brier"],
                  pred_correlation=float(np.corrcoef(oof_ctx, oof_all)[0, 1]),
                  calibration=cal,
                  feature_importance=dict(zip(feats, final.feature_importances_.tolist())))
    return df, report, final
