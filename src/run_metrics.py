"""
Player- and team-level off-ball running metrics, plus the situational splits.

PLAYER (all per-90, above a minutes floor, on USABLE reconstructions only)
  * RunThreat90   -- Σ run_xt: the threat a player's runs generate, in the same
                     xT currency as the rest of the project. Transparent.
  * RunValue90    -- THE HEADLINE. Σ of the fitted value attributable to the RUN
                     FEATURES specifically (model with run features minus the same
                     model without them, identical rows and folds). This isolates
                     the run from the pass that found it.
  * DeltaV90      -- ΔV = V(after) − V(before) across the two freeze-frames. This is
                     the possession value of the PASS-AND-RUN TOGETHER, credited to
                     the receiver. Diagnostic: ΔV correlates +0.136 with how far the
                     BALL moved and +0.146 with how far the PLAYER moved -- i.e. it
                     is genuinely joint, so it must NOT be read as "the run's value"
                     on its own. Reported, labelled honestly, but not the headline.
  * InBehind90    -- runs received beyond the defensive line.
  * SepGained     -- mean separation gained from the nearest defender.
  * RunDistance   -- mean run length (a style descriptor, not a quality one).

SITUATIONAL (the part a coach actually asks about)
  splits by game state (trailing / level / leading), phase (open play / counter),
  and zone -- because "does he still make the run at 1-0 up?" is a real question
  and a player's profile is often state-dependent.
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd

import config


def _per90(x, minutes):
    return np.where(minutes > 0, x / minutes * 90.0, np.nan)


def _positions(events: pd.DataFrame) -> pd.DataFrame:
    pos = (events.dropna(subset=["position"])
           .groupby(["player", "player_id"])["position"]
           .agg(lambda s: s.value_counts().index[0]).reset_index())

    def bucket(p):
        p = str(p)
        if "Goalkeeper" in p: return "GK"
        if "Back" in p: return "DEF"
        if "Forward" in p or "Striker" in p or "Center Forward" in p: return "FWD"
        return "MID"
    pos["pos_group"] = pos["position"].apply(bucket)
    return pos.rename(columns={"player": "runner", "player_id": "runner_id"})


def player_run_metrics(runs: pd.DataFrame, minutes: pd.DataFrame,
                       events: pd.DataFrame) -> pd.DataFrame:
    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()

    # FINAL-THIRD RESTRICTION on the headline value metric.
    # `run_value_added` measures how much the run features improve the PREDICTION,
    # which is not the same as football merit: a centre-back dropping into space
    # reliably leads to retained possession, so the model rewards it. Tapsoba --
    # 754 runs, mean reception on the halfway line, mean forward component -1.1 m,
    # 0.1% in behind -- scored near Wirtz. Off-ball running, as a scout means it,
    # happens in the final third, so the headline value is computed there.
    # Empirically this is the right cut: restricting to the attacking half only
    # moved Tapsoba from 10th to 15th, while the final third moves him to 34th and
    # leaves Boniface / Mbappe / Trincao / Hofmann / Wirtz on top -- an attacking
    # movement list. The all-runs version is kept as RunValueAll90 for transparency.
    att = r[r["receipt_x"] >= config.FINAL_THIRD_X]
    att_val = (att.groupby(["runner", "runner_id"])["run_value_added"].sum()
               .rename("run_value_att").reset_index())
    att_n = (att.groupby(["runner", "runner_id"])["run_value_added"].size()
             .rename("att_runs").reset_index())

    # Group by PLAYER ONLY, not by (player, team).
    # A player who changed clubs inside the pooled seasons (Messi: Barcelona ->
    # PSG) otherwise gets one row per club, each holding only part of his runs --
    # while `minutes` is his TOTAL across the pool. That mismatched numerator and
    # denominator roughly halved every transferred player's per-90. Caught by
    # sanity-checking why Messi ranked 27th.
    team_of = (r.groupby(["runner", "runner_id"])["team"]
               .agg(lambda s: s.value_counts().index[0]).reset_index())

    agg = r.groupby(["runner", "runner_id"]).agg(
        runs=("run_xt", "count"),
        run_threat=("run_xt", "sum"),
        run_value_added=("run_value_added", "sum"),
        in_behind=("in_behind", "sum"),
        sep_gained=("separation_gained", "mean"),
        mean_run_distance=("run_distance", "mean"),
        mean_run_fwd=("run_fwd", "mean"),
        final_third_runs=("zone", lambda s: (s == "Final third").sum()),
        # --- possession value (the fitted model) and time-window outcomes ----
        delta_v=("delta_V", "sum"),
        delta_v_360=("delta_V_360_only", "sum"),
        shots_5s=("shot_5s", "sum"),
        xg_added=("xg_5s", "sum"),
        retain_rate=("retain_10s", "mean"),
        loss_rate=("lost_5s", "mean"),
        value_lost=("pv_lost_5s", "sum"),
        counter_runs=("phase", lambda s: (s == "Counter").sum()),
        # --- defensive context the runs were made against --------------------
        # How surrounded was he when he received, how tight was the block he
        # played against, and did the run itself break out of an enclosure?
        enc_receipt=("encirclement", "mean"),
        enc_escaped=("encirclement_escaped", "mean"),
        block_faced=("block_spread", "mean"),
        second_def=("second_nearest_def", "mean"),
        crowded_runs=("def_within_10", lambda s: (s >= 3).sum()),
        surrounded_runs=("encirclement", lambda s: (s >= 0.5).sum()),
    ).reset_index().merge(team_of, on=["runner", "runner_id"], how="left")

    m = agg.merge(att_val, on=["runner", "runner_id"], how="left") \
           .merge(att_n, on=["runner", "runner_id"], how="left")
    m = m.merge(_positions(events), on=["runner", "runner_id"], how="left")
    mm = minutes.rename(columns={"player": "runner", "player_id": "runner_id"})
    m = m.merge(mm[["runner", "runner_id", "minutes", "matches"]],
                on=["runner", "runner_id"], how="left")
    m["minutes"] = m["minutes"].fillna(0)
    m = m[m["minutes"] >= config.MIN_MINUTES].copy()

    m["RunThreat90"] = _per90(m["run_threat"], m["minutes"])
    # headline: value of runs made in the FINAL THIRD (see the note above)
    m["RunValue90"] = _per90(m["run_value_att"].fillna(0), m["minutes"])
    # transparency: the same measure over every run, including deep recycling
    m["RunValueAll90"] = _per90(m["run_value_added"].fillna(0), m["minutes"])
    m["AttRuns90"] = _per90(m["att_runs"].fillna(0), m["minutes"])
    m["InBehind90"] = _per90(m["in_behind"], m["minutes"])
    m["Runs90"] = _per90(m["runs"], m["minutes"])
    m["FinalThirdRuns90"] = _per90(m["final_third_runs"], m["minutes"])
    # Separation is NEGATIVE for everyone -- defenders converge as the ball
    # arrives, so a raw mean reads as "who loses least". We baseline against the
    # pool mean so 0 = a typical run and positive = genuinely gets more separation
    # than the average run, which is what a reader expects the number to mean.
    pool_sep = float(r["separation_gained"].mean())
    m["SepGained"] = m["sep_gained"]                      # raw, kept for transparency
    m["SepVsAvg"] = m["sep_gained"] - pool_sep            # the interpretable version
    m.attrs["pool_sep_baseline"] = pool_sep
    m["ThreatPerRun"] = m["run_threat"] / m["runs"].replace(0, np.nan)
    # --- defensive-context metrics ------------------------------------------
    # Encirclement runs 0 (a clear escape side) to 1 (defenders on every side).
    m["Encirclement"] = m["enc_receipt"]
    m["EncEscaped"] = m["enc_escaped"]          # >0 = the run broke out of an enclosure
    m["BlockFaced"] = m["block_faced"]          # how spread the defence was; see caveat
    m["SecondDef"] = m["second_def"]            # 2nd-nearest defender: marked vs swarmed
    m["CrowdedRuns90"] = _per90(m["crowded_runs"].fillna(0), m["minutes"])
    m["SurroundedRuns90"] = _per90(m["surrounded_runs"].fillna(0), m["minutes"])
    # share of a player's runs made while genuinely enclosed
    m["PctSurrounded"] = m["surrounded_runs"] / m["runs"].replace(0, np.nan)
    # possession value added by the runs themselves (the fitted model)
    m["DeltaV90"] = _per90(m["delta_v"].fillna(0), m["minutes"])
    m["DeltaV360_90"] = _per90(m["delta_v_360"].fillna(0), m["minutes"])
    m["Shot5s90"] = _per90(m["shots_5s"].fillna(0), m["minutes"])
    m["xGAdded90"] = _per90(m["xg_added"].fillna(0), m["minutes"])
    m["ValueLost90"] = _per90(m["value_lost"].fillna(0), m["minutes"])
    m["CounterRuns90"] = _per90(m["counter_runs"].fillna(0), m["minutes"])
    m["RetainRate"] = m["retain_rate"]
    m["LossRate"] = m["loss_rate"]
    # net: value the runs create minus the value surrendered when they break down
    m["NetValue90"] = m["DeltaV90"] - m["ValueLost90"]

    for c in ["RunThreat90", "RunValue90", "InBehind90", "Runs90",
              "FinalThirdRuns90", "SepGained", "SepVsAvg", "ThreatPerRun",
              "DeltaV90", "DeltaV360_90", "Shot5s90", "xGAdded90",
              "ValueLost90", "CounterRuns90", "RetainRate", "NetValue90",
              "RunValueAll90", "AttRuns90",
              "Encirclement", "EncEscaped", "BlockFaced", "SecondDef",
              "CrowdedRuns90", "SurroundedRuns90", "PctSurrounded"]:
        m[f"{c}_pct"] = (m[c].rank(pct=True) * 100).round(0)

    return m.sort_values("RunValue90", ascending=False).reset_index(drop=True)


def situational(runs: pd.DataFrame, by: str = "game_state") -> pd.DataFrame:
    """Run profile split by situation -- the 'when does he do it?' view."""
    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()
    g = r.groupby(by, observed=True).agg(
        runs=("run_xt", "count"),
        run_threat_mean=("run_xt", "mean"),
        in_behind_rate=("in_behind", "mean"),
        sep_gained=("separation_gained", "mean"),
        mean_distance=("run_distance", "mean"),
        progress_rate=("y_progress", "mean"),
        delta_v=("delta_V", "mean"),
        shot_5s=("shot_5s", "mean"),
        xg_5s=("xg_5s", "mean"),
        retain_10s=("retain_10s", "mean"),
        lost_5s=("lost_5s", "mean"),
    ).reset_index()
    return g


def player_situational(runs: pd.DataFrame, player: str, minutes: pd.DataFrame,
                       by: str = "game_state") -> pd.DataFrame:
    """One player's run profile across situations, per 90 where meaningful."""
    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1) & (runs["runner"] == player)].copy()
    if len(r) == 0:
        return pd.DataFrame()
    g = r.groupby(by, observed=True).agg(
        runs=("run_xt", "count"),
        run_threat=("run_xt", "sum"),
        threat_per_run=("run_xt", "mean"),
        in_behind=("in_behind", "sum"),
        sep_gained=("separation_gained", "mean"),
        progress_rate=("y_progress", "mean"),
    ).reset_index()
    return g


def team_run_metrics(runs: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """Team run profiles.

    `min_matches` matters: in the pooled club seasons the four focus clubs have
    26-35 matches while every opponent has 1-4, so an unfiltered table is topped
    by teams with a single game of data. Only clubs with a real sample are ranked.
    """
    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()
    t = r.groupby("team").agg(
        matches=("match_id", "nunique"),
        runs=("run_xt", "count"),
        run_threat=("run_xt", "sum"),
        in_behind=("in_behind", "sum"),
        sep_gained=("separation_gained", "mean"),
        progress_rate=("y_progress", "mean"),
    ).reset_index()
    for c in ["runs", "run_threat", "in_behind"]:
        t[f"{c}_per_match"] = t[c] / t["matches"]
    t = t[t["matches"] >= min_matches]
    return t.sort_values("run_threat_per_match", ascending=False).reset_index(drop=True)
