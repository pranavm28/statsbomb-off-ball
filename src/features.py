"""
Action table: one row per on-ball action, enriched with threat + 360 context.

This is the single source the maps, the xPass model, and the metrics read. Each
action records where it started/ended, whether it succeeded, its threat delta,
and the 360 freeze-frame context: pressure on the ball, defenders bypassed
(packing), and the space it was played into. The pressure/space label columns
were removed with the on-ball action maps they existed to colour.
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from src import pitch
from src import space as spacemod
from src import threat as threatmod

ON_BALL = ["Pass", "Carry", "Dribble", "Shot"]


def _xy(loc):
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def build_action_table(events: pd.DataFrame, frames: pd.DataFrame, threat_grid) -> pd.DataFrame:
    ev = events[events["type"].isin(ON_BALL)].copy()
    ev = ev.sort_values(["match_id", "period", "minute", "second", "index"])

    ev[["start_x", "start_y"]] = ev["location"].apply(lambda l: pd.Series(_xy(l)))

    def _end(r):
        if r["type"] == "Pass":
            return _xy(r.get("pass_end_location"))
        if r["type"] == "Carry":
            return _xy(r.get("carry_end_location"))
        if r["type"] == "Shot":
            return config.GOAL_CENTER
        return (r["start_x"], r["start_y"])
    ev[["end_x", "end_y"]] = ev.apply(lambda r: pd.Series(_end(r)), axis=1)

    def _ok(r):
        if r["type"] == "Pass":
            return pd.isna(r.get("pass_outcome"))
        if r["type"] == "Dribble":
            return r.get("dribble_outcome") == "Complete"
        if r["type"] == "Shot":
            return r.get("shot_outcome") == "Goal"
        return True
    ev["success"] = ev.apply(_ok, axis=1)

    ev["dist_goal"] = ev.apply(lambda r: pitch.dist_to_goal(r["start_x"], r["start_y"]), axis=1)
    ev["angle_goal"] = ev.apply(lambda r: pitch.angle_to_goal(r["start_x"], r["start_y"]), axis=1)
    ev["threat_start"] = ev.apply(lambda r: threatmod.threat_at(threat_grid, r["start_x"], r["start_y"]), axis=1)
    ev["threat_end"] = ev.apply(lambda r: threatmod.threat_at(threat_grid, r["end_x"], r["end_y"]), axis=1)
    ev["threat_added"] = (ev["threat_end"] - ev["threat_start"]).where(ev["success"], 0.0)
    ev["shot_xg"] = ev["shot_statsbomb_xg"].fillna(0.0)

    ev["length"] = np.hypot(ev["end_x"] - ev["start_x"], ev["end_y"] - ev["start_y"])
    ev["fwd_gain"] = ev["end_x"] - ev["start_x"]
    ev["progressive"] = (ev["type"].isin(["Pass", "Carry"]) & ev["success"]
                         & (ev["fwd_gain"] >= config.PROGRESSIVE_MIN_GAIN))

    ev = _attach_360(ev, frames)

    ev["line_break"] = (ev["progressive"] & (ev["defenders_bypassed"].fillna(0) >= 1))
    return ev.reset_index(drop=True)


def _attach_360(ev: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    cols = ["def_dist", "n_def_ahead", "n_tm_ahead", "defenders_bypassed",
            "receiver_space", "def_in_lane", "def_in_cone", "keeper_dist",
            "has_frame", "receiver_visible", "ctrl_start", "ctrl_end",
            "epv_start", "epv_end", "def_behind_receiver", "def_within_10"]
    out = {c: np.full(len(ev), np.nan) for c in cols}
    ev = ev.reset_index(drop=True)
    for mid, idx in ev.groupby("match_id").groups.items():
        mf = frames[frames["match_id"] == mid]
        by_id = dict(tuple(mf.groupby("id")))
        for i in idx:
            r = ev.loc[i]
            fr = by_id.get(r["id"])
            teammates, defenders, keeper = pitch.parse_frame(fr)
            if defenders is None:
                out["has_frame"][i] = 0
                continue
            out["has_frame"][i] = 1
            start = (r["start_x"], r["start_y"]); end = (r["end_x"], r["end_y"])
            # EPV = pitch control x threat, at the actor's spot and the action's end
            pts = np.array([start, end], dtype=float)
            ctrl = spacemod.pitch_control(pts, teammates, defenders)
            out["ctrl_start"][i] = ctrl[0]; out["ctrl_end"][i] = ctrl[1]
            out["epv_start"][i] = ctrl[0] * (r["threat_start"] if np.isfinite(r["threat_start"]) else 0.0)
            out["epv_end"][i] = ctrl[1] * (r["threat_end"] if np.isfinite(r["threat_end"]) else 0.0)
            out["def_dist"][i] = pitch.nearest_distance(start, defenders)
            out["n_def_ahead"][i] = pitch.count_ahead(r["start_x"], r["start_y"], defenders)
            out["n_tm_ahead"][i] = pitch.teammates_ahead(r["start_x"], r["start_y"], teammates)
            if r["type"] in ("Pass", "Carry"):
                out["defenders_bypassed"][i] = pitch.count_between(start, end, defenders)
                out["receiver_space"][i] = pitch.nearest_distance(end, defenders)
                out["def_in_lane"][i] = pitch.count_between(start, end, defenders, band=2.0)
                # how many defenders has the RECEIVER got behind (goalside of him),
                # and how crowded is his immediate area
                out["def_behind_receiver"][i] = int((defenders[:, 0] > r["end_x"]).sum())
                out["def_within_10"][i] = int((np.hypot(defenders[:, 0] - r["end_x"],
                                                        defenders[:, 1] - r["end_y"]) <= 10).sum())
                va = fr["visible_area"].iloc[0] if len(fr) else None
                out["receiver_visible"][i] = 1.0 if pitch.point_in_visible_area(end, va) else 0.0
            if r["type"] == "Shot":
                out["def_in_cone"][i] = pitch.count_between(start, config.GOAL_CENTER, defenders, band=8.0)
                out["keeper_dist"][i] = pitch.nearest_distance(config.GOAL_CENTER, keeper[None, :]) if keeper is not None else np.nan
    for c in cols:
        ev[c] = out[c]
    return ev
