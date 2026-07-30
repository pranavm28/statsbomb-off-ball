"""
AI-ASSISTED MODULE, written to my specification. The outcome set (shots in 5s,
retention over 10s, possession lost, and the possession VALUE lost weighted by where the
turnover happened) was specified by me. See AI_USAGE.md section 2.8.

TIME-based outcomes for a possession state.

Everything else in this project counts in *actions* ("within the next 5 actions").
Actions are a poor clock: five tippy-tappy passes take three seconds, five duels
take thirty. For valuing a RUN, real time is the honest unit -- a run either
produces a shot in the next few seconds or it does not.

For a state observed at (match, period, t, team) we compute, in seconds:

    shot_5s        team takes a shot within 5s                 -> the value target
    xg_5s          sum of StatsBomb xG on those shots          -> xG added after the run
    retain_10s     team still has the ball 10s later           -> retention (team ctx)
    lost_5s        possession turned over within 5s            -> risk
    pv_lost_5s     threat surrendered at the turnover          -> value lost, not just count

`pv_lost_5s` is the piece people usually skip: losing the ball on the halfway line
and losing it while your full-backs are 60 yards up are not the same event, so we
value the loss by the threat of the position it happened in.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from src import threat as threatmod

SHOT_WINDOW = 5.0      # seconds
RETAIN_WINDOW = 10.0   # seconds
LOSS_WINDOW = 5.0      # seconds


def _abs_time(df):
    """Seconds within the period (StatsBomb `minute` is cumulative match time)."""
    return df["minute"].astype(float) * 60.0 + df["second"].astype(float)


def attach_outcomes(states: pd.DataFrame, events: pd.DataFrame, threat_grid,
                    time_col="t_sec", team_col="team") -> pd.DataFrame:
    """
    Add the time-window outcome columns to any table of possession states that
    carries match_id, period, `time_col` and `team_col`.
    """
    ev = events.copy()
    ev["t_sec"] = _abs_time(ev)
    ev[["ex", "ey"]] = ev["location"].apply(
        lambda l: pd.Series((float(l[0]), float(l[1]))
                            if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 2
                            else (np.nan, np.nan)))
    ev["is_shot"] = (ev["type"] == "Shot").astype(int)
    ev["xg"] = ev.get("shot_statsbomb_xg", pd.Series(0.0, index=ev.index)).fillna(0.0)

    out = {c: np.full(len(states), np.nan) for c in
           ["shot_5s", "xg_5s", "retain_10s", "lost_5s", "pv_lost_5s"]}
    st = states.reset_index(drop=True)

    for (mid, per), idx in st.groupby(["match_id", "period"]).groups.items():
        e = ev[(ev["match_id"] == mid) & (ev["period"] == per)].sort_values("t_sec")
        if len(e) == 0:
            continue
        et = e["t_sec"].to_numpy()
        e_team = e["possession_team"].to_numpy()
        e_shot = e["is_shot"].to_numpy()
        e_shot_team = e["team"].to_numpy()
        e_xg = e["xg"].to_numpy()
        ex, ey = e["ex"].to_numpy(), e["ey"].to_numpy()

        for i in idx:
            t = float(st.at[i, time_col]); team = st.at[i, team_col]
            lo = np.searchsorted(et, t, side="left")

            # --- shots by this team in the next 5s -----------------------
            hi5 = np.searchsorted(et, t + SHOT_WINDOW, side="right")
            w = slice(lo, hi5)
            sh = (e_shot[w] == 1) & (e_shot_team[w] == team)
            out["shot_5s"][i] = float(sh.any())
            out["xg_5s"][i] = float(e_xg[w][sh].sum()) if sh.any() else 0.0

            # --- possession lost within 5s, and the threat surrendered ---
            other = e_team[w] != team
            out["lost_5s"][i] = float(other.any())
            if other.any():
                k = np.argmax(other)                       # first turnover in window
                lx, ly = ex[w][k], ey[w][k]
                out["pv_lost_5s"][i] = (threatmod.threat_at(threat_grid, lx, ly)
                                        if np.isfinite(lx) and np.isfinite(ly) else 0.0)
            else:
                out["pv_lost_5s"][i] = 0.0

            # --- still in possession 10s later ---------------------------
            hi10 = np.searchsorted(et, t + RETAIN_WINDOW, side="right")
            w10 = slice(lo, hi10)
            out["retain_10s"][i] = float(not (e_team[w10] != team).any())

    for c, v in out.items():
        st[c] = v
    return st
