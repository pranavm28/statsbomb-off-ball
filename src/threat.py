"""
Threat surface -- "how dangerous is the ball at this location".

A grid Expected-Threat (xT) model (Karun-Singh style): each cell's value is the
probability-weighted value of what a team can do from there, solved by value
iteration. This is the interpretable "danger" layer that the pitch-control
surface is multiplied against to get Expected Possession Value (EPV).

Exposes both the coarse grid and a fast lookup on any (x, y) or a full fine grid,
so EPV can be evaluated on the same resolution as pitch control.
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd

import config


def _xy(loc):
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def _cell(x, y, nx, ny):
    cx = np.clip((x / config.PITCH_LENGTH * nx).astype(int), 0, nx - 1)
    cy = np.clip((y / config.PITCH_WIDTH * ny).astype(int), 0, ny - 1)
    return cx, cy


def build_threat(events: pd.DataFrame):
    """Fit the xT threat surface from move/shot events. Returns the (NX, NY) grid."""
    NX, NY = config.THREAT_NX, config.THREAT_NY
    ev = events.copy()
    ev[["x", "y"]] = ev["location"].apply(lambda l: pd.Series(_xy(l)))
    ev = ev.dropna(subset=["x", "y"])

    is_move = ev["type"].isin(["Pass", "Carry"])
    move_ok = is_move & ~((ev["type"] == "Pass") & ev["pass_outcome"].notna())
    cx, cy = _cell(ev["x"].values, ev["y"].values, NX, NY)
    ev["cx"], ev["cy"] = cx, cy

    shots = np.zeros((NX, NY)); moves = np.zeros((NX, NY)); goalp = np.zeros((NX, NY))
    for (i, j), g in ev.groupby(["cx", "cy"]):
        shots[i, j] = (g["type"] == "Shot").sum()
        moves[i, j] = g.index.isin(ev[move_ok].index).sum()
        sg = g[g["type"] == "Shot"]
        if len(sg):
            goalp[i, j] = sg["shot_statsbomb_xg"].fillna(0).mean()

    total = shots + moves
    with np.errstate(invalid="ignore", divide="ignore"):
        p_shot = np.where(total > 0, shots / total, 0.0)
        p_move = np.where(total > 0, moves / total, 0.0)

    # transition matrix from completed moves
    T = np.zeros((NX, NY, NX, NY))
    mv = ev[move_ok].copy()
    mv[["ex", "ey"]] = mv.apply(
        lambda r: pd.Series(_xy(r["pass_end_location"] if r["type"] == "Pass"
                                 else r["carry_end_location"])), axis=1)
    mv = mv.dropna(subset=["ex", "ey"])
    ecx, ecy = _cell(mv["ex"].values, mv["ey"].values, NX, NY)
    for (i, j, k, l) in zip(mv["cx"].values, mv["cy"].values, ecx, ecy):
        T[i, j, k, l] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        Tsum = T.sum(axis=(2, 3), keepdims=True)
        T = np.where(Tsum > 0, T / Tsum, 0.0)

    xt = np.zeros((NX, NY))
    shot_value = p_shot * goalp
    for _ in range(config.XT_ITERATIONS):
        xt = shot_value + p_move * np.einsum("ijkl,kl->ij", T, xt)
    return xt


def threat_at(grid, x, y) -> float:
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    cx = int(np.clip(x / config.PITCH_LENGTH * config.THREAT_NX, 0, config.THREAT_NX - 1))
    cy = int(np.clip(y / config.PITCH_WIDTH * config.THREAT_NY, 0, config.THREAT_NY - 1))
    return float(grid[cx, cy])


def threat_on_grid(grid, gx, gy):
    """Vectorised threat lookup for arrays of pitch coordinates gx, gy."""
    cx = np.clip((gx / config.PITCH_LENGTH * config.THREAT_NX).astype(int), 0, config.THREAT_NX - 1)
    cy = np.clip((gy / config.PITCH_WIDTH * config.THREAT_NY).astype(int), 0, config.THREAT_NY - 1)
    return grid[cx, cy]
