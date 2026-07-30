"""
Pitch geometry and 360 freeze-frame feature extraction.

All coordinates are StatsBomb (120 x 80), and StatsBomb open data is
normalised so the team in possession always attacks toward x=120 (verified
empirically from shot locations -- see README). Every freeze-frame here is
already in that same "attack -> right" frame, so:
    * the goal being attacked is at (120, 40)
    * "toward goal" means larger x
    * opponents (defenders) are freeze-frame players with teammate == False

No shapely dependency: point-in-polygon and distances are done with numpy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config

GOAL_X, GOAL_Y = config.GOAL_CENTER


# ---------------------------------------------------------------------------
# basic geometry
# ---------------------------------------------------------------------------
def dist_to_goal(x: float, y: float) -> float:
    return float(np.hypot(GOAL_X - x, GOAL_Y - y))


def angle_to_goal(x: float, y: float) -> float:
    """Angle (radians) subtended by the goal mouth from (x, y). Wider = better."""
    # goalposts at y = 36 and y = 44 on the x=120 line
    g = 3.66  # half goal width in metres-ish on the 80-wide pitch
    a = np.hypot(GOAL_X - x, (GOAL_Y - g) - y)
    b = np.hypot(GOAL_X - x, (GOAL_Y + g) - y)
    c = 2 * g
    cos = (a * a + b * b - c * c) / (2 * a * b + 1e-9)
    return float(np.arccos(np.clip(cos, -1, 1)))


def parse_frame(frame_rows: pd.DataFrame):
    """
    Split a per-event group of 360 rows into (teammates, defenders, keeper_def).

    Returns numpy arrays of [x, y] for the actor's teammates and for opponents,
    plus the defending keeper location (or None).
    """
    if frame_rows is None or len(frame_rows) == 0:
        return None, None, None
    locs = np.array([l for l in frame_rows["location"].values], dtype=float)
    tm = frame_rows["teammate"].values.astype(bool)
    kp = frame_rows["keeper"].values.astype(bool)
    teammates = locs[tm]
    defenders = locs[~tm]
    # defending keeper = keeper who is NOT a teammate of the actor
    def_kp = locs[(~tm) & kp]
    keeper = def_kp[0] if len(def_kp) else None
    return teammates, defenders, keeper


def nearest_distance(pt, others) -> float:
    if others is None or len(others) == 0:
        return np.nan
    d = np.hypot(others[:, 0] - pt[0], others[:, 1] - pt[1])
    return float(d.min())


def count_between(start, end, defenders, band: float = None) -> int:
    """
    Count defenders 'bypassed' (packing) by an action from start -> end:
    opponents whose x lies between start_x and end_x (i.e. goalside of the
    ball at release but ballside of the target) and within a lateral band of
    the straight line between the two points.
    """
    if defenders is None or len(defenders) == 0:
        return 0
    band = config.PACKING_BAND if band is None else band
    sx, sy = start
    ex, ey = end
    lo, hi = min(sx, ex), max(sx, ex)
    # candidate defenders sitting between the two x-coordinates
    inx = (defenders[:, 0] >= lo) & (defenders[:, 0] <= hi)
    if not inx.any():
        return 0
    cand = defenders[inx]
    # perpendicular distance from the passing line
    dx, dy = ex - sx, ey - sy
    seg_len = np.hypot(dx, dy) + 1e-9
    perp = np.abs((cand[:, 0] - sx) * dy - (cand[:, 1] - sy) * dx) / seg_len
    return int((perp <= band).sum())


def count_ahead(x, y, defenders, band: float = 12.0) -> int:
    """Opponents goalside of the ball (larger x) within a central lateral band."""
    if defenders is None or len(defenders) == 0:
        return 0
    ahead = (defenders[:, 0] > x) & (np.abs(defenders[:, 1] - y) <= band + 20)
    return int(ahead.sum())


def teammates_ahead(x, y, teammates, band: float = 12.0) -> int:
    if teammates is None or len(teammates) == 0:
        return 0
    ahead = (teammates[:, 0] > x) & (np.abs(teammates[:, 1] - y) <= band + 20)
    return int(ahead.sum())


def point_in_visible_area(pt, visible_area) -> bool:
    """Ray-casting point-in-polygon; visible_area is a flat [x0,y0,x1,y1,...]."""
    if visible_area is None or (isinstance(visible_area, float)):
        return True
    try:
        poly = np.array(visible_area, dtype=float).reshape(-1, 2)
    except Exception:
        return True
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Defensive compactness / encirclement
# ---------------------------------------------------------------------------
def compactness(pt, defenders, radii=(5.0, 10.0), near_radius=15.0):
    """
    How SURROUNDED is a player at `pt`, and does he have a way out?

    Nearest-defender distance alone is a poor description of pressure: one marker
    two metres away is treated identically to three defenders closing from three
    sides. These features separate those cases.

    Returns
      n_within_<r>    defenders inside each radius            -- local density
      second_nearest  distance to the 2nd nearest defender    -- marked vs swarmed
      encirclement    1 - (largest angular gap / 2pi) among defenders inside
                      `near_radius`. 0 means a clear escape side; towards 1 means
                      bodies on every side.
    """
    out = {f"n_within_{int(r)}": 0 for r in radii}
    out["second_nearest"] = np.nan
    out["encirclement"] = 0.0
    if defenders is None or len(defenders) == 0:
        return out

    d = np.hypot(defenders[:, 0] - pt[0], defenders[:, 1] - pt[1])
    for r in radii:
        out[f"n_within_{int(r)}"] = int((d <= r).sum())
    if len(d) >= 2:
        out["second_nearest"] = float(np.sort(d)[1])

    near = defenders[d <= near_radius]
    if len(near) >= 2:
        ang = np.sort(np.arctan2(near[:, 1] - pt[1], near[:, 0] - pt[0]))
        gaps = np.diff(ang)
        # the wrap-around gap closes the circle
        wrap = (ang[0] + 2 * np.pi) - ang[-1]
        largest = float(max(gaps.max(), wrap))
        out["encirclement"] = float(1.0 - largest / (2 * np.pi))
    return out


def block_spread(defenders):
    """Spread of the defending unit -- a compact block vs a stretched one."""
    if defenders is None or len(defenders) < 3:
        return np.nan
    return float(np.hypot(defenders[:, 0].std(), defenders[:, 1].std()))
