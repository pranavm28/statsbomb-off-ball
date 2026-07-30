"""
The space engine: pitch control -> Expected Possession Value (EPV) surface.

PITCH CONTROL (positional). Each player exerts an isotropic Gaussian influence
that decays with distance; a team's control at a point is its share of total
influence there:

    control_att(x) = sum_j exp(-||x - att_j||^2 / 2s^2)
                     -----------------------------------------------
                     sum_j att_infl + sum_k def_infl        (in [0,1])

This is deliberately velocity-free: StatsBomb 360 is a single freeze-frame with
no player velocities, so a Spearman time-to-arrive model is not identifiable.
Stated as a limitation; with tracking data we would orient each influence by
velocity. (Fernandez & Bornn, "Wide Open Spaces"; Spearman pitch control.)

EPV = control_att(x) * threat(x). Summed over the pitch it is the amount of
*dangerous space a team controls* -- the scalar the metrics are built on.
"""
from __future__ import annotations
import numpy as np

import config
from src import threat as threatmod


def make_grid(nx: int, ny: int):
    """Cell-centre coordinates of an nx-by-ny pitch grid, flattened to (M, 2)."""
    xs = (np.arange(nx) + 0.5) / nx * config.PITCH_LENGTH
    ys = (np.arange(ny) + 0.5) / ny * config.PITCH_WIDTH
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cell_area = (config.PITCH_LENGTH / nx) * (config.PITCH_WIDTH / ny)
    return pts, gx, gy, cell_area


def _influence(pts, players, sigma):
    """Sum of Gaussian influence of `players` at each grid point -> (M,)."""
    if players is None or len(players) == 0:
        return np.zeros(len(pts))
    d2 = ((pts[:, None, 0] - players[None, :, 0]) ** 2 +
          (pts[:, None, 1] - players[None, :, 1]) ** 2)
    return np.exp(-d2 / (2 * sigma ** 2)).sum(axis=1)


def pitch_control(pts, attackers, defenders, sigma=None):
    """Attacking team's control probability at each grid point -> (M,) in [0,1]."""
    sigma = config.PC_SIGMA if sigma is None else sigma
    a = _influence(pts, attackers, sigma)
    d = _influence(pts, defenders, sigma)
    return a / (a + d + 1e-9)


def epv_surface(pts, attackers, defenders, threat_grid, sigma=None):
    """EPV = control x threat at each grid point -> (M,)."""
    ctrl = pitch_control(pts, attackers, defenders, sigma)
    thr = threatmod.threat_on_grid(threat_grid, pts[:, 0], pts[:, 1])
    return ctrl * thr, ctrl, thr


def controlled_threat(pts, attackers, defenders, threat_grid, cell_area, sigma=None):
    """Scalar: total dangerous space the attacking team controls (sum of EPV)."""
    epv, _, _ = epv_surface(pts, attackers, defenders, threat_grid, sigma)
    return float(epv.sum() * cell_area)


def player_space(pts, attackers, defenders, threat_grid, cell_area, sigma=None):
    """
    Attribute controlled dangerous space to each attacker by Voronoi ownership
    (nearest attacker owns the cell). Returns an array aligned to `attackers`.
    """
    if attackers is None or len(attackers) == 0:
        return np.array([])
    epv, _, _ = epv_surface(pts, attackers, defenders, threat_grid, sigma)
    d2 = ((pts[:, None, 0] - attackers[None, :, 0]) ** 2 +
          (pts[:, None, 1] - attackers[None, :, 1]) ** 2)
    owner = d2.argmin(axis=1)
    out = np.zeros(len(attackers))
    for a in range(len(attackers)):
        out[a] = epv[owner == a].sum() * cell_area
    return out


def denial_counterfactual(pts, attackers, defenders, threat_grid, cell_area, sigma=None):
    """
    Space-denial value of each defender = increase in the attacking team's
    controlled dangerous space if that defender were removed (leave-one-out).
    Returns an array aligned to `defenders` (higher = denies more danger).
    """
    if defenders is None or len(defenders) == 0:
        return np.array([])
    base = controlled_threat(pts, attackers, defenders, threat_grid, cell_area, sigma)
    out = np.zeros(len(defenders))
    for k in range(len(defenders)):
        without = np.delete(defenders, k, axis=0)
        out[k] = controlled_threat(pts, attackers, without, threat_grid, cell_area, sigma) - base
    return out
