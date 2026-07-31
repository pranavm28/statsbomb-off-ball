"""
StatsBomb open-data loaders with local parquet caching.

Data credit: StatsBomb Open Data, used under the StatsBomb open-data user
agreement (non-commercial / research, attribution required).

The heavy step is pulling events + 360 frames for every match. It is cached to
data/raw/*.parquet so it only happens once. `load_all()` is the single entry
point used by the build pipeline.
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from statsbombpy import sb  # noqa: E402

import config  # noqa: E402

_EV_CACHE = config.DATA_RAW / f"events_{config.COMPETITION_ID}_{config.SEASON_ID}.parquet"
_FR_CACHE = config.DATA_RAW / f"frames_{config.COMPETITION_ID}_{config.SEASON_ID}.parquet"
_MIN_CACHE = config.DATA_RAW / f"minutes_{config.COMPETITION_ID}_{config.SEASON_ID}.parquet"


def match_ids(limit: int | None = None) -> list[int]:
    m = sb.matches(competition_id=config.COMPETITION_ID, season_id=config.SEASON_ID)
    ids = sorted(m["match_id"].astype(int).tolist())
    return ids[:limit] if limit else ids


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def load_events(limit: int | None = None, refresh: bool = False) -> pd.DataFrame:
    if _EV_CACHE.exists() and not refresh and limit is None:
        return pd.read_parquet(_EV_CACHE)
    frames = []
    ids = match_ids(limit)
    for i, mid in enumerate(ids, 1):
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            frames.append(ev)
        except Exception as e:  # keep going; report at the end
            print(f"  ! events failed for {mid}: {e}")
        if i % 8 == 0:
            print(f"  events {i}/{len(ids)}")
    out = pd.concat(frames, ignore_index=True)
    if limit is None:
        out.to_parquet(_EV_CACHE)
    return out


# ---------------------------------------------------------------------------
# 360 freeze frames  (one row per visible player per event)
# ---------------------------------------------------------------------------
def load_frames(limit: int | None = None, refresh: bool = False) -> pd.DataFrame:
    if _FR_CACHE.exists() and not refresh and limit is None:
        return pd.read_parquet(_FR_CACHE)
    frames = []
    ids = match_ids(limit)
    for i, mid in enumerate(ids, 1):
        try:
            fr = sb.frames(match_id=mid, fmt="dataframe")
            frames.append(fr)
        except Exception as e:
            print(f"  ! frames failed for {mid}: {e}")
        if i % 8 == 0:
            print(f"  frames {i}/{len(ids)}")
    out = pd.concat(frames, ignore_index=True)
    # store location/visible_area as plain lists so parquet is happy
    if limit is None:
        out.to_parquet(_FR_CACHE)
    return out


# ---------------------------------------------------------------------------
# minutes played (for per-90 metrics) -- derived from events
# ---------------------------------------------------------------------------
def compute_minutes(events: pd.DataFrame, refresh: bool = False,
                    persist: bool = True) -> pd.DataFrame:
    """
    Minutes per player per match from Starting XI + substitution events.
    Tournament-robust: uses the last event minute of each match as full time.

    `persist` guards the cache: partial runs (a match subset) must NOT write
    the cache, or a later full run would silently reuse subset minutes.
    """
    if _MIN_CACHE.exists() and not refresh and persist:
        return pd.read_parquet(_MIN_CACHE)

    rows = []
    for mid, mev in events.groupby("match_id"):
        full_time = mev["minute"].max() + 1
        # starters from the two Starting XI tactics events
        starters = {}
        for _, r in mev[mev["type"] == "Starting XI"].iterrows():
            tac = r.get("tactics")
            if isinstance(tac, dict):
                for p in tac.get("lineup", []):
                    starters[p["player"]["name"]] = (p["player"]["id"], r["team"], 0)
        on = {name: (pid, team, start) for name, (pid, team, start) in starters.items()}
        end = {name: full_time for name in on}
        # substitutions
        subs = mev[mev["type"] == "Substitution"]
        for _, r in subs.iterrows():
            off_name = r["player"]
            minute = r["minute"]
            if off_name in end:
                end[off_name] = minute
            repl = r.get("substitution_replacement")
            if isinstance(repl, str):
                on[repl] = (r.get("substitution_replacement_id"), r["team"], minute)
                end[repl] = full_time
        for name, (pid, team, start) in on.items():
            rows.append(dict(match_id=mid, player=name, player_id=pid, team=team,
                             minutes=max(0, end.get(name, full_time) - start)))
    out = pd.DataFrame(rows)
    out = out.groupby(["player", "player_id"], as_index=False).agg(
        minutes=("minutes", "sum"),
        team=("team", "last"),
        matches=("match_id", "nunique"),
    )
    if persist:
        out.to_parquet(_MIN_CACHE)
    return out


def load_all(limit: int | None = None, refresh: bool = False):
    print("Loading events ...")
    ev = load_events(limit=limit, refresh=refresh)
    print(f"  {len(ev):,} events over {ev['match_id'].nunique()} matches")
    print("Loading 360 frames ...")
    fr = load_frames(limit=limit, refresh=refresh)
    print(f"  {len(fr):,} freeze-frame player rows")
    print("Computing minutes ...")
    mins = compute_minutes(ev, refresh=refresh or limit is not None,
                           persist=(limit is None))
    return ev, fr, mins


# ---------------------------------------------------------------------------
# Multi-competition loading (the pooled club seasons)
# ---------------------------------------------------------------------------
def _cache(kind, cid, sid):
    return config.DATA_RAW / f"{kind}_{cid}_{sid}.parquet"


def load_competition(cid: int, sid: int, label: str, refresh: bool = False):
    """Events + frames for one competition-season, cached per competition."""
    ev_fp, fr_fp = _cache("events", cid, sid), _cache("frames", cid, sid)
    if ev_fp.exists() and fr_fp.exists() and not refresh:
        ev, fr = pd.read_parquet(ev_fp), pd.read_parquet(fr_fp)
    else:
        m = sb.matches(competition_id=cid, season_id=sid)
        ids = sorted(m["match_id"].astype(int).tolist())
        evs, frs = [], []
        for i, mid in enumerate(ids, 1):
            try:
                e = sb.events(match_id=mid); e["match_id"] = mid; evs.append(e)
                frs.append(sb.frames(match_id=mid, fmt="dataframe"))
            except Exception as exc:
                print(f"  ! {label} match {mid}: {exc}")
            if i % 10 == 0:
                print(f"  {label}: {i}/{len(ids)}")
        ev = pd.concat(evs, ignore_index=True)
        fr = pd.concat(frs, ignore_index=True)
        ev.to_parquet(ev_fp); fr.to_parquet(fr_fp)
    ev["competition"] = label
    fr["competition"] = label
    return ev, fr


def load_pooled(refresh: bool = False):
    """Load and concatenate every competition in config.COMPETITIONS."""
    evs, frs = [], []
    for cid, sid, label in config.COMPETITIONS:
        print(f"Loading {label} ...")
        e, f = load_competition(cid, sid, label, refresh=refresh)
        print(f"  {len(e):,} events | {len(f):,} frame rows | {e['match_id'].nunique()} matches")
        evs.append(e); frs.append(f)
    events = pd.concat(evs, ignore_index=True)
    frames = pd.concat(frs, ignore_index=True)
    print(f"POOLED: {events['match_id'].nunique()} matches, {len(events):,} events")
    minutes = compute_minutes(events, refresh=True, persist=False)
    return events, frames, minutes
