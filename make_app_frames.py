"""
Build data/processed/frames_app.parquet -- the freeze-frames the app's run
explorer needs, and nothing else.

Why this exists
---------------
The run explorer draws two freeze-frames per run: where everyone stood when the
pass was struck, and where they stood when it arrived. Those come from the raw
360 caches in data/raw, which total ~240 MB and are not in the repo. Without
them the explorer can only say "freeze-frames not cached for this match", which
is exactly what a deployed copy of the app used to do.

So we keep only the frames that are actually reachable from a run in
runs.parquet, and store them narrowly: the location list becomes two float32
columns, and match_id/id stay so the app can filter the same way it always did.
That turns ~240 MB into a file small enough to commit.

Run after build_runs.py:

    python make_app_frames.py
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd

import config

COLS = ["id", "match_id", "teammate", "actor", "location"]


def _xy(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split a column of [x, y] locations into two float arrays.

    Parquet hands these back as ndarrays rather than lists, so this deliberately
    does not test for `list` -- that assumption has bitten this project before.
    """
    arr = np.full((len(series), 2), np.nan, dtype="float64")
    for i, loc in enumerate(series.to_numpy()):
        if loc is None:
            continue
        try:
            if len(loc) >= 2:
                arr[i, 0] = float(loc[0])
                arr[i, 1] = float(loc[1])
        except TypeError:
            continue
    return arr[:, 0], arr[:, 1]


def main() -> None:
    runs = pd.read_parquet(config.DATA_PROC / "runs.parquet")
    usable = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)]

    needed = pd.unique(
        pd.concat([usable["pass_event_id"], usable["receipt_event_id"]]).dropna()
    )
    needed = set(needed.tolist())
    print(f"runs usable & is_run   : {len(usable):,}")
    print(f"frame event ids needed : {len(needed):,}")

    keep = []
    for cid, sid, label in config.COMPETITIONS:
        fp = config.DATA_RAW / f"frames_{cid}_{sid}.parquet"
        if not fp.exists():
            print(f"  skip (no cache)      : {label}")
            continue
        fr = pd.read_parquet(fp, columns=COLS)
        sub = fr[fr["id"].isin(needed)]
        if len(sub):
            x, y = _xy(sub["location"])
            keep.append(pd.DataFrame({
                "id": sub["id"].astype("string"),
                "match_id": sub["match_id"].astype("int32"),
                "teammate": sub["teammate"].astype("bool"),
                "actor": sub["actor"].fillna(False).astype("bool"),
                "x": x.astype("float32"),
                "y": y.astype("float32"),
            }))
        print(f"  {label:<34} {len(sub):>9,} frame rows kept")
        del fr, sub

    if not keep:
        raise SystemExit("No raw frame caches found -- run build_runs.py first.")

    out = pd.concat(keep, ignore_index=True)
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)

    fp = config.DATA_PROC / "frames_app.parquet"
    out.to_parquet(fp, index=False, compression="zstd")

    covered = out["id"].nunique()
    print(f"\nwrote {fp}")
    print(f"  rows            : {len(out):,}")
    print(f"  distinct events : {covered:,} of {len(needed):,} needed "
          f"({100 * covered / max(len(needed), 1):.1f}%)")
    print(f"  size            : {fp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
