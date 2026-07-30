"""
Pooled build: four club seasons of StatsBomb 360 -> off-ball runs -> possession
value -> player/team metrics.

    python build_runs.py            # full build (first run downloads ~170 MB, then cached)
    python build_runs.py --limit 4  # quick slice for a smoke test

Stages
  1  load the four competitions (cached per competition)
  2  fit the xT threat surface
  3  build the action table (used for action-window labels)
  4  RECONSTRUCT RUNS from the release/receipt freeze-frame pairs
  5  fit V(state) = P(shot within 5s) and take delta V across each run
  6  attach time-window outcomes (retention, loss, value lost, xG)
  7  aggregate player + team metrics
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd

import config
from src import data as datamod
from src import threat as threatmod
from src import features as feat
from src import runs as runsmod
from src import value as valmod
from src import outcomes as outmod
from src import run_metrics as rm
from src import ablation as abl


def main(limit=None):
    t0 = time.time()

    print("[1/7] loading competitions ...")
    events, frames, minutes = datamod.load_pooled()
    if limit:
        keep = sorted(events["match_id"].unique())[:limit]
        events = events[events["match_id"].isin(keep)]
        frames = frames[frames["match_id"].isin(keep)]
        minutes = datamod.compute_minutes(events, refresh=True, persist=False)
    minutes.to_parquet(config.DATA_PROC / "minutes.parquet")

    print("[2/7] fitting threat surface ...")
    threat_grid = threatmod.build_threat(events)
    np.save(config.DATA_PROC / "threat_grid.npy", threat_grid)

    print("[3/7] building action table ...")
    actions = feat.build_action_table(events, frames, threat_grid)
    actions.to_parquet(config.DATA_PROC / "actions.parquet")

    print("[4/7] reconstructing off-ball runs ...")
    runs = runsmod.build_runs(events, frames, threat_grid)
    runs = runsmod.apply_pitch_bounds(runs)
    runs = runsmod.add_game_state(runs, events)
    runs = runsmod.label_outcomes(runs, actions)
    comp = events[["match_id", "competition"]].drop_duplicates()
    runs = runs.merge(comp, on="match_id", how="left")
    print(f"      {len(runs):,} runs | usable {runs['usable'].mean():.0%} "
          f"| implausible {(1 - runs['plausible'].mean()):.1%}")

    print("[5/7] fitting possession value V(state) and delta V ...")
    states = valmod.build_states(runs, events, threat_grid)
    scored, vreport, _ = valmod.fit_value(states, target="shot_5s")
    runs = valmod.delta_v(scored, runs)
    json.dump(vreport, open(config.OUTPUTS / "value_report.json", "w"), indent=2)
    print(f"      V base AUC {vreport['metrics_base']['auc']:.4f} -> "
          f"+360 {vreport['metrics_plus_360']['auc']:.4f} "
          f"(lift {vreport['auc_lift']:+.4f})")

    print("      running the information-tier ablation ...")
    states.to_parquet(config.DATA_PROC / "states.parquet")
    ab = abl.run_ablation(states)
    json.dump(ab, open(config.OUTPUTS / "ablation_report.json", "w"), indent=2)
    for t in ab["tiers"]:
        g = "" if t["gain_vs_previous"] is None else f"  (+{t['gain_vs_previous']:.4f})"
        print(f"        {t['tier']:32} AUC {t['auc']:.4f}{g}")
    print(f"        TOTAL gain attributable to 360: {ab['total_gain_from_360']:+.4f}")

    print("[6/7] attaching time-window outcomes to each run ...")
    runs["t_sec"] = runs["minute"] * 60.0 + runs["second"]
    runs = outmod.attach_outcomes(runs, events, threat_grid)

    print("[7/7] fitting run-value model + aggregating ...")
    use = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()
    scored_runs, rreport, _ = runsmod.fit_run_value(use)
    runs = runs.merge(scored_runs[["match_id", "index", "run_value", "rv_context",
                                   "run_value_added"]], on=["match_id", "index"], how="left")
    runs.to_parquet(config.DATA_PROC / "runs.parquet")
    json.dump(rreport, open(config.OUTPUTS / "run_report.json", "w"), indent=2)

    players = rm.player_run_metrics(runs, minutes, events)
    teams = rm.team_run_metrics(runs)
    players.to_csv(config.OUTPUTS / "run_player_metrics.csv", index=False)
    teams.to_csv(config.OUTPUTS / "run_team_metrics.csv", index=False)

    print(f"\nDone in {time.time() - t0:.0f}s | {len(players)} qualified players")
    cols = ["runner", "team", "pos_group", "minutes", "DeltaV90", "RunThreat90",
            "Shot5s90", "xGAdded90", "InBehind90"]
    cols = [c for c in cols if c in players.columns]
    print(players.sort_values(cols[4], ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
