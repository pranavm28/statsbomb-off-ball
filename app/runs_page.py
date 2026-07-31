"""
The off-ball runs page: leaderboard, run maps, situational splits, and the
validation of whether run features add anything over context.

Kept in its own module so the main app file stays readable and every function
here is small enough to talk through line-by-line in a walkthrough.
"""
# Implementation written by Claude Code under my direction, then reviewed and
# corrected line by line. Design decisions, thresholds and validation are mine.
# See AI_USAGE.md for the split of work and the errors I caught.
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import viz


def run_map(runs_p: pd.DataFrame, colour_by="in_behind", title=None, emphasise=False):
    """
    A player's reconstructed runs. Runs are OFF-BALL movement, so they follow the
    house CARRY convention -- dotted, with a dot at the receiving end -- not the
    comet used for passes. The distinction is deliberate: a comet would imply the
    player had the ball.

    emphasise=True pushes the bulk category into the background: lower alpha,
    lower zorder, thinner line, smaller dot. A high-volume player has hundreds of
    ordinary runs and a handful of interesting ones, and at equal weight the
    ordinary ones bury the rest. Only the categories flagged as focus keep full
    weight, and they are drawn last so they sit on top.
    """
    fig, ax, pitch = viz.new_pitch(figsize=(9, 6))

    # (label, rows, colour, is_focus)
    if colour_by == "in_behind":
        groups = [("In behind the line", runs_p[runs_p["in_behind"] == 1], viz.PROG_PASS, True),
                  ("In front of the line", runs_p[runs_p["in_behind"] == 0], viz.PASS_SUCC, False)]
    elif colour_by == "game_state":
        # no category here is "the interesting one", so nothing is demoted
        groups = [("Trailing", runs_p[runs_p["game_state"] == "Trailing"], viz.PROG_CARRY, True),
                  ("Level", runs_p[runs_p["game_state"] == "Level"], viz.AMBER, True),
                  ("Leading", runs_p[runs_p["game_state"] == "Leading"], viz.PROG_PASS, True)]
    else:  # threat generated
        hi = runs_p[runs_p["run_xt"] > runs_p["run_xt"].quantile(0.75)]
        lo = runs_p[runs_p["run_xt"] <= runs_p["run_xt"].quantile(0.75)]
        groups = [("Low threat", lo, viz.PASS_FAIL, False),
                  ("High threat", hi, viz.PROG_PASS, True)]

    # background first, focus last, so the runs worth reading sit on top
    for label, g, col, focus in sorted(groups, key=lambda t: t[3]):
        if not len(g):
            continue
        if emphasise and not focus:
            viz.dotted(pitch, ax, g["origin_x"], g["origin_y"], g["receipt_x"], g["receipt_y"],
                       col, lw=1.3, alpha=0.22, z=3, dot=13)
        else:
            viz.dotted(pitch, ax, g["origin_x"], g["origin_y"], g["receipt_x"], g["receipt_y"],
                       col, lw=2.1, alpha=0.95, z=6, dot=42)

    # legend keeps the declared order, which reads better than the draw order
    items = [(f"{label}: {len(g)}", col, "dotted") for label, g, col, _ in groups if len(g)]
    viz.legend(ax, items)
    if title:
        viz.title(fig, title, "runs shown from where the player started to where he received")
    return fig


# Display surnames. Taking the last token is wrong for Iberian and Brazilian naming:
# "Kylian Mbappe Lottin" -> "Lottin", "Neymar da Silva Santos Junior" -> "Junior".
_SURNAMES = [("Mbapp", "Mbappé"), ("Neymar", "Neymar"), ("Braithwaite", "Braithwaite"),
             ("Trinc", "Trincão"), ("Messi", "Messi"), ("Alba", "Jordi Alba"),
             ("Boniface", "Boniface"), ("Wirtz", "Wirtz"), ("Dembélé", "Dembélé"),
             ("Hakimi", "Hakimi"), ("Frimpong", "Frimpong"), ("Ekitike", "Ekitike")]


def _surname(name: str) -> str:
    if not isinstance(name, str):
        return "?"
    for needle, disp in _SURNAMES:
        if needle in name:
            return disp
    return name.split()[-1] if name.split() else name


def _frame_players(fr, event_id):
    """Teammate and opponent coordinates for one freeze-frame."""
    f = fr[fr["id"] == event_id]
    if len(f) == 0:
        return None, None
    locs = np.column_stack([f["x"].to_numpy(float), f["y"].to_numpy(float)])
    tm = f["teammate"].to_numpy(bool)
    return locs[tm], locs[~tm]


@st.cache_data(show_spinner=False)
def _load_all_frames():
    """
    Every freeze-frame the run explorer can reach, normalised to x/y columns.

    Prefers data/processed/frames_app.parquet -- the slim extract built by
    make_app_frames.py, which covers all reachable frames in ~19 MB and is
    committed, so a deployed copy has them. Falls back to the full raw 360
    caches when those are present, which keeps a local checkout unchanged.
    """
    slim = config.DATA_PROC / "frames_app.parquet"
    if slim.exists():
        return pd.read_parquet(slim, columns=["id", "match_id", "teammate", "x", "y"])

    parts = []
    for cid, sid, _ in config.COMPETITIONS:
        fp = config.DATA_RAW / f"frames_{cid}_{sid}.parquet"
        if not fp.exists():
            continue
        fr = pd.read_parquet(fp, columns=["id", "match_id", "teammate", "location"])
        xy = np.full((len(fr), 2), np.nan)
        for i, loc in enumerate(fr["location"].to_numpy()):
            if loc is not None and len(loc) >= 2:
                xy[i] = (float(loc[0]), float(loc[1]))
        fr = fr.drop(columns=["location"])
        fr["x"], fr["y"] = xy[:, 0], xy[:, 1]
        parts.append(fr)
    return pd.concat(parts, ignore_index=True) if parts else None


def _load_frames_for(match_id: int):
    """Freeze-frames for one match, or None if this match has none."""
    fr = _load_all_frames()
    if fr is None:
        return None
    sub = fr[fr["match_id"] == match_id]
    return sub if len(sub) else None


def _run_explorer(runs: pd.DataFrame):
    """The presentation piece: one run, both freeze-frames, both trajectories."""
    st.markdown("**Pick a run and watch it.** Ghosted dots are where everyone stood when "
                "the pass was struck; solid dots are where they stood when it arrived. "
                "The gold comet is the ball; the green dotted line is the player's run.")

    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()
    c1, c2, c3 = st.columns(3)
    # Run Value first: it is the metric the leaderboard is built from, so it should
    # be the default way of finding runs to look at.
    rank_by = c1.selectbox("Find runs by",
                           ["run_value_added", "delta_V", "run_xt", "xg_5s", "run_distance"],
                           format_func=lambda x: {"run_value_added": "Run Value (this run)",
                                                  "delta_V": "Pass + run together (ΔV)",
                                                  "run_xt": "Threat added (xT)",
                                                  "xg_5s": "xG in the next 5s",
                                                  "run_distance": "Run distance"}[x])
    ph = c2.selectbox("Phase", ["All", "Counter", "Open play", "Set piece / other"])
    who = c3.selectbox("Player", ["All"] + sorted(r["runner"].dropna().unique().tolist()))
    st.caption("**Run Value** is this run's own credit — how much the *manner* of the movement "
               "raised the chance the possession progresses, over and above where he ran between. "
               "It is the per-run number that sums into Run Value / 90 on the Leaderboard. "
               "**ΔV** is a different quantity: the pass and the run valued together.")

    dctx = st.radio("Defensive context", ["All runs", "Received enclosed (encirclement ≥ 0.5)",
                                          "Received with a clear side (< 0.2)",
                                          "Into a crowd (3+ defenders within 10 m)"],
                    horizontal=True)
    if ph != "All":
        r = r[r["phase"] == ph]
    if dctx.startswith("Received enclosed"):
        r = r[r["encirclement"] >= 0.5]
    elif dctx.startswith("Received with"):
        r = r[r["encirclement"] < 0.2]
    elif dctx.startswith("Into a crowd"):
        r = r[r["def_within_10"] >= 3]
    if who != "All":
        r = r[r["runner"] == who]
    if rank_by not in r.columns or r[rank_by].notna().sum() == 0:
        st.warning("No runs available for that filter."); return
    top = r.nlargest(60, rank_by)
    if len(top) == 0:
        st.warning("No runs available for that filter."); return

    top = top.copy()
    # Label with the metric actually being sorted on, not always dV -- otherwise the
    # list looks unsorted. Surnames come from a lookup rather than the last token:
    # "Kylian Mbappe Lottin" would otherwise display as "Lottin".
    metric_label = {"run_value_added": "Run Value", "delta_V": "ΔV", "run_xt": "xT",
                    "xg_5s": "xG 5s", "run_distance": "dist"}[rank_by]
    fmt = "{:.1f} m" if rank_by == "run_distance" else "{:+.3f}"
    top["label"] = (top["runner"].map(_surname) + "  ·  " + top["team"]
                    + "  ·  " + top["minute"].astype(int).astype(str) + "'"
                    + f"  ·  {metric_label} " + top[rank_by].map(fmt.format))
    pick = st.selectbox("Run", top["label"].tolist())
    row = top[top["label"] == pick].iloc[0]

    fr = _load_frames_for(int(row["match_id"]))
    left, right = st.columns([1.75, 1])
    with left:
        if fr is None:
            st.warning(
                "No freeze-frames available for this match. Build them with "
                "`python make_app_frames.py` (needs the raw 360 caches from "
                "`python build_runs.py`).")
        else:
            mates_rel, defs_rel = _frame_players(fr, row["pass_event_id"])
            mates_rec, defs_rec = _frame_players(fr, row["receipt_event_id"])
            fig = viz.run_explorer(
                mates_rel, defs_rel, mates_rec, defs_rec,
                (row["ball_before_x"], row["ball_before_y"]),
                (row["origin_x"], row["origin_y"]),
                (row["receipt_x"], row["receipt_y"]),
                label=row["runner"],
                subtitle=f'{row["team"]}  |  {row["competition"]}  |  {int(row["minute"])} min  |  {row["phase"]}')
            st.pyplot(fig, use_container_width=True)
    with right:
        st.markdown("#### The run")

        # THE number for this run: the per-run atom that sums into Run Value / 90.
        # Shown first, and above dV, because dV is the pass and the run jointly and
        # is easily mistaken for the run's own credit.
        rva = row.get("run_value_added", float("nan"))
        st.metric("Run Value — this run", f"{rva:+.3f}" if pd.notna(rva) else "—")
        ctx, full = row.get("rv_context", float("nan")), row.get("run_value", float("nan"))
        if pd.notna(ctx) and pd.notna(full):
            g, h = st.columns(2)
            g.metric("From where he ran", f"{ctx:.3f}")
            g.caption("start and end points only")
            h.metric("Once you see how", f"{full:.3f}")
            h.caption("+ the movement itself")
        st.caption("P(possession progresses) once the model can see **how** he moved, minus the "
                   "same model knowing only **where** he ran between. The difference is this "
                   "run's credit — and it is exactly what sums into **Run Value / 90**.")

        in_f3 = float(row["receipt_x"]) >= config.FINAL_THIRD_X
        st.caption(("✓ Received in the final third, so this run **counts** toward Run Value / 90."
                    if in_f3 else
                    "⌀ Received outside the final third, so this run scores here but contributes "
                    "**nothing** to Run Value / 90 — the headline metric is final-third only."))

        st.metric("Pass + run together (ΔV)", f'{row["delta_V"]:+.3f}')
        st.caption("A different quantity: `V(after) − V(before)` from the state model, crediting "
                   "the pass and the run jointly. Bigger than the run's own credit, and not a "
                   "substitute for it.")

        a, b = st.columns(2)
        a.metric("Distance", f'{row["run_distance"]:.1f} m')
        b.metric("Forward", f'{row["run_fwd"]:+.1f} m')
        a.metric("Separation", f'{row["separation_gained"]:+.1f} m')
        b.metric("Space on receipt", f'{row["space_at_receipt"]:.1f} m')
        st.markdown("#### What happened next")
        c, d = st.columns(2)
        c.metric("Shot in 5s", "Yes" if row.get("shot_5s", 0) else "No")
        d.metric("xG in 5s", f'{row.get("xg_5s", 0):.3f}')
        c.metric("Kept ball 10s", "Yes" if row.get("retain_10s", 0) else "No")
        d.metric("Lost in 5s", "Yes" if row.get("lost_5s", 0) else "No")
        st.markdown("#### Defensive context")
        e, f = st.columns(2)
        e.metric("Encirclement", f'{row.get("encirclement", 0):.2f}')
        e.caption("0 = escape side, 1 = surrounded")
        f.metric("Defenders within 10 m", int(row.get("def_within_10", 0)))
        e.metric("2nd-nearest defender", f'{row.get("second_nearest_def", float("nan")):.1f} m')
        f.metric("Block spread", f'{row.get("block_spread", float("nan")):.1f}')
        st.caption(f'**Game state:** {row.get("game_state", "?")}  ·  '
                   f'**Zone:** {row.get("zone", "?")}  ·  '
                   f'**Defenders goalside:** {int(row["def_goalside_receipt"])}')


def _defensive_context(runs: pd.DataFrame, players: pd.DataFrame):
    """
    Explore the defence the runs were made against.

    Nearest-defender distance alone cannot separate "marked by one man" from
    "surrounded by three". These two measures can:
      encirclement  0 = a clear escape side, towards 1 = defenders on every side
      block spread  how stretched the defending unit is (see the caveat below)
    """
    import plotly.graph_objects as go
    st.markdown("How much does the **shape of the defence** change what a run is worth? "
                "Nearest-defender distance cannot tell one marker apart from three closing "
                "in. **Encirclement** can: 0 means an open side to escape into, towards 1 "
                "means bodies all around.")

    r = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)].copy()

    # ---- outcomes by how enclosed the reception was ----------------------
    st.markdown("#### Does being surrounded change the outcome?")
    zone = st.selectbox("Pitch zone", ["Final third", "Middle third", "Own third", "All"])
    rz = r if zone == "All" else r[r["zone"] == zone]
    if len(rz) > 200:
        rz = rz.copy()
        rz["band"] = pd.cut(rz["encirclement"], [-0.01, 0.2, 0.4, 0.6, 1.01],
                            labels=["Clear side (<0.2)", "0.2–0.4", "0.4–0.6",
                                    "Enclosed (>0.6)"])
        g = rz.groupby("band", observed=True).agg(
            runs=("shot_5s", "size"), shot_in_5s=("shot_5s", "mean"),
            xg_in_5s=("xg_5s", "mean"), kept_ball=("retain_10s", "mean"),
            lost_in_5s=("lost_5s", "mean")).reset_index()
        st.dataframe(g.style.format({"shot_in_5s": "{:.3f}", "xg_in_5s": "{:.4f}",
                                     "kept_ball": "{:.3f}", "lost_in_5s": "{:.3f}",
                                     "runs": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)

    # ---- who receives in traffic -----------------------------------------
    st.markdown("#### Who does it the hard way?")
    st.markdown("Run value against the share of runs received **enclosed**. Top-right is a "
                "player producing value *while* receiving in traffic — a different profile "
                "from someone equally productive in space.")
    pl = players.dropna(subset=["PctSurrounded", "RunValue90"])
    fig = go.Figure(go.Scatter(
        x=pl["PctSurrounded"], y=pl["RunValue90"], mode="markers+text",
        text=[n.split()[-1] for n in pl["runner"]], textposition="top center",
        textfont=dict(size=9, color="#9aa3a0"),
        marker=dict(size=11, color=pl["xGAdded90"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="xG+/90"),
                    line=dict(width=1, color="#12102a")),
        hovertext=pl["runner"], hoverinfo="text"))
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d7d7de"),
                      xaxis=dict(title="share of runs received enclosed", gridcolor="#2a2a2a"),
                      yaxis=dict(title="Run Value / 90", gridcolor="#2a2a2a"),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- the caveat, stated where someone would otherwise misread it ------
    st.markdown("#### A feature that looks important and is not what it seems")
    st.warning("**How spread out the defence is** ranked 2nd of 20 features by importance, "
               "which invites the headline *'runs are worth more against a stretched "
               "defence'*. **The data says the opposite, by 7.6×.** Within the final third, "
               "the most compact quarter of defences concede a shot within 5s on **23.9%** "
               "of runs; the most stretched quarter, **3.1%**.\n\n"
               "The reason is not tactical. A defence is compact precisely *because* the "
               "ball is deep in its own box — even inside the final third, spread still "
               "correlates −0.43 with how advanced the reception is. So the feature is "
               "largely a restatement of 'how close to goal this is'. It is kept because it "
               "predicts; it is **not** presented as a football finding. A high importance "
               "score means a model used a feature, not that the feature means what it "
               "looks like.")

    bs = r.dropna(subset=["block_spread"]).copy()
    ft = bs[bs["receipt_x"] >= 80]
    if len(ft) > 400:
        ft["q"] = pd.qcut(ft["block_spread"], 4,
                          labels=["Most compact", "Q2", "Q3", "Most stretched"])
        gg = ft.groupby("q", observed=True).agg(
            runs=("shot_5s", "size"), shot_in_5s=("shot_5s", "mean"),
            xg_in_5s=("xg_5s", "mean"),
            mean_depth=("receipt_x", "mean")).reset_index()
        st.dataframe(gg.style.format({"shot_in_5s": "{:.3f}", "xg_in_5s": "{:.4f}",
                                      "mean_depth": "{:.1f}", "runs": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Final-third receptions only. Note `mean_depth` rising as the block "
                   "stretches — that column is the giveaway.")


def render(runs: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame,
           report: dict, minutes: pd.DataFrame):
    st.title("Off-ball runs")
    st.markdown("Event data credits whoever touches the ball. **The player who made the "
                "run gets nothing.** Two 360 freeze-frames — one when the pass is struck, "
                "one when it is received — let us reconstruct the movement in between, "
                "and measure whether it created danger and separation.")

    # ---- the honest caveat, stated first, not buried --------------------
    st.info("**What this does and does not see.** We only observe runs that *received "
            "the ball*. A decoy run that dragged a defender away and was never picked "
            "out is invisible here. This measures **runs that got the ball**, not all "
            "off-ball movement — so it under-credits selfless runners.")

    tab0, tab1, tab2, tab5, tab3, tab4 = st.tabs(
        ["🔎 Run explorer", "Leaderboard", "Player runs", "🛡 Defensive context",
         "Situational", "Validation"])

    # ------------------------------------------------------------- explorer
    with tab0:
        _run_explorer(runs)

    # ---------------------------------------------------- defensive context
    with tab5:
        _defensive_context(runs, players)

    # ---------------------------------------------------------------- board
    with tab1:
        metric = st.selectbox(
            "Rank by",
            ["RunValue90", "DeltaV90", "RunThreat90", "xGAdded90", "Shot5s90",
             "NetValue90", "InBehind90", "CounterRuns90", "RetainRate", "SepVsAvg",
             "Runs90", "Encirclement", "PctSurrounded", "SurroundedRuns90",
             "SecondDef"],
            format_func=lambda x: {
                "RunValue90": "Run Value / 90 — the RUN's own contribution (headline)",
                "DeltaV90": "ΔV / 90 — pass-and-run together (joint, not run-only)",
                "RunThreat90": "Run Threat / 90 — xT added (descriptive)",
                "xGAdded90": "xG added / 90 (shots within 5s of the run)",
                "Shot5s90": "Shots within 5s / 90",
                "NetValue90": "Net value / 90 (ΔV minus value lost)",
                "InBehind90": "Runs in behind / 90",
                "CounterRuns90": "Counter-attack runs / 90",
                "RetainRate": "Retention rate (ball kept 10s)",
                "SepVsAvg": "Separation vs an average run (m)",
                "Runs90": "Run volume / 90",
                "Encirclement": "Encirclement — how surrounded when he receives (0-1)",
                "PctSurrounded": "% of runs received while enclosed",
                "SurroundedRuns90": "Runs received while enclosed / 90",
                "SecondDef": "2nd-nearest defender (m) — marked vs swarmed"}[x])
        pos = st.multiselect("Position", ["GK", "DEF", "MID", "FWD"],
                             default=["DEF", "MID", "FWD"])
        topn = st.slider("Show top N", 5, 40, 10)
        v = players[players["pos_group"].isin(pos)].sort_values(metric, ascending=False).head(topn)
        cols = ["runner", "team", "pos_group", "minutes", "runs", "RunValue90",
                "xGAdded90", "InBehind90", "CounterRuns90",
                "Encirclement", "PctSurrounded", "SecondDef", "RetainRate"]
        cols = [c for c in cols if c in v.columns]
        show = v[cols].copy()
        show.columns = ["Player", "Team", "Pos", "Mins", "Runs", "RunVal/90", "xG+/90",
                        "Behind/90", "Counter/90", "Encircled", "% enclosed",
                        "2nd def (m)", "Retain"][:len(cols)]
        num3 = [c for c in ["RunVal/90", "xG+/90"] if c in show.columns]
        num2 = [c for c in ["Behind/90", "Counter/90", "Encircled", "% enclosed",
                            "2nd def (m)", "Retain"] if c in show.columns]
        grad = {"RunValue90": "RunVal/90", "DeltaV90": "RunVal/90",
                "RunThreat90": "RunVal/90", "xGAdded90": "xG+/90",
                "Shot5s90": "RunVal/90", "NetValue90": "RunVal/90",
                "InBehind90": "Behind/90", "CounterRuns90": "Counter/90",
                "RetainRate": "Retain", "SepVsAvg": "RunVal/90", "Runs90": "Runs",
                "Encirclement": "Encircled", "PctSurrounded": "% enclosed",
                "SurroundedRuns90": "Encircled", "SecondDef": "2nd def (m)"}[metric]
        sty = show.style.format({c: "{:.3f}" for c in num3})
        sty = sty.format({c: "{:.2f}" for c in num2})
        sty = sty.format({"Mins": "{:.0f}", "Runs": "{:.0f}"})
        if grad in show.columns:
            sty = sty.background_gradient(cmap="Greens", subset=[grad])
        st.dataframe(sty, use_container_width=True, height=520, hide_index=True)
        st.caption("**RunVal/90** isolates the RUN's own contribution. **Encircled** is how "
                   "surrounded he was when he received: 0 means a clear escape side, towards "
                   "1 means defenders on every side. **2nd def** is the distance to the "
                   "*second* nearest defender — it separates being marked by one man from "
                   "being genuinely swarmed.")

    # ---------------------------------------------------------------- maps
    with tab2:
        who = st.selectbox("Player", players.sort_values("RunThreat90",
                                                         ascending=False)["runner"].tolist())
        rp = runs[(runs["runner"] == who) & (runs["usable"] == 1) & (runs["is_run"] == 1)]
        colour = st.radio("Colour runs by", ["in_behind", "threat", "game_state"],
                          horizontal=True,
                          format_func=lambda x: {"in_behind": "In behind the line",
                                                 "threat": "Threat generated",
                                                 "game_state": "Game state"}[x])
        c1, c2 = st.columns([1.7, 1])
        with c1:
            st.pyplot(run_map(rp, colour, title=who), use_container_width=True)
        with c2:
            p = players[players["runner"] == who].iloc[0]
            st.metric("Runs (usable)", int(p["runs"]))
            st.metric("Run threat / 90", f'{p["RunThreat90"]:.3f}')
            st.metric("Runs in behind / 90", f'{p["InBehind90"]:.2f}')
            st.metric("Mean separation gained", f'{p["SepGained"]:.2f} m')
            st.metric("Mean run distance", f'{p["mean_run_distance"]:.1f} m')

    # ---------------------------------------------------------- situational
    with tab3:
        st.markdown("**When does the movement happen?** A run made chasing a game is a "
                    "different act from one made protecting a lead.")
        by = st.radio("Split by", ["game_state", "phase", "zone"], horizontal=True,
                      format_func=lambda x: {"game_state": "Game state",
                                             "phase": "Phase of play",
                                             "zone": "Pitch zone"}[x])
        from src import run_metrics as rm
        overall = rm.situational(runs, by=by)
        st.dataframe(overall.style.format({
            "run_threat_mean": "{:.4f}", "in_behind_rate": "{:.3f}",
            "sep_gained": "{:.2f}", "mean_distance": "{:.2f}",
            "progress_rate": "{:.3f}", "runs": "{:,.0f}"}),
            use_container_width=True, hide_index=True)

        who2 = st.selectbox("Player detail", players.sort_values(
            "RunThreat90", ascending=False)["runner"].tolist(), key="sit_player")
        pdet = rm.player_situational(runs, who2, minutes, by=by)
        if len(pdet):
            st.dataframe(pdet.style.format({
                "run_threat": "{:.3f}", "threat_per_run": "{:.4f}",
                "sep_gained": "{:.2f}", "progress_rate": "{:.3f}",
                "runs": "{:,.0f}", "in_behind": "{:.0f}"}),
                use_container_width=True, hide_index=True)

    # ----------------------------------------------------------- validation
    with tab4:
        st.markdown("### Do the run features add anything over *where it happened*?")
        st.markdown("Same target — *does the possession produce a shot or a final-third "
                    "entry within 5 actions?* — fit twice on **identical rows and folds**: "
                    "context only (origin, reception point, threat, pass length) versus "
                    "context **plus** the run features (distance, direction, speed, "
                    "separation gained, defenders broken, in-behind).")
        mc, mp = report.get("metrics_context", {}), report.get("metrics_plus_run", {})
        k = st.columns(4)
        k[0].metric("Context only AUC", f'{mc.get("auc", 0):.4f}')
        k[1].metric("+ run features AUC", f'{mp.get("auc", 0):.4f}',
                    f'{report.get("auc_lift", 0):+.4f}')
        k[2].metric("Prediction correlation", f'{report.get("pred_correlation", 0):.3f}')
        k[3].metric("Runs modelled", f'{mc.get("n", 0):,}')

        fi = report.get("feature_importance", {})
        if fi:
            from src.runs import RUN_FEATURES
            items = sorted(fi.items(), key=lambda x: x[1])
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=[v for _, v in items], y=[k2 for k2, _ in items], orientation="h",
                marker=dict(color=[viz.PROG_PASS if k2 in RUN_FEATURES else "#4a4a55"
                                   for k2, _ in items])))
            fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title="LightGBM importance", gridcolor="#2a2a2a"),
                              font=dict(color="#d7d7de"),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.markdown("**Feature importance** — run features in green, context in grey.")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Reconstruction quality")
        q = st.columns(3)
        q[0].metric("Runs reconstructed", f'{len(runs):,}')
        q[1].metric("Unambiguous match", f'{runs["clean_match"].mean():.0%}')
        q[2].metric("Physically plausible", f'{runs["plausible"].mean():.0%}')
        st.caption("A reconstruction is *ambiguous* when the second-nearest team-mate is "
                   "nearly as close as the nearest (we may have picked the wrong player), "
                   "and *implausible* when the implied sprint exceeds "
                   f"{config.MAX_RUN_SPEED} m/s. Only runs passing both are used.")


