"""
Valuing the run — off-ball movement from StatsBomb 360.

One story, one dataset. Every page reads the pooled four-club-season run data.

  Off-ball runs   explorer / leaderboard / player runs / situational / validation
  Player profile  percentile card + that player's run map and action maps
  Teams           who generates run value, and from which phases
  Method          the model, the reconstruction, what breaks it, and what I
                  tried first (the reception null result)

Run:  streamlit run app/streamlit_app.py    (from the project root)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config                      # noqa: E402
from src import viz                # noqa: E402
from src import run_metrics as rm  # noqa: E402
from app import runs_page          # noqa: E402

st.set_page_config(page_title="Valuing the run", page_icon="🏃", layout="wide")

st.markdown("""
<style>
 .block-container{padding-top:2rem;max-width:1200px;}
 .metric-card{background:#1e1e1e;border:1px solid #3a3a3a;border-radius:14px;padding:16px 18px;}
 .pct-row{display:flex;align-items:center;gap:12px;margin:7px 0;}
 .pct-label{width:230px;font-size:13px;color:#d7d7de;}
 .pct-track{flex:1;height:10px;background:#2a2a2a;border-radius:6px;overflow:hidden;}
 .pct-fill{height:100%;border-radius:6px;}
 .pct-val{width:70px;text-align:right;font-variant-numeric:tabular-nums;font-weight:700;font-size:13px;}
 .big-num{font-size:30px;font-weight:800;letter-spacing:-.02em;}
 .eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#9aa3d1;}
 .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;background:#2a2740;color:#877ae8;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load_runs():
    fp = config.DATA_PROC / "runs.parquet"
    return pd.read_parquet(fp) if fp.exists() else None

@st.cache_data(show_spinner=False)
def load_players():
    fp = config.OUTPUTS / "run_player_metrics.csv"
    return pd.read_csv(fp) if fp.exists() else None

@st.cache_data(show_spinner=False)
def load_teams():
    fp = config.OUTPUTS / "run_team_metrics.csv"
    return pd.read_csv(fp) if fp.exists() else None

@st.cache_data(show_spinner=False)
def load_minutes():
    fp = config.DATA_PROC / "minutes.parquet"
    return pd.read_parquet(fp) if fp.exists() else None

@st.cache_data(show_spinner=False)
def load_json(name):
    fp = config.OUTPUTS / name
    return json.load(open(fp)) if fp.exists() else {}

def pct_color(p):
    return (viz.T_ELITE if p >= 80 else viz.T_ABOVE if p >= 60 else
            viz.T_AVG if p >= 40 else viz.T_BELOW if p >= 20 else viz.T_POOR)


def pct_row(label, value, pct, fmt="{:.2f}"):
    p = 0 if pd.isna(pct) else int(pct)
    col = pct_color(p)
    st.markdown(
        f'<div class="pct-row"><div class="pct-label">{label}</div>'
        f'<div class="pct-track"><div class="pct-fill" style="width:{p}%;background:{col};"></div></div>'
        f'<div class="pct-val">{fmt.format(value)}</div>'
        f'<div style="width:40px;text-align:right;color:{col};font-size:12px;font-weight:700;">'
        f'{p}<span style="font-size:9px">pc</span></div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ guard
runs, players = load_runs(), load_players()
if runs is None or players is None:
    st.error("No run outputs found. Build them first:\n\n`python build_runs.py`")
    st.stop()

vreport, rreport = load_json("value_report.json"), load_json("run_report.json")

st.sidebar.markdown("### 🏃 Valuing the **run**")
st.sidebar.caption(config.COMPETITION_NAME + " · StatsBomb open data")
page = st.sidebar.radio("View", ["Off-ball runs", "Player profile", "Teams", "Method"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Runs reconstructed from paired 360 freeze-frames (pass release → ball receipt). "
    "Value = a fitted possession-value model, not an xT lookup. "
    "Data © StatsBomb (attribution required).")


# ============================================================ OFF-BALL RUNS
if page == "Off-ball runs":
    runs_page.render(runs, players, load_teams(), rreport, load_minutes())


# ========================================================== PLAYER PROFILE
elif page == "Player profile":
    st.title("Player profile")
    who = st.selectbox("Player",
                       players.sort_values("RunValue90", ascending=False)["runner"].tolist())
    p = players[players["runner"] == who].iloc[0]
    rp = runs[(runs["runner"] == who) & (runs["usable"] == 1) & (runs["is_run"] == 1)]

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(f"### {who}")
        st.markdown(f'<span class="pill">{p["team"]}</span>&nbsp;'
                    f'<span class="pill">{p.get("position","")}</span>&nbsp;'
                    f'<span style="color:#9aa3d1">{p["minutes"]:.0f} mins · '
                    f'{int(p["runs"])} runs</span>', unsafe_allow_html=True)
        st.markdown("#### Percentile profile "
                    "<span style='color:#7a8087;font-size:12px'>(vs the qualified pool)</span>",
                    unsafe_allow_html=True)
        pct_row("Run Value /90 (final third)", p["RunValue90"], p["RunValue90_pct"], "{:.3f}")
        pct_row("xG added /90", p["xGAdded90"], p["xGAdded90_pct"], "{:.3f}")
        pct_row("Runs in behind /90", p["InBehind90"], p["InBehind90_pct"])
        pct_row("Counter-attack runs /90", p["CounterRuns90"], p["CounterRuns90_pct"])
        pct_row("Run threat (xT) /90", p["RunThreat90"], p["RunThreat90_pct"], "{:.3f}")
        pct_row("Separation vs avg run (m)", p["SepVsAvg"], p["SepVsAvg_pct"])
        pct_row("Retention after run", p["RetainRate"], p["RetainRate_pct"])
        st.caption("Run Value is computed on final-third runs — see Method for why.")
    with right:
        colour = st.radio("Colour runs by", ["in_behind", "threat", "game_state"],
                          horizontal=True,
                          format_func=lambda x: {"in_behind": "In behind",
                                                 "threat": "Threat",
                                                 "game_state": "Game state"}[x])
        st.pyplot(runs_page.run_map(rp, colour, title=who), use_container_width=True)

    st.markdown("#### His runs, by situation")
    by = st.radio("Split by", ["game_state", "phase", "zone"], horizontal=True,
                  format_func=lambda x: {"game_state": "Game state", "phase": "Phase",
                                         "zone": "Zone"}[x], key="prof_split")
    det = rm.player_situational(runs, who, load_minutes(), by=by)
    if len(det):
        st.dataframe(det.style.format({"run_threat": "{:.3f}", "threat_per_run": "{:.4f}",
                                       "sep_gained": "{:.2f}", "progress_rate": "{:.3f}",
                                       "runs": "{:,.0f}", "in_behind": "{:.0f}"}),
                     use_container_width=True, hide_index=True)


# ==================================================================== TEAMS
elif page == "Teams":
    st.title("Teams")
    teams = load_teams()
    st.markdown("Only clubs with a real sample are ranked — in the pooled seasons the four "
                "focus clubs have 26–35 matches while every opponent has 1–4.")
    if teams is not None and len(teams):
        show = teams[["team", "matches", "runs_per_match", "run_threat_per_match",
                      "in_behind_per_match", "progress_rate"]].copy()
        show.columns = ["Team", "Matches", "Runs/match", "Run threat/match",
                        "In behind/match", "Progression rate"]
        st.dataframe(show.style.background_gradient(cmap="Greens",
                                                    subset=["Run threat/match"])
                     .format({"Runs/match": "{:.1f}", "Run threat/match": "{:.3f}",
                              "In behind/match": "{:.2f}", "Progression rate": "{:.3f}",
                              "Matches": "{:.0f}"}),
                     use_container_width=True, hide_index=True)

    st.markdown("### Where runs pay off")
    by = st.radio("Split by", ["phase", "game_state", "zone"], horizontal=True,
                  format_func=lambda x: {"phase": "Phase of play",
                                         "game_state": "Game state", "zone": "Zone"}[x],
                  key="team_split")
    g = rm.situational(runs, by=by)
    st.dataframe(g.style.format({"run_threat_mean": "{:.4f}", "in_behind_rate": "{:.3f}",
                                 "sep_gained": "{:.2f}", "mean_distance": "{:.2f}",
                                 "progress_rate": "{:.3f}", "delta_v": "{:.4f}",
                                 "shot_5s": "{:.3f}", "xg_5s": "{:.4f}",
                                 "retain_10s": "{:.3f}", "lost_5s": "{:.3f}",
                                 "runs": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("The headline split: runs on the **counter** carry far more threat and "
               "progress far more often than runs in settled possession.")


# =================================================================== METHOD
elif page == "Method":
    st.title("Method, validation & what breaks it")

    st.markdown("### The possession-value model")
    st.markdown("""
**V(state) = P(the team takes a shot within the next 5 SECONDS | ball position, the
player's position, and the defensive structure around both).**
Fitted with LightGBM, **out-of-fold**, **match-grouped** (`GroupKFold`) — events are
nested in possessions inside matches, so a random split would leak.

A run is an action that moves the team between two states, so **ΔV = V(after) − V(before)**.
`xT` can only see where the ball is; **V can see the defenders.**
""")
    mb, mp = vreport.get("metrics_base", {}), vreport.get("metrics_plus_360", {})
    k = st.columns(4)
    k[0].metric("V — location only", f'{mb.get("auc", 0):.4f}')
    k[1].metric("V — with 360", f'{mp.get("auc", 0):.4f}',
                f'{vreport.get("auc_lift", 0):+.4f}')
    k[2].metric("Brier (360)", f'{mp.get("brier", 0):.4f}')
    k[3].metric("States modelled", f'{mb.get("n", 0):,}')

    fi = vreport.get("feature_importance", {})
    if fi:
        from src.value import F360_STATE
        import plotly.graph_objects as go
        items = sorted(fi.items(), key=lambda x: x[1])
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k2 for k2, _ in items],
                               orientation="h",
                               marker=dict(color=[viz.PROG_PASS if k2 in F360_STATE
                                                  else "#4a4a55" for k2, _ in items])))
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d7d7de"),
                          xaxis=dict(title="LightGBM importance", gridcolor="#2a2a2a"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("360 features in green — `near_def_player` and `near_def_ball` rank "
                   "2nd and 3rd, so the freeze-frame is doing real work.")

    st.markdown("### What Run Value actually measures")
    st.markdown("""
Two scouts watch the same move. **Scout A** sees only two dots — where the move started
and where the ball ended up. **Scout B** sees those dots *plus the run*: how far the
receiver came, how fast, whether he shook his marker, how many defenders he got behind.

Both are asked *"is this attack going somewhere?"*
**Run Value is how much Scout B's answer differs from Scout A's.**

| | Attached to | What it is |
|---|---|---|
| **xT** | a location | A lookup — "the ball being here is worth X" |
| **V / ΔV** | a situation | A probability — "from here, a shot follows X% of the time" |
| **Run Value** | the **information the run carries** | A **difference between two estimates** — how much knowing *how* he moved changes the forecast |

So it is **not a probability of anything**. It is an information-gain quantity: how much
the *manner* of the movement explains, over and above the *geography* of it. That is why
it can be negative, and why it cannot be added to goals.
""")

    _sf = config.OUTPUTS / "sensitivity_report.json"
    if _sf.exists():
        sens = json.load(open(_sf))
        st.warning(
            "**The ordering is target-sensitive — and that follows from the definition.** "
            "Because the metric measures the share of the outcome explained by *movement* "
            "rather than *position*, the split between the two depends on what you ask the "
            "model to explain. Change the target and the ranking changes. Four "
            "configurations were tested; only the shipped one has face validity, and it was "
            "selected on that basis — which is weak evidence, so it is reported rather than "
            "hidden.\n\n"
            "**Use it to shortlist profiles, not to separate eleventh from fourteenth.**")
        rows = [{"Design": c["design"].replace("  <= SHIPPED", " (shipped)"),
                 "Target": c["target"],
                 "Lift": "—" if c["lift"] is None else f'+{c["lift"]:.4f}',
                 "Top of ranking": c["top"]} for c in sens["configurations_tested"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Same reconstruction, same rows, same paired-ablation discipline in every "
                   "row. Only the target changes. The fix — not yet built — is a "
                   "counterfactual attribution that holds the ball fixed and varies only "
                   "the player, so it cannot proxy for phase of play.")

    st.markdown("### The three metrics, and why they differ")
    st.markdown("""
| Metric | What it is |
|---|---|
| **Run Value / 90** | **Headline.** Fitted value attributable to the **run features** (model with them minus the same model without, identical rows/folds), on **final-third** runs. |
| **ΔV / 90** | `V(after) − V(before)` — the **pass and run together**, credited to the receiver. Correlates +0.136 with how far the *ball* moved and +0.146 with how far the *player* moved, so it is genuinely joint and is **not** read as the run alone. |
| **Run Threat / 90** | Σ `xT(reception) − xT(origin)` — two grid lookups subtracted. Transparent, explains in one sentence, and explicitly **not** a model. |
""")
    st.warning("**Why Run Value is restricted to the final third.** `run_value_added` "
               "measures how much the run features improve the *prediction*, which is not "
               "the same as football merit — a centre-back dropping into space reliably "
               "leads to retained possession, so the model rewards it. Tapsoba (754 runs, "
               "mean reception on halfway, mean forward component −1.1 m, 0.1% in behind) "
               "scored near Wirtz on the unrestricted version. Restricting to the attacking "
               "half moved him only 10th→15th; the **final third** moves him to 34th and "
               "leaves an attacking-movement list on top. That is the cut used.")

    st.markdown("### Reconstruction quality")
    q = st.columns(3)
    q[0].metric("Runs reconstructed", f'{len(runs):,}')
    q[1].metric("Unambiguous match", f'{runs["clean_match"].mean():.0%}')
    q[2].metric("Physically plausible", f'{runs["plausible"].mean():.0%}')
    st.caption("The receiver's start position is *inferred* — the nearest team-mate dot to "
               "the reception point, bounded by ball-flight time. A reconstruction implying "
               f"a sprint faster than {config.MAX_RUN_SPEED} m/s is a mis-match, not a run. "
               "Only runs passing both gates are used.")

    st.markdown("### What breaks it")
    st.markdown("""
- **Selection bias — the headline caveat.** We only see runs that **received the ball**.
  A decoy run that dragged a defender away and was never picked out is invisible.
- **The origin is inferred**, not observed.
- **Ball-flight window only** — and 360 has **no velocity**, so acceleration and the
  build-up to the run are invisible.
- **Pooled heterogeneity** — four leagues, four strong teams, different eras.
- **`SepVsAvg` is zone-confounded** — receiving deep keeps separation easily.
""")

    st.markdown("### What I tried first, and why I moved on")
    rec = load_json("reception_report.json")
    if rec:
        st.markdown("Before the run work I built a **space model** (pitch control × threat "
                    "= EPV) and asked whether 360 space features help predict *whether a "
                    "reception leads somewhere*. They did not.")
        b, pl = rec.get("metrics_base", {}), rec.get("metrics_plus", {})
        c = st.columns(3)
        c[0].metric("Location only", f'{b.get("auc", 0):.4f}')
        c[1].metric("+ 360 space", f'{pl.get("auc", 0):.4f}',
                    f'{rec.get("auc_lift", 0):+.4f}')
        c[2].metric("Prediction correlation", f'{rec.get("pred_correlation", 0):.3f}')
        st.info("A **null result, reported rather than buried.** Where you receive already "
                "encodes how much space you have, and the lift was *negative* in the final "
                "third and in tight space. That is what pointed me at off-ball runs: 360 is "
                "indispensable precisely where you must see a player who is **not on the "
                "ball** — which is the run, not the reception.")
