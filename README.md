# Valuing the run: off-ball movement from StatsBomb 360

**Event data credits whoever touches the ball. The player who made the run gets
nothing.** This project reconstructs off-ball runs from StatsBomb 360 freeze-frames
and values them — so "does he move well without the ball?", the question every scout
asks and no event feed answers, becomes a number you can rank, explain and defend.

> Data: **StatsBomb Open Data**, under the StatsBomb open-data user agreement
> (non-commercial / research; **attribution to StatsBomb required**).

---

## The idea in one picture

Every completed pass is bracketed by **two** freeze-frames:

| | what we see |
|---|---|
| **At release** (passer's frame) | the future receiver is in shot, but as an *anonymous* team-mate dot |
| **At receipt** (the Ball Receipt event's own frame) | the receiver is flagged `actor` — **exact position, by name** — plus every defender |

The displacement between the two **is the run he made while the ball travelled**.
Because defenders appear in *both* frames, we can also measure whether the run
**gained separation from his marker** — something no event feed records.

## TL;DR results

- **93,202 receipts** examined across **123 matches**. **84%** pass the plausibility
  gates (unambiguous nearest teammate, reachable distance); **66% (61,823)** are gated
  *and* involve real movement (≥ 2 m) — those are the runs the metric uses.
- **Model A — state value.** `V(state) = P(shot within the next 5 SECONDS | ball,
  player, and the defensive structure around both)` — fitted out-of-fold, match-grouped.
  The 360 contribution is measured by a **four-tier ablation** on identical rows and
  folds: event-only **AUC 0.9238** → **+ defensive structure 0.9444**, so **+0.0212**.
  (A naive "with vs without 360" comparison gives +0.0069 and is wrong — the runner's
  own position is itself a 360 feature. See `AI_USAGE.md`.)
- **Model B — run value.** A *different* target: **progression**, i.e. a shot *or* a
  final-third entry within the next 5 actions. Identical rows and folds, context-only
  **AUC 0.7806** → **+ run features 0.8002** (**+0.0197**). Run Value per 90 is the
  per-run decomposition of that lift.
- **Attacking full-backs are the hidden engine.** 5 of the top 15 by Run Threat/90
  are `DEF`: Frimpong, Tella, Dest, Jordi Alba, Nuno Mendes — across **three
  different clubs and leagues**, so it is not one team's quirk.
- **Counter-attacks are where runs pay.** In-behind rate **17.2%** on the counter vs
  **2.7%** in open play; possessions progress **64.8%** vs **33.7%**.
- **Messi ranks 24th of 59 — and that is the metric working.** See below.

## Data: why these four seasons

There is **no full league season with 360 in the open data**. Every "league" release
is a single club's season. So we pool all four that exist:

| Season | Club | Matches |
|---|---|---|
| La Liga 2020/21 | Barcelona | 35 |
| Ligue 1 2021/22 | PSG | 26 |
| Ligue 1 2022/23 | PSG | 31 |
| Bundesliga 2023/24 | Bayer Leverkusen | 31 |

**123 matches — qualified players appear in 11–93 of them** (Messi 93, across three).
Depth per player is exactly what a movement metric needs; a tournament gives at most
seven games. **Trade-off, stated honestly:** four different leagues and eras, each a
single strong team, so players are compared across tactical systems and opponent
quality. `competition` is carried on every row so it can be controlled or filtered.

---

## Definitions

| Metric | Definition |
|---|---|
| **run** | receiver's displacement between the release frame and the receipt frame (≥ 2 m) |
| **Run Value / 90** | **THE HEADLINE.** Σ of the fitted value attributable to the **run features specifically** (model with them minus the same model without, identical rows and folds). Isolates the run from the pass that found it. |
| **ΔV / 90** | `V(after) − V(before)` across the two freeze-frames — the value of the **pass and run together**, credited to the receiver. Diagnostic: across all runs ΔV correlates +0.134 with how far the *ball* travelled and +0.147 with how far the *player* ran; **inside the final third the ball term overtakes it** (+0.241 vs +0.217), which is why ΔV was rejected as the headline — it partly ranks passers. |
| **Run Threat / 90** | Σ `xT(reception) − xT(origin)` — descriptive, two grid lookups subtracted. Kept because it explains in one sentence; explicitly **not** a model. |
| **xG added / 90** | xG of shots taken within 5s of the run |
| **Net value / 90** | ΔV minus possession value surrendered when the move breaks down |
| **In-behind / 90** | runs received with ≤1 defender goalside |
| **Sep vs Avg** | mean separation change **relative to the pool average run** (raw is negative for everyone: defenders converge as the ball arrives) |

**Model A.** `V(state) = P(shot within the next 5 SECONDS | ball position, player
position, defensive structure)`, LightGBM, **out-of-fold**, **match-grouped**
(`GroupKFold`) — events are nested in possessions in matches, so a random split would
leak.

**Two clocks, and they differ — worth knowing.** The descriptive outcomes recorded per
run are in **seconds**: shot in 5s, xG in 5s, retention over 10s, loss within 5s, and
the **threat surrendered** at the turnover (losing it on halfway ≠ losing it with your
full-backs upfield). Seconds are right there because five short passes take three
seconds and five duels take thirty, so action counts are a poor clock for elapsed
danger. **Model B's `progression` label is the exception**: it looks at the next **5
actions** of the possession, because progression is a property of the possession's
sequence of touches rather than of elapsed time.

## Reconstruction quality (the honest part)

The receiver's start position is an **inference**: the nearest team-mate dot to the
reception point, bounded by what is reachable in the ball-flight time. Two gates:

- **Ambiguity** — if the second-nearest team-mate is nearly as close, we may have
  picked the wrong player. Flagged.
- **Plausibility** — a reconstruction implying a sprint faster than 9.5 m/s is a
  *mis-match*, not a run. 2.0% of cases; excluded.

**84%** of receipts pass both gates. The runs the metric uses are those that also
involve real movement (≥ 2 m): **66% (61,823)** of all receipts.

## What it does NOT see — selection bias

We only observe runs that **received the ball**. The decoy run that dragged a
centre-back across and was never picked out is **invisible**. This measures *runs
that got the ball*, not all off-ball movement, and it under-credits selfless runners.
Said first, not in a footnote.

## Messi ranks 24th — why that is correct

Not a bug (though there *was* one — see `AI_USAGE.md`). His profile: mean forward
component **+0.04 m** (essentially none) versus a pool average of +0.21 and Mbappé's
+1.42; in-behind rate **1.8%** vs Mbappé's 8.7%. Late-career Messi receives to feet,
static or dropping, and creates with the ball. **The metric measures one thing —
off-ball running — and it is not a player rating.** A metric that ranked Messi first
at everything would be a metric that measures nothing.

---

## Reproduce

**Just run the app.** The precomputed outputs are committed, so no rebuild and no
StatsBomb download is needed:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Rebuild the dataset and refit the models** — needs Python 3.11 or 3.12, because
LightGBM has no wheel for 3.13+ yet:

```bash
pip install -r requirements-pipeline.txt
python build_runs.py          # downloads ~240 MB of 360 once, then cached
python make_app_frames.py     # slim freeze-frame extract the run explorer reads
```

Outputs → `outputs/run_player_metrics.csv`, `run_team_metrics.csv`,
`run_report.json`; intermediates → `data/processed/runs.parquet`.

The two requirement files are split on purpose. The app only *reads* the parquet
files, so it needs no ML stack; `requirements.txt` is therefore small and uses
version floors so it installs on whatever Python a host offers.
`requirements-pipeline.txt` pins exact versions instead, so re-running the
pipeline reproduces the reported numbers rather than merely approximating them.

## Code layout

```
config.py              the four competitions + every parameter
build_runs.py          pooled build: load -> threat -> reconstruct runs -> model -> metrics
src/
  data.py              StatsBomb loaders, per-competition parquet cache, minutes
  pitch.py             geometry + freeze-frame parsing (numpy, no shapely)
  threat.py            xT threat surface (value iteration)
  runs.py              RUN RECONSTRUCTION + quality gates + the run-value model
  run_metrics.py       player / team aggregation + situational splits
  features.py          action table (threat + 360 context) used for outcome labels
  space.py             pitch control -> EPV surface (supporting analysis)
  decision.py          xPass + passing-option valuation (supporting analysis)
  viz.py               house pitch style: comets for passes, DOTTED for runs/carries
app/
  streamlit_app.py     the app shell
  runs_page.py         leaderboard, run maps, situational splits, validation
walkthrough/           prep notes: concepts to research + code tour & narrative
WRITEUP.md             Part 2: the metric, the decisions, what breaks it, both audiences
AI_USAGE.md            tools, prompts, and where I overrode / corrected the assistant
```

## Limitations

- **Selection bias** (above) — the headline caveat.
- **Inferred origin** — the run's start is matched, not observed; gated but not certain.
- **Ball-flight window only** — we see the movement during the pass, not the whole run,
  and 360 has **no velocity**, so acceleration and the run *before* release are invisible.
- **Pooled heterogeneity** — four leagues/eras, four strong teams.
- **ΔV is joint** — it credits the receiver for a pass-and-run event; use Run Value
  for run-only attribution.
- **`Sep vs Avg` is zone-confounded** — receiving deep keeps separation easily, so it
  reads as a style descriptor within a position, not a quality ranking across positions.
