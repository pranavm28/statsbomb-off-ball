# AI Usage Log

**Tool:** Claude Code (Anthropic), used throughout as a coding and analysis pair.

I can explain every line of code and every modelling decision in this repository. This log
is written so a reader can see exactly where the thinking came from.

---

## 1. Split of work

| | Mine | The assistant's |
|---|---|---|
| Problem choice | Off-ball running — the question I wanted to answer | — |
| Modelling frame | **Model it like xG: fit a probability to an outcome we can actually observe** | Recommended **LightGBM** as the estimator for that frame |
| Data choice | Pooling four club seasons, per-player depth over breadth | Checked availability, corrected one wrong premise of mine |
| Metric design | Every definition, every threshold, every restriction | Implemented them |
| Validation design | Ablation on identical rows/folds; report nulls | Wrote the CV code, ran it |
| Visual grammar | My existing published style (comets / dotted) | Implemented against it |
| One real technical find | — | Spotted that the Ball Receipt event carries **its own** freeze-frame |
| Code volume | Reviewed and directed | Wrote most of it |

**Short version:** I decided *what* to build and *how it would be judged*. The assistant
wrote the code and ran the experiments. Three full builds were binned before this one,
which was only affordable because implementation was cheap.

---

## 2. The build, step by step

Each step: **my call → what came back → what I changed.**

### Step 1 — Rejected the first build

- **My call:** asked for a possession-value model on StatsBomb 360.
- **Came back:** an expected-threat spine with a line-breaking-pass metric. Competent.
- **What I changed:** killed it. Two reasons:
  - It used nothing specific to 360 — almost all of it was doable with event data alone.
  - Nothing in it required a decision, so there was nothing to talk about.
- **New instruction:** survey the published 360 research first, then come back with
  concepts native to the data — space, pressure, being outnumbered.
- *This reset is why the project exists in its current form.*

### Step 2 — Chose the architecture from a shortlist

- **My call:** asked for ten candidate concepts rather than one recommendation.
- **What I changed:** picked the structure myself — one engine with lenses hanging off it,
  not three unrelated metrics.
- **My constraint:** it had to be explainable in football language *and* defensible
  technically, because the walkthrough is both at once.

### Step 3 — Set the visual grammar

- **Came back:** pitch maps with arbitrary colours and plain lines.
- **What I changed:** pointed it at my own published style — **passes as comet lines, runs
  and carries as dotted** — and told it to read my existing `player-print` viz module as
  the source of truth instead of inventing a look.

### Step 4 — Rejected the space metric as unmodelled

- **My question:** *what are we actually modelling, and what is this metric adding?*
- **The honest answer:** only the threat layer was fitted. Pitch control was an unvalidated
  geometric heuristic, and the metric predicted no outcome at all.
- **What I changed:** required it be rebuilt as a real model with a downstream target —
  and said up front that a **null result would be reportable**, not something to hide.
- **Result:** null. +0.0026 AUC, and the two models' own predictions correlated with each
  other at 0.97 — adding the freeze-frame barely changed the number for any given
  reception. Lift was *negative* in the final third and in tight space.
- **What I did with it:** published it. It is on the app's Method page.

### Step 5 — Proposed the off-ball running concept

- **My framing:** event data credits whoever touches the ball; **the player who made the
  run gets nothing.** The question scouts ask most, and the one question that genuinely
  needs 360, because you have to see a player who is not on the ball.
- **My reasoning from the null:** if 360 adds nothing to valuing a reception, its value has
  to be where event data is blind.
- **My second call:** a tournament gives 3–7 matches per player — far too thin for a
  movement metric. Asked for league data.
- **Where I was wrong, and it was caught:** StatsBomb's open "Bundesliga 2023/24" is not a
  league season — it is Leverkusen's 34 matches, other clubs appearing only twice each. My
  premise was wrong; my instinct was right for a better reason (34 matches *per player*
  beats seven), so we kept the switch and reframed it honestly.

### Step 6 — Pushed for more data, and took the answer

- **My call:** asked for a full league season.
- **Came back:** there is **no** full league season with 360 in the open data — every
  "league" release is one club's season.
- **My decision:** pool all four that exist — Barcelona 20/21, PSG 21/22, PSG 22/23,
  Leverkusen 23/24. **123 matches; qualified players appear in 11–93 of them.**

### Step 7 — Caught the "not a model" problem a second time

- **My question, again:** *what is the actual possession-value modelling here? Are we just
  modelling xT?*
- **The honest answer:** the headline was `Σ [xT(reception) − xT(origin)]` — two lookups on
  a grid, subtracted. Arithmetic, not modelling. The same weakness I had already rejected
  once.
- **What I required:** an explicit fitted value function, in the xG spirit — a probability
  attached to an observed outcome:

```
V(state) = P(shot within the next 5 SECONDS | ball position, player position,
             defensive structure around both)

run value = ΔV = V(after) − V(before)
```

- **Why this frame:** it is the assessment's own language (value on states, and on the
  actions that move between them), and it makes the 360 contribution *testable* — V can be
  fitted twice, with and without the freeze-frame features.
- **What survived from the old version:** the xT delta, kept as a plain-language companion
  and clearly labelled descriptive, not as the model.

### Step 8 — Specified the outcomes and the clock

- **My list:** shots within 5s, retention over 10s, possessions lost **and possession value
  lost** within 5s, counter-attacking context, xT added, xG added at the end of runs.
- **My call on the clock:** **seconds, not "next N actions"** — five short passes take three
  seconds, five duels take thirty.
- **My call on losses:** value the loss by the threat of *where* it happened. Losing it on
  halfway is not the same event as losing it with your full-backs sixty yards up.

### Step 9 — Set the deliverable format

- **My call:** a live app, not a static page — I need to answer "show me player X" during a
  session, and a running app is itself the answer to "how would this surface in a product?"
- **My spec for the demo view:** one run, both freeze-frames overlaid, pass and run drawn
  separately, outcome alongside.

### Step 10 — Questioned the strength and the novelty

- **My questions:** how strong is this really? Is the run-detection too vague? Is this what
  professionals actually do?
- **What it produced:** the honest framing now in the write-up — the *concept* is mainstream
  (SkillCorner sells off-ball run products), the *method* is unusual because professionals
  use tracking data, so this is the best available approximation from freeze-frames, with
  the missing 20% named.

### Step 11 — Caught a mislabelled baseline

- **My question:** is +0.0071 AUC actually good, and what would we compute *without* 360,
  given the whole method depends on it?
- **What this exposed:** the "baseline" was labelled in the code as *what event data alone
  could produce*, but it contained the runner's starting position — which **is** the
  360-inferred origin. The comparison was never "with and without the camera".
- **What I required:** a layered ablation, since there is no version of this project without
  the camera at all.

| Layer | AUC | Gain |
|---|---|---|
| 1. Event data only | 0.9238 | — |
| 2. + where the runner is | 0.9375 | **+0.0137** |
| 3. + the defenders around him | 0.9444 | +0.0069 |
| 4. + how surrounded he is | 0.9450 | +0.0006 |

- **Total from the camera: +0.0212** — three times what was being reported. Seeing the
  runner is worth about twice what seeing the defence is.

### Step 12 — Proposed compactness, accepted a split result

- **My point:** describing pressure only as "distance to the nearest defender" cannot tell
  one marker apart from three converging. Asked for a compactness / encirclement measure.
- **Result, both halves reported:**
  - In the state model: **null** (+0.0006). Left out rather than carried for appearances.
  - In the run model: **a real gain**, +0.0178 → **+0.0197**. A run is a *change*, and
    compactness gives the model a language for describing what changed.
- **What I changed after:** dropped the three weakest features outright — counting defenders
  within 5 m and 10 m ranked 20th–22nd of 23. What survived was the **geometry** of the
  defence, not the head-count.

### Step 13 — Required a suspicious feature be checked before it became a claim

- **The temptation:** defensive spread ranked 2nd of 20 by importance. Obvious headline:
  *runs are worth more against a stretched defence.*
- **My call:** verify it by phase before it goes anywhere near the write-up.
- **Result: the opposite, by 7.6×.** Compact blocks concede far more shots. Reason is not
  tactical — a defence is compact *because* the ball is deep in its own box (spread
  correlates −0.43 with reception depth even inside the final third).
- **What I did:** kept the feature (it predicts), refused to present it as a football
  finding. Unchecked, the write-up would have stated it backwards.

### Step 14 — Pinned down what the metric actually is, and shipped the caveat

- **My question:** what does "value added" really mean? Is it xT? Is it just P(scoring
  within 5s) dressed up?
- **The answer that came out of it:** neither. It is a **difference between two estimates** —
  how much the *manner* of the movement explains over and above the *geography* of it. An
  information-gain quantity, not a probability. That is why it can go negative and cannot
  be added to goals.
- **What this exposed:** because the metric is the share explained by movement rather than
  position, **changing the target changes the split** — and therefore the ranking. Four
  configurations were tested; three produce orderings with no face validity.
- **My decision:** ship the current configuration and **report the sensitivity as a
  finding** rather than quietly pick the flattering one. Recorded in
  `outputs/sensitivity_report.json` and shown on the app's Method page.
- **My framing for it:** the *method* is defensible; the *ordering* is target-dependent. Use
  it to shortlist profiles, not to separate 11th from 14th.

---

## 3. Errors caught, and how

Listed plainly. None of these threw an error — each produced a number that looked
reasonable and was wrong.

| # | The error | How it surfaced | Fix |
|---|---|---|---|
| 1 | Coordinate convention assumed, not verified | I required an empirical check before any geometry was written | Confirmed from shot locations that attack is normalised toward x=120 |
| 2 | A y-axis flip about to be copied from my own Opta work | I noticed the conventions differ — Opta puts y=0 at the right touchline, StatsBomb at the top | Flip omitted. Copying my own convention blindly would have mirrored every map |
| 3 | Danger-denied metric silently returned zero for everyone | Every player scoring exactly zero is impossible | Parquet returns locations as ndarray, not list — an `isinstance(l, list)` test nulled every row |
| 4 | Stale minutes cache from a partial run | A full build reported nine qualified players | Partial runs no longer write the cache |
| 5 | Baseline and treatment compared on different populations | The reported "lift" was negative and made no sense | Rebuilt on identical rows and folds |
| 6 | Players who changed club counted twice | I asked why Messi ranked so low | Grouping was on player-and-team while minutes were pooled — halved every transferred player's per-90 |
| 7 | ΔV credited to the runner is joint with the passer | Busquets — who does not make off-ball runs — appeared 10th | Diagnosed by correlation (+0.136 ball vs +0.146 player); headline switched to the isolated contribution |
| 8 | Run value rewarded prediction, not merit | Tapsoba — 754 runs, mean reception on halfway, mean forward component −1.1 m — scored near Wirtz | Headline restricted to final-third runs; cut chosen by testing three alternatives |
| 9 | Runs drawn starting outside the pitch | I spotted it on a player map | StatsBomb clamps event locations but **not** freeze-frame positions; raw frames reach y = −7.8 |
| 10 | Accented characters rendering as boxes | I spotted it on the visuals | Display face lacks c-acute and c-caron; fixed with a per-glyph font fallback |
| 11 | The "+360" baseline was mislabelled as event-data-only | I asked what we would compute without 360 at all | Replaced with the four-layer ablation; true contribution is +0.0212, not +0.0071 |
| 12 | An important-looking feature meant the opposite | I required it be checked by phase first | Defensive spread is a depth proxy; the headline would have been backwards |

Numbers 3, 6, 9, 11 and 12 would have shipped unnoticed. They were caught by asking whether
the output made football sense, not by the code failing.

---

## 4. What the assistant recommended that I kept

Being fair about this — some of its suggestions were good and I took them as they were:

- **LightGBM as the estimator.** I set the frame (supervised, fit a probability to an
  observed outcome, xG-style). It proposed gradient-boosted trees and the reasoning held up:
  strong interactions, non-linear thresholds, native missing-value handling, no scaling
  needed, and fast enough that ablation became cheap. I verified the parameters do what the
  documentation says before keeping them.
- **The Ball Receipt freeze-frame pairing.** The genuinely useful unprompted technical find,
  and the thing the whole reconstruction depends on. I checked it against the raw data
  before building on it — 86% of receipts carry their own frame, and the receiver is tagged
  as the actor.
- **GroupKFold, out-of-fold reporting.** Standard and correct for nested data. I verified the
  grouping was on match, not possession.
- **The literature survey.** Pitch control, EPV, xPass 360, support-creation research —
  useful input; I chose what to build on.

## 5. What I rejected

- The entire first build — competent but generic, used nothing specific to 360.
- The heuristic space metric — only the threat layer was fitted; it predicted no outcome.
- Leading with the xT delta — arithmetic on a grid, not a model.
- Copying my own y-flip convention — different coordinate system.
- Action-based windows — replaced with seconds.
- ΔV as the headline — joint with the pass.
- Three of the four compactness features — near-zero contribution.
- "Runs are worth more against a stretched defence" — the data says the opposite.

---

## 6. Representative prompts

Trimmed for length and paired with what each one changed. Almost all are rejections,
challenges or specifications rather than requests for code.

| What I sent | What it changed |
|---|---|
| *"I don't really like the simplistic approach here. Consolidate 10 ideas that leverage 360's uniqueness, is an actual MODELLED metric, and can be applied for teams and players."* | Rejected the first build outright. Forced a fitted model rather than a weighted descriptive index. |
| *"My idea revolves around off-ball running, receiving them, using the possession-value approach to model these runs. I would also recommend using data which is not WC 2022 as the sample is lesser."* | The core concept of the project, and the decision to pool four club seasons instead of a tournament. |
| *"What are we modelling exactly with our SpaceVal? What is the modelling part? What value is it adding, from both a modelling and football POV?"* | Killed the space metric. It was an unfitted heuristic wearing a model's vocabulary. |
| *"Shots within 5s, retention over the next 10 seconds, possession value lost within 5s, runs made in counter-attacking context, xT added from runs, xG added at the end of runs."* | Specified the outcome set the runs are scored against, rather than accepting a single convenient target. |
| *"The plotting looks wrong as a run is from outside, diagnose this. StatsBomb coordinates are already 120x80 if I am not wrong."* | Found that StatsBomb clamps event coordinates but **not** freeze-frames — raw y reached −7.8, so runs were being drawn off the pitch. |
| *"Is an increase of 0.0071 AUC actually good? What would we be computing without 360?"* | The most valuable question I asked. It exposed that the "without 360" baseline already contained a 360 feature. Replaced with the four-tier ablation; the true contribution is **+0.0212**, not +0.0071. |
| *"What exactly does the value added part really measure? Is it like xT? Is it just a glorified way of saying probability of scoring within 5s?"* | Forced the metric to be defined properly as an information gain, and surfaced the target-sensitivity limitation that is now reported rather than buried. |

---

## 7. What I would still verify

- Bootstrapped confidence intervals on the ranking — there are none, and it is the biggest gap.
- A **counterfactual attribution** (hold the ball and defenders fixed, move only the player)
  to replace the residual, which is what makes the ordering target-sensitive.
- A conceding-side term so ΔV is risk-adjusted.
- A logistic-regression baseline — never run, and it would quantify what the tree model's
  flexibility actually buys.
- Sensitivity of every result to the 5-second and 10-second windows, currently chosen by
  judgement rather than tuned.

---

*Data: StatsBomb Open Data, used under the StatsBomb open-data user agreement
(non-commercial / research use; attribution required).*
