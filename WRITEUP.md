# Write-up — Valuing the run

## Part 1 — the possession-value model

### The problem I set out to solve

Event data credits the player who touches the ball. **The player who made the run gets
nothing.** A striker who times a run in behind and drags a centre-back with him appears
in the data only if the pass finds him and he then does something with it. "Does he move
well without the ball?" is one of the first questions any scout asks, and no event feed
answers it.

It is also the question where StatsBomb 360 is genuinely *required* rather than merely
nice to have: you have to see a player who is not on the ball.

### The data trick that makes it possible

Every completed pass is bracketed by **two** freeze-frames:

- **at release** — the passer's frame. The future receiver is in it, but as an
  *anonymous* team-mate dot.
- **at receipt** — the Ball Receipt event carries its **own** frame, and there the
  receiver is flagged `actor`: exact position, by name, plus every defender.

The displacement between the two **is the run he made while the ball travelled**. And
because defenders appear in *both* frames, we can measure whether the run **gained
separation from his marker** — something no event feed records.

### The model (this is the "what did you actually model" answer)

    V(state) = P(the team in possession takes a shot within the next 5 SECONDS
                 | ball position, the player's position, and the defensive
                   structure around both)

`V` is **fitted** — LightGBM, out-of-fold, **match-grouped** (`GroupKFold`; events are
nested in possessions inside matches, so a random split would leak). A run is an action
that moves the team between two states, so:

    run value  =  ΔV  =  V(after) − V(before)

That is exactly the brief's framing: value on in-possession game states, and on the
actions that move between them.

**Why ΔV and not just xT.** xT can only see *where the ball is*. V can see *the
opposition*. Two receptions on the same coordinate have identical xT but very different
V if one has three defenders goalside and the other has none. Proven, not asserted:
**V is fit twice on identical rows and folds**, with and without the freeze-frame
features — **AUC 0.9373 → 0.9444 (+0.0071)**.

I keep **Run Threat/90** (the Σ xT delta) alongside, explicitly labelled as
*descriptive*: it is two grid lookups subtracted, not a model. It earns its place
because it explains in one sentence to a non-technical audience. Leading with it alone
would have been the weak answer.

### Time, not action counts

Every window is in **seconds**. Five tippy-tappy passes take three seconds; five duels
take thirty — actions are a poor clock for judging a run. So: **shot within 5s**,
**retention over 10s**, **possession lost within 5s**, and crucially **value lost**,
weighting a turnover by the threat of *where* it happened. Losing the ball on halfway
and losing it with your full-backs sixty yards upfield are not the same event.

### Reconstruction quality

The receiver's start position is an **inference** — the nearest team-mate dot to the
reception point, bounded by what is reachable in the ball-flight time. Two gates:

- **Ambiguity** — if the second-nearest team-mate is nearly as close, we may have picked
  the wrong player.
- **Plausibility** — a reconstruction implying a sprint faster than 9.5 m/s is a
  *mis-match*, not a run. 2.0% of cases.

**93,202 runs reconstructed across 127 matches; 85% pass both gates** and are used.

### Data: why these four seasons

There is **no full league season with 360 in the open data** — every "league" release is
a single club's season. So I pooled all four that exist: **Barcelona 2020/21, PSG
2021/22, PSG 2022/23, Bayer Leverkusen 2023/24 — 127 matches**, and 26–93 matches per
player (Messi appears in three of them). Depth per player is what a movement metric
needs; a tournament gives at most seven games.

*Trade-off:* four leagues and eras, each a single strong team, so players are compared
across tactical systems and opponent quality. `competition` is on every row so it can be
controlled or filtered.

---

## Part 2 — the player metric

### 2a. The metric

**Run Value / 90** — the fitted value attributable to the **run features specifically**
(the model with them minus the same model without, on identical rows and folds), per 90
minutes, over the pooled four seasons, above a 600-minute floor. **ΔV/90** sits beside
it as the *pass-and-run* value — reported, but not read as the run alone. Alongside it the table carries
**xT/90** (descriptive), **xG added/90** (shots within 5s of the run), **shots in 5s**,
**runs in behind**, **counter-attack runs**, **retention**, and **separation vs an
average run** — so a reader sees the profile, not just a rank.

### 2b. Every decision between raw values and the ranking

1. **Run Value, not ΔV, is the headline — because ΔV is joint.** ΔV = V(after) −
   V(before) spans the pass *and* the run, so it partly credits the passer: it
   correlates +0.136 with how far the *ball* travelled and +0.146 with how far the
   *player* did. Caught when Busquets — who does not make off-ball runs — appeared
   10th on ΔV with a *negative* xT delta. The headline is therefore the isolated
   run contribution; ΔV is reported and labelled honestly.
2. **A fitted V, not the xT delta, is the possession-value layer.** The xT version is
   arithmetic on a location grid; V is a fitted model that has seen the defenders.
   *Shortcut acknowledged:* V targets shots-in-5s, a proxy for scoring probability
   rather than goals themselves — with more time I would model P(goal) and add a
   conceding term.
3. **Seconds, not actions**, for every outcome window (see above).
4. **Value the loss, don't just count it.** `pv_lost_5s` weights a turnover by the
   threat of the position it happened in; `NetValue90` is ΔV minus value surrendered.
5. **Quality gates before aggregation** — ambiguous matches and impossible sprints are
   excluded, not silently averaged in.
6. **Group by player, not player-team.** Players who moved club inside the pool (Messi:
   Barcelona → PSG) initially got one row per club, each holding part of their runs
   while `minutes` was the pooled total — halving their per-90. Caught by asking why
   Messi ranked so low. Fixed.
7. **Separation baselined against the pool.** Raw separation change is negative for
   everyone (defenders converge as the ball arrives), so a raw mean reads as "who loses
   least". `SepVsAvg` re-centres it so 0 = a typical run.

### 2c. What breaks it

- **Selection bias — the headline caveat.** We only see runs that **received the ball**.
  The decoy run that dragged a centre-back across and was never picked out is
  **invisible**. This measures *runs that got the ball*, not all off-ball movement, and
  it under-credits selfless runners. *Fix:* full tracking, where every run is observed
  whether or not it is found.
- **The origin is inferred**, not observed — gated, but not certain.
- **Ball-flight window only.** We see the movement *during the pass*, not the whole run,
  and 360 has **no velocity**, so acceleration and the build-up to the run are invisible.
- **Pooled heterogeneity** — four leagues, four strong teams, different eras.
- **`SepVsAvg` is zone-confounded** — receiving deep keeps separation easily, so it reads
  as a style descriptor within a position, not a cross-position quality ranking.

### 2d. Say it two ways

**~200 words, to a sporting director (no jargon):**

> Every stat we have credits the player who touches the ball. The player who made the
> run that pulled a defender out of position gets nothing. Using camera data that
> records where all 22 players were standing at two moments — when the pass was struck
> and when it arrived — we can now see the run itself: how far he went, whether he beat
> his marker to the space, and whether the team ended up with a shot in the next five
> seconds. Ranked this way, the players who come out on top are the ones you would
> expect from watching, which is the first sign the number is honest. Two things stand
> out for recruitment. First, **attacking full-backs are the hidden engine** — several
> rank among the very best runners in the sample, across three different clubs and
> leagues, and none of that shows up in goals or assists. Second, **runs on the counter
> are worth several times a run in settled possession**, so a player's value depends
> heavily on how his team plays. One caveat I would not hide: we only see runs that
> actually received the ball, so the selfless decoy run is still invisible.

**Technical note (for the data team):**

> ΔV/90 where V = P(shot within 5s | ball, player and defensive structure), LightGBM,
> out-of-fold, match-grouped over 127 matches; a run's value is V(receipt state) −
> V(release state), reconstructed from paired release/receipt freeze-frames with
> ambiguity and sprint-plausibility gates (85% retained). 360 lifts V from 0.9373 to
> 0.9444 AUC. I would trust it more with (a) tracking, to remove the selection bias on
> unreceived runs and to get velocity, (b) a conceding-side term so ΔV is properly
> risk-adjusted, and (c) bootstrapped confidence intervals on the ranking.
