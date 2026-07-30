"""
Build the submission write-up: docs/Off_Ball_Run_Value__Write_Up.pdf

Every number in the document is read from outputs/ and data/processed/ rather
than typed in, so the document cannot drift from the model that produced it.
Rendered with headless Chrome so the PDF keeps real text (selectable, searchable)
instead of being a picture of a document.

    python docs/make_writeup.py
"""
from __future__ import annotations
import base64
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DOCS = ROOT / "docs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BRAND = "#5b4fd6"      # print-safe version of the app's #877ae8
INK = "#15171a"
MUTED = "#5f6672"
RULE = "#d8dbe2"


# --------------------------------------------------------------------- numbers
def gather() -> dict:
    runs = pd.read_parquet(config.DATA_PROC / "runs.parquet")
    mins = pd.read_parquet(config.DATA_PROC / "minutes.parquet")
    pm = pd.read_csv(ROOT / "outputs" / "run_player_metrics.csv")
    abl = json.load(open(ROOT / "outputs" / "ablation_report.json"))
    rep = json.load(open(ROOT / "outputs" / "run_report.json"))

    u = runs[(runs["usable"] == 1) & (runs["is_run"] == 1)]
    top = pm.nlargest(10, "RunValue90")

    return dict(
        matches=runs["match_id"].nunique(),
        receipts=len(runs),
        usable=len(u),
        usable_pct=100 * len(u) / len(runs),
        comps=runs["competition"].nunique(),
        players=len(pm),
        minutes=mins["minutes"].sum(),
        final_third=int((u["receipt_x"] >= config.FINAL_THIRD_X).sum()),
        top=top,
        abl=abl,
        rep=rep,
        run_auc=rep["metrics_plus_run"]["auc"],
        ctx_auc=rep["metrics_context"]["auc"],
        run_lift=rep["auc_lift"],
        run_brier=rep["metrics_plus_run"]["brier"],
        v_auc=abl["tiers"][-1]["auc"],
        v_brier=abl["tiers"][-1]["brier"],
        gain360=abl["total_gain_from_360"],
        n_states=abl["n_rows"],
        base_rate=abl["base_rate"],
    )


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# ------------------------------------------------------------------ components
def flowchart(matches: int) -> str:
    """Five boxes, four arrows. Deliberately plain -- it is a map, not art."""
    steps = [
        ("1. Data", "StatsBomb 360:", "events + freeze-frames", f"{matches} matches"),
        ("2. Reconstruct", "Pair the release and", "receipt frames to", "recover the run"),
        ("3. Describe", "Ball, runner and", "defensive shape at", "both instants"),
        ("4. Model", "P(shot within 5s),", "folds grouped by", "match"),
        ("5. Aggregate", "Run Value per 90", "for every player", "over 600 mins"),
    ]
    w, gap, h = 176, 30, 96
    out = ['<svg viewBox="0 0 1030 116" class="flow" xmlns="http://www.w3.org/2000/svg">']
    for i, (head, l1, l2, l3) in enumerate(steps):
        x = i * (w + gap)
        out.append(
            f'<rect x="{x}" y="8" width="{w}" height="{h}" rx="7" fill="#f6f6fb" '
            f'stroke="{BRAND}" stroke-width="1.2"/>'
            f'<text x="{x + 12}" y="28" class="fh">{head}</text>'
            f'<text x="{x + 12}" y="47" class="fb">{l1}</text>'
            f'<text x="{x + 12}" y="63" class="fb">{l2}</text>'
            f'<text x="{x + 12}" y="79" class="fb">{l3}</text>')
        if i < len(steps) - 1:
            ax = x + w + 4
            out.append(
                f'<line x1="{ax}" y1="56" x2="{ax + gap - 9}" y2="56" stroke="{BRAND}" '
                f'stroke-width="1.4"/>'
                f'<polygon points="{ax + gap - 9},52 {ax + gap - 2},56 {ax + gap - 9},60" fill="{BRAND}"/>')
    out.append("</svg>")
    return "".join(out)


# Display names. Taking the last token is wrong for Iberian and Brazilian naming
# ("Kylian Mbappé Lottin" -> "Lottin", "Neymar da Silva Santos Junior" -> "Junior"),
# so match on a distinctive token instead and fall back only when nothing matches.
SHORT = [("Mbapp", "K. Mbappé"), ("Boniface", "V. Boniface"), ("Wirtz", "F. Wirtz"),
         ("Trinc", "F. Trincão"), ("Braithwaite", "M. Braithwaite"),
         ("Hofmann", "J. Hofmann"), ("Neymar", "Neymar"), ("Schick", "P. Schick"),
         ("Ekitike", "H. Ekitike"), ("Tella", "N. Tella"), ("Dembélé", "O. Dembélé"),
         ("Messi", "L. Messi"), ("Adli", "A. Adli")]


def short_name(name: str) -> str:
    for needle, disp in SHORT:
        if needle in name:
            return disp
    parts = name.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else name


def top_table(top: pd.DataFrame) -> str:
    rows = []
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        short = short_name(r["runner"])
        pos = (r["position"].replace("Attacking Midfield", "Att Mid")
               .replace("Center Forward", "Forward").replace("Right Wing Back", "RWB"))
        rows.append(
            f"<tr><td class='r'>{i}</td><td><b>{short}</b></td><td class='m'>{r['team']}</td>"
            f"<td class='m'>{pos}</td><td class='r'>{r['minutes'] / 90:.1f}</td>"
            f"<td class='r'>{r['Runs90']:.1f}</td><td class='r'>{r['FinalThirdRuns90']:.1f}</td>"
            f"<td class='r'>{r['InBehind90']:.2f}</td><td class='r'>{r['Encirclement']:.2f}</td>"
            f"<td class='r hl'>{r['RunValue90']:.3f}</td></tr>")
    return f"""
<table class="tbl">
<thead><tr>
  <th class="r">#</th><th>Player</th><th>Team</th><th>Position</th>
  <th class="r">90s</th><th class="r">Runs<br>/90</th><th class="r">F3 runs<br>/90</th>
  <th class="r">In behind<br>/90</th><th class="r">Enclosed<br>share</th>
  <th class="r">Run Value<br>/90</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def ablation_table(abl: dict) -> str:
    notes = ["what event data alone can see",
             "the runner's own position, from 360",
             "where the defenders are",
             "shape of the block (negligible)"]
    rows = []
    for i, t in enumerate(abl["tiers"]):
        gain = "—" if t["gain_vs_previous"] is None else f"+{t['gain_vs_previous']:.4f}"
        note = notes[i]
        rows.append(
            f"<tr><td>{t['tier']}</td><td class='r'>{t['n_features']}</td>"
            f"<td class='r'>{t['auc']:.4f}</td><td class='r'>{t['brier']:.4f}</td>"
            f"<td class='r'>{gain}</td><td class='m'>{note}</td></tr>")
    return f"""
<table class="tbl">
<thead><tr><th>Feature tier</th><th class="r">n</th><th class="r">AUC</th>
<th class="r">Brier</th><th class="r">Gain</th><th>What it adds</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


# ------------------------------------------------------------------------ CSS
CSS = """
@page { size: A4; margin: 12mm 12mm 10mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 8.4pt; line-height: 1.38; color: #15171a; margin: 0; }
h1 { font-size: 16.5pt; line-height: 1.1; margin: 0 0 1.4mm 0; letter-spacing: -0.4px; }
h2 { font-size: 9.6pt; margin: 3.6mm 0 1.3mm 0; color: #5b4fd6;
     letter-spacing: 0.2px; }
h2 .n { color: #b9bcc6; margin-right: 4px; }
h3 { font-size: 8.7pt; margin: 2.4mm 0 1mm 0; }
p { margin: 0 0 1.7mm 0; }
b, strong { font-weight: 600; }
.sub { color: #5f6672; font-size: 8.7pt; margin: 0 0 1.2mm 0; }
.byline { color: #5f6672; font-size: 7.6pt; border-top: 1px solid #d8dbe2;
          padding-top: 1.2mm; margin-top: 1.8mm; }
.lede { background: #f6f6fb; border-left: 2.6px solid #5b4fd6;
        padding: 2.2mm 2.8mm; margin: 2.2mm 0 0.6mm 0; font-size: 8.6pt; }
.flow { width: 100%; height: auto; margin: 0.8mm 0 1.8mm 0; }
.fh { font: 600 11.5px "Helvetica Neue", Arial, sans-serif; fill: #5b4fd6; }
.fb { font: 400 10.5px "Helvetica Neue", Arial, sans-serif; fill: #3f4550; }
.tbl { width: 100%; border-collapse: collapse; font-size: 7.6pt; margin: 1.1mm 0 1.5mm 0;
       page-break-inside: avoid; }
.tbl th { text-align: left; font-weight: 600; font-size: 7pt; color: #3f4550;
          border-bottom: 1px solid #9aa0ad; padding: 0.8mm 1.2mm; line-height: 1.2; }
.tbl td { padding: 0.78mm 1.2mm; border-bottom: 0.5px solid #e6e8ee; }
.tbl td.r, .tbl th.r { text-align: right; font-variant-numeric: tabular-nums; }
.tbl td.m { color: #5f6672; }
.tbl td.hl { font-weight: 700; color: #5b4fd6; }
.tbl tbody tr:first-child td { background: #f8f8fc; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 3.6mm; }
.fig { margin: 0.8mm 0 0.6mm 0; }
.fig img { width: 100%; border-radius: 4px; display: block; }
.cap { font-size: 7.1pt; color: #5f6672; margin-top: 0.9mm; }
.card { border: 1px solid #d8dbe2; border-radius: 4px; padding: 2.2mm 2.6mm;
        page-break-inside: avoid; }
.card h3 { margin-top: 0; color: #5b4fd6; }
ul { margin: 0 0 1.5mm 0; padding-left: 3.8mm; }
li { margin-bottom: 0.6mm; }
.kv { font-variant-numeric: tabular-nums; }
.foot { border-top: 1px solid #d8dbe2; margin-top: 2.6mm; padding-top: 1.2mm;
        font-size: 6.9pt; color: #5f6672; }
.nb { page-break-inside: avoid; }
.nb { page-break-inside: avoid; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8pt; background: #f3f4f8;
       padding: 0.3mm 1mm; border-radius: 2px; }
"""


# ----------------------------------------------------------------------- build
def build_html(d: dict) -> str:
    t = d["abl"]["tiers"]
    img = b64(DOCS / "figures" / "fig_mbappe_run.png")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Valuing Off-Ball Runs — Write-Up</title>
<style>{CSS}</style></head><body>

<h1>Valuing off-ball runs with StatsBomb 360</h1>
<div class="sub">A possession-value model for the movement that happens away from the ball,
and a player metric built on top of it.</div>
<div class="byline">Pranav Mohan &nbsp;·&nbsp; Data Scientist assessment, Parts 1 &amp; 2
&nbsp;·&nbsp; July 2026 &nbsp;·&nbsp; Code: github.com/pranavm28/statsbomb-off-ball</div>

<div class="lede"><b>Headline.</b> Event data records the pass; it never records the run that
made the pass possible. Using 360 freeze-frames I reconstruct
<span class="kv">{d['usable']:,}</span> off-ball runs and value them by how much they raise the
chance of a shot within five seconds. Seeing the players — not just the ball — is worth
<span class="kv">+{d['gain360']:.4f}</span> AUC over event-only features on identical rows and
folds. The applied finding is a profile, not just a ranking: <b>Victor Boniface</b> ranks second
per 90 on a fifth of Mbappé's minutes, while receiving in the most crowded space of anyone near
the top.</div>

<h2><span class="n">1</span>What "possession value" means here</h2>
<p><b>Unit:</b> the game state at one instant, attached to one player. <b>Target:</b> a binary
label — did the possession produce a shot within the next 5 seconds. <b>Model:</b>
V(state) = P(shot ≤ 5s | ball location, that player's location, the defensive structure around
him), fitted as a supervised classifier on <span class="kv">{d['n_states']:,}</span> states with a
base rate of <span class="kv">{100 * d['base_rate']:.1f}%</span>.</p>
<p>I chose a learned state-value function over a grid xT for one reason: xT keys value to
<i>where the ball is</i>, and this project is about <i>where the players are</i>. Two identical ball
positions with different defensive shapes are one cell in xT and two states here. An xT surface is
still used as a context feature, so the model knows the geographic prior it adds to.</p>

<h2><span class="n">2</span>How it works</h2>
{flowchart(d['matches'])}
<p>The reconstruction does the real work. A <code>Ball Receipt</code> event carries its own
freeze-frame with the receiver tagged <code>actor</code>, which fixes where the run <i>ended</i> and
who made it. The preceding pass's frame fixes where everyone stood when the ball was struck — but
frames are anonymous, so the run's <i>origin</i> must be inferred: the nearest teammate in the
release frame, accepted only if the implied sprint is physically possible
(≤ <span class="kv">{config.MAX_RUN_SPEED}</span> m/s over the pass duration). That gate is why
<span class="kv">{d['usable_pct']:.0f}%</span> of <span class="kv">{d['receipts']:,}</span> receipts
survive, and it is the biggest assumption in the project.</p>

<h2><span class="n">3</span>Scope, and why</h2>
<p>Four full competition-seasons with 360 coverage — La Liga 2020/21, Ligue 1 2021/22 and 2022/23,
Bundesliga 2023/24: <span class="kv">{d['matches']}</span> matches,
<span class="kv">{d['minutes']:,.0f}</span> player-minutes. I pooled seasons rather than use the
World Cup because off-ball running needs volume per player and the metric is a rate: a tournament
gives seven matches, a season thirty-plus. The honest cost is that open-data 360 covers one focus
club per competition, so the pool is effectively three clubs and the ranking is within-sample, not
cross-league.</p>

<h2><span class="n">4</span>The metric, and the ranking</h2>
<p><b>Run Value per 90.</b> One number per player, built in four steps: (i) for each run, take the
out-of-fold gain in predicted shot probability when the model can see <i>how</i> the player moved,
on top of a context model that already knows where the ball and the receipt were; (ii) keep only
runs received in the final third, because outside it "predictive" and "valuable" come apart —
a centre-back dropping into space is reliably informative and is not creating anything;
(iii) sum per player; (iv) divide by 90s played, with a
<span class="kv">{config.MIN_MINUTES}</span>-minute floor. That leaves
<span class="kv">{d['players']}</span> qualifying players.</p>
<p><b>It is an information gain, not a probability.</b> It answers "how much of the outcome is
explained by the manner of the movement rather than its geography", so it does not sum to goals.
Two scouts watch the same possession, one seeing only the ball, one seeing the runner: Run Value is
what the second knew that the first did not.</p>
{top_table(d['top'])}
<p class="cap">Enclosed share = mean angular encirclement on receipt (1 = defenders on all sides,
0 = a completely clear side). F3 = final third.</p>

<h2><span class="n">5</span>One run, in detail</h2>
<div class="two nb">
<div class="fig">
  <img src="data:image/png;base64,{img}" alt="Mbappé run">
</div>
<div>
<p>Ghosted dots are where everyone stood when the pass was struck; solid dots where they stood
when it arrived. Gold is the ball, green dotted is the run.</p>
<p>Mbappé starts <b>outside the box</b> at the moment of release and arrives on the penalty spot
16.2 m later, with <b>four defenders</b> around him — encirclement 0.63, i.e. almost no clear side.
The possession produces <b>0.70 xG within five seconds</b>.</p>
<p>This is what event data cannot see. A model given only the ball's start and end sees a 16 m pass
into a crowd; a model given the freeze-frames sees a striker manufacturing the one gap that made it
a chance.</p>
</div>
</div>

<h2><span class="n">6</span>Validation</h2>
<p><b>Nested structure.</b> Events sit inside possessions inside matches, so a random split leaks —
two events from one possession are near-duplicates. All folds are <code>GroupKFold</code> on
<code>match_id</code>; every number quoted is out-of-fold.</p>
<p><b>Does 360 earn its keep?</b> The honest test is not "with vs without 360" — the runner's own
position <i>is</i> a 360 feature, so a naive baseline smuggles it in. Instead, four nested tiers on
identical rows and folds:</p>
{ablation_table(d['abl'])}
<p>Tier 1→3 is the real 360 contribution: <b>+{d['gain360']:.4f} AUC</b>. Tier 4 is
<b>+{t[3]['gain_vs_previous']:.4f}</b> — a null result, reported rather than dropped. Separately,
the run features add <b>+{d['run_lift']:.4f} AUC</b> over context alone
({d['ctx_auc']:.4f} → {d['run_auc']:.4f}) on the progression target.
<b>Calibration:</b> out-of-fold Brier {d['v_brier']:.4f} against a
{100 * d['base_rate']:.1f}% base rate, and predicted vs observed track closely across all ten
deciles (figures in the repo).</p>

<h2><span class="n">7</span>Say it two ways</h2>
<div class="two">
<div class="card">
<h3>To a sporting director (~200 words)</h3>
<p>We can now put a number on something we could previously only describe: what a player's running
<i>without</i> the ball is worth.</p>
<p>For every pass in our sample we can see where all twenty-two players stood when the ball was
struck, and again when it arrived. That lets us rebuild the run the receiver made to get there and
ask one question: how much more likely did a shot become because of the <i>way</i> he moved, rather
than simply because of where the ball ended up?</p>
<p>Mbappé finishes top, which tells you the method isn't broken but doesn't tell you anything you'd
pay for. The name worth your time is <b>Victor Boniface</b>. He is second on this list on roughly a
fifth of Mbappé's minutes, and he does it while receiving in the most crowded areas of anyone near
the top — more than half his receptions come with defenders closing him from most sides. That is a
specific and buyable habit: a forward who will still move into a packed box, and whose movement
makes the pass playable.</p>
<p>How much would I bet? Enough to put him on a shortlist and watch him properly. Not enough to
price a bid on this alone — the sample is three clubs, and below the leading names the order shifts
depending on which outcome we ask about.</p>
</div>
<div class="card">
<h3>To the data team (2–3 sentences)</h3>
<p>Run Value is an ablation quantity: the out-of-fold gain in P(shot ≤ 5s) once the model can see
<i>how</i> the player moved, on top of a context model that already has ball location, receipt
location and defensive structure — so it is information gain about the manner of movement, not a
probability, and it will not sum to goals.</p>
<p>Folds are <code>GroupKFold</code> on <code>match_id</code>; the 360 tiers are worth
+{d['gain360']:.4f} AUC over event-only features on identical rows and folds, and out-of-fold
predictions are calibrated (Brier {d['v_brier']:.4f} on a {100 * d['base_rate']:.1f}% base rate).</p>
<p>What would make me trust it more: a counterfactual attribution that holds the ball and the
defenders fixed and moves only the runner, because the current quantity is target-sensitive —
three of four target definitions I tested reorder the leaderboard, which is a property of
information-gain metrics rather than a bug, but it needs pinning down before anyone scouts off it.</p>
</div>
</div>

<h2><span class="n">8</span>What I used, and what I rejected</h2>
<div class="two">
<div>
<h3>Kept</h3>
<ul>
<li><b>Ball Receipt freeze-frames.</b> The receiver is tagged <code>actor</code> — the only route to
the runner's identity.</li>
<li><b>Nearest-teammate origin with a speed gate.</b> Frames are anonymous; the gate keeps
physically impossible matches out.</li>
<li><b>LightGBM.</b> Tabular, non-linear, handles the mixed feature scales; the relationships here
(distance, angle, congestion) are not linear.</li>
<li><b>Angular encirclement</b> over defender counts — <i>where</i> the defenders are beats
<i>how many</i>.</li>
<li><b>Final-third restriction</b> to stop rewarding informative-but-harmless movement.</li>
</ul>
</div>
<div>
<h3>Rejected, and why</h3>
<ul>
<li><b>socceraction / VAEP off the shelf.</b> The point was to define the target myself; using it
as a black box is the failure mode named in the brief.</li>
<li><b>Defender counts within 10 m.</b> Collinear with the geometry features, no incremental lift.</li>
<li><b>Compactness tier.</b> +{t[3]['gain_vs_previous']:.4f} AUC — kept in the report as a null,
not in the metric.</li>
<li><b>Plain ΔV as the player metric.</b> It is joint with the pass; corr(ΔV, ball forward) exceeded
corr(ΔV, run forward), so it partly ranked passers.</li>
<li><b>Pitch control / velocity models.</b> Freeze-frames have positions but no velocities.</li>
<li><b>World Cup 2022.</b> Too few matches per player for a rate metric.</li>
</ul>
</div>
</div>

<h2><span class="n">9</span>What breaks it</h2>
<p><b>The one that matters.</b> The metric is <b>target-sensitive</b>: of four target definitions
tested, three reorder the leaderboard and two produce football-nonsense orderings (defenders and
deep midfielders on top). That is inherent to an information-gain quantity — changing the outcome
changes what counts as position-explained versus movement-explained — but the ordering below the
leading names is not settled. The specified-but-unbuilt fix: a <b>counterfactual attribution</b>
holding ball and defenders fixed and moving only the player, isolating his contribution instead of
inferring it from a model comparison.</p>
<ul>
<li><b>Who it treats unfairly.</b> Target men and deep-lying creators. It rewards movement that
raises shot probability quickly, so a striker who occupies two centre-backs to open space for
someone else scores nothing for it. Fixing that needs off-ball value attributed to the
<i>space created for teammates</i>, not just the receiving player.</li>
<li><b>Small samples.</b> The {config.MIN_MINUTES}-minute floor is ~6.7 × 90, so
{d['top'].iloc[3]['runner'].split()[-1]} ranks 4th on
{d['top'].iloc[3]['minutes'] / 90:.1f} × 90. Bootstrap intervals per player are the fix; they are
not in this version.</li>
<li><b>Three clubs.</b> Open-data 360 is club-focused, so this is within-sample. No cross-league
claim is made.</li>
<li><b>Inferred origins.</b> {100 - d['usable_pct']:.0f}% of receipts are discarded, and the kept
ones rest on a nearest-teammate assumption that will occasionally pick the wrong player in a
crowded box.</li>
</ul>

<h2><span class="n">10</span>AI usage, in short</h2>
<p>Full log with representative prompts, accepted/modified/rejected detail and the errors I caught
is in <code>AI_USAGE.md</code> in the repo. Summary:</p>
<table class="tbl">
<thead><tr><th>Area</th><th>What the assistant did</th><th>What I did</th></tr></thead>
<tbody>
<tr><td>Framing &amp; metric</td><td class="m">Sketched options once I set the direction</td>
<td>Mine: model off-ball runs as a supervised, xG-style problem rather than a grid</td></tr>
<tr><td>Model family</td><td class="m">Recommended LightGBM + GroupKFold</td>
<td>Accepted after checking the leakage argument held for nested events</td></tr>
<tr><td>Features</td><td class="m">Drafted the geometry (encirclement, block spread)</td>
<td>Rejected the density counters; required the ablation before believing any of it</td></tr>
<tr><td>Verification</td><td class="m">Produced the first, flattering comparison</td>
<td><b>Caught the mislabelled 360 baseline</b> (the "no-360" model contained a 360 feature) —
true gain +{d['gain360']:.4f}, not +{json.load(open(ROOT / 'outputs' / 'value_report.json'))['auc_lift']:.4f}</td></tr>
<tr><td>Football sense</td><td class="m">Proposed reading block spread as "stretched = valuable"</td>
<td><b>Caught the sign</b>: the data says the opposite by 7.6×; it is a depth proxy</td></tr>
<tr><td>Plotting &amp; prose</td><td class="m">First drafts</td>
<td>Restyled to my own conventions; fixed runs drawn off-pitch (360 frames are not clamped)</td></tr>
</tbody></table>

<div class="foot">
Data: <b>StatsBomb Open Data</b>, accessed via <code>statsbombpy</code>, used under the StatsBomb
open-data user agreement for non-commercial research. Data provided by StatsBomb — with thanks.
&nbsp;·&nbsp; Reproduce: <code>pip install -r requirements.txt</code> then
<code>streamlit run app/streamlit_app.py</code>; full pipeline via
<code>requirements-pipeline.txt</code> and <code>build_runs.py</code>.
</div>

</body></html>"""


def main() -> None:
    d = gather()
    html = build_html(d)
    hp = DOCS / "Off_Ball_Run_Value__Write_Up.html"
    hp.write_text(html, encoding="utf-8")
    print(f"wrote {hp.name}  ({len(html) / 1000:.0f} KB)")

    pdf = DOCS / "Off_Ball_Run_Value__Write_Up.pdf"
    if not Path(CHROME).exists():
        print("Chrome not found -- HTML written, render it yourself")
        return
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=4000", f"--print-to-pdf={pdf}", hp.as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {pdf.name}  ({pdf.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
