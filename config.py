"""
Central configuration.

Everything competition-specific lives here so the pipeline is portable:
swap COMPETITION_ID / SEASON_ID and the whole build re-runs on another
StatsBomb open-data competition that has 360 coverage.

The project is a 360-native SPACE model. Three metrics hang off one engine:
  1. Space Value   -- pitch control x threat (EPV) per player / team
  2. Decision Index -- did the passer choose the highest-value option available?
  3. Space Denial  -- how much dangerous space a defender/team removes (counterfactual)
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Competition scope
# ---------------------------------------------------------------------------
# THE FOUR CLUB SEASONS IN STATSBOMB'S OPEN 360 DATA.
#
# There is no full league season with 360 in the open data -- every "league"
# release is a single club's season. Pooling all four gives 127 matches and,
# critically for a MOVEMENT metric, 26-93 matches per player (Messi appears in
# three of them). A tournament gives at most 7. Depth per player is what an
# off-ball running metric needs, so we pool.
#
# Trade-off, stated in the write-up: four different leagues/eras, each a single
# strong team, so players are compared across tactical systems and opponent
# quality. `competition` is carried on every row so it can be controlled for.
COMPETITIONS = [
    (11, 90,  "La Liga 2020/21 (Barcelona)"),
    (7,  235, "Ligue 1 2022/23 (PSG)"),
    (7,  108, "Ligue 1 2021/22 (PSG)"),
    (9,  281, "Bundesliga 2023/24 (Bayer Leverkusen)"),
]
COMPETITION_NAME = "Four club seasons (Barcelona, PSG x2, Leverkusen)"

# kept for any single-competition path / cache naming
COMPETITION_ID, SEASON_ID = 9, 281

# ---------------------------------------------------------------------------
# Pitch geometry (StatsBomb coords: 120 x 80, team in possession attacks -> x=120)
# ---------------------------------------------------------------------------
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = (120.0, 40.0)

# Threat surface (grid Expected Threat) -- the "how dangerous is this spot" layer
THREAT_NX = 24
THREAT_NY = 16
XT_ITERATIONS = 40

# Pitch-control surface
# NB: 360 is a single freeze-frame with NO velocity, so this is a POSITIONAL
# pitch-control model (isotropic Gaussian player influence), not a Spearman
# time-to-arrive model. Stated as a limitation; with tracking we'd add motion.
PC_SIGMA = 9.0            # player influence radius (metres)
PC_GRID_NX = 48           # surface resolution for aggregates / metrics
PC_GRID_NY = 32
PC_GRID_NX_VIZ = 72       # finer resolution for the app surfaces
PC_GRID_NY_VIZ = 48

# ---------------------------------------------------------------------------
# Metric parameters
# ---------------------------------------------------------------------------
PROGRESSIVE_MIN_GAIN = 5.0
PACKING_BAND = 6.0        # +/- metres to count a bypassed defender (line-break maps)

DECISION_MIN_OPTIONS = 3  # need >= this many teammate options to score a decision
FINAL_THIRD_X = 80.0      # x beyond which we call it the final third

# Reception Value: a reception "led somewhere" if the team shoots or enters the
# final third within this many on-ball actions.
RECEPTION_HORIZON = 5

MIN_MINUTES = 600         # minutes floor for player leaderboards

# ---------------------------------------------------------------------------
# Off-ball run reconstruction
# ---------------------------------------------------------------------------
# A run is measured over the BALL FLIGHT window: from the passer's release
# freeze-frame to the receiver's ball-receipt freeze-frame.
MAX_RUN_SPEED = 9.5       # m/s ceiling used to bound a plausible run
MIN_RUN_DISTANCE = 2.0    # below this the receiver did not meaningfully move
MATCH_AMBIGUITY = 1.5     # 2nd-nearest / nearest ratio below this = ambiguous match
IN_BEHIND_MAX_DEF = 1     # <= this many defenders goalside => run went in behind
RUN_HORIZON = 5           # actions after the reception used for the outcome label

# StatsBomb clamps EVENT locations to the pitch but NOT 360 freeze-frame player
# positions -- raw frames run to x in [-5.8, 124.3], y in [-7.8, 89.4]. A player
# genuinely can be a metre or so off the touchline (throw-ins, stepping out), but
# 7 m outside is broadcast-projection noise or a bad match, and the inferred run
# origin is then wrong. Within tolerance we clip to the boundary; beyond it the
# reconstruction is not trusted.
PITCH_TOLERANCE = 2.0     # metres a player may legitimately be off the pitch

# subsampling to keep per-frame space computations tractable over 125k actions
SPACE_SAMPLE = 30000      # attacking on-ball frames sampled for space aggregates
DENIAL_SAMPLE = 12000     # defensive frames sampled for space-denial counterfactuals

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
for _p in (DATA_RAW, DATA_PROC, OUTPUTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 7
