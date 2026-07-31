"""
AI-ASSISTED MODULE. Implemented against my existing published visual grammar rather than
an invented one: passes are comets, runs and carries are dotted. The assistant was
directed to read player-print/core/viz/pitch.py as the source of truth.
See AI_USAGE.md section 2.3.

Pitch maps in the playerprint house style.

THE TWO RULES (same as player-print/core/viz/pitch.py):
    PASSES  are COMET lines  (solid, tapering thin->thick, dot on the end)
    CARRIES are DOTTED lines  (dashed, dot on the end)

Palette lifted from the published cards. Ground #1e1e1e, white lines, rounded font.

Coordinates: this project is StatsBomb (y=0 at the TOP), and mplsoccer's statsbomb
pitch already inverts the y-axis to match — verified empirically. So, UNLIKE the Opta
playerprint, there is NO y-flip here; raw StatsBomb coordinates are passed straight in.
"""
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from mplsoccer import Pitch

# ----- palette (from playerprint core.ui.theme) ---------------------------
BG          = "#1e1e1e"
PAGE_BG     = "#131313"
PITCH_LINE  = "#ffffff"
PROG_PASS   = "#00ff7f"   # progressive pass  (comet)
PASS_SUCC   = "#24a8ff"   # completed pass    (comet)
PASS_FAIL   = "#9a9a9a"   # incomplete pass   (comet)
PROG_CARRY  = "#ff5959"   # progressive carry (dotted)
CARRY_OTHER = "#B266FF"   # other carry       (dotted)
TAKEON      = "#2EC5FF"
SHOT        = "#ff6be6"
GOAL        = "#ffd700"
BRAND       = "#877ae8"
AMBER       = "#e0b64a"
TEXT        = "#f2f3f5"
MUTED       = "#7a8087"
T_ELITE, T_ABOVE, T_AVG, T_BELOW, T_POOR = "#3ddc97", "#7ed07a", "#e0b64a", "#e08550", "#d9536a"

# team colours for freeze-frames / surfaces
ATK = BRAND         # team in possession
DEF = "#ff5959"     # defending team
BALL = "#ffd700"


def _register_font():
    """
    Use Arial Rounded Bold where it is installed, otherwise fall back cleanly.

    The font is licensed, so it is deliberately not shipped in this repo. A local
    fonts/ directory is checked first for anyone who has their own copy; failing
    that we take the macOS system location; failing that DejaVu Sans, which every
    matplotlib install has. Charts render either way, only the typeface changes.
    """
    root = Path(__file__).resolve().parents[1]
    for p in [root / "fonts" / "Arial Rounded Bold.ttf",
              Path("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf")]:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
            return font_manager.FontProperties(fname=str(p)).get_name()
    return "DejaVu Sans"

FONT = _register_font()

# Arial Rounded Bold has no glyph for several characters we actually hit:
#   c-acute (Milinkovic-Savic, Brozovic), c-caron, the middle dot, and arrows.
# matplotlib >= 3.6 accepts a family LIST and falls back per glyph, so names with
# Slavic diacritics render correctly instead of as empty boxes.
FONT_STACK = [FONT, "DejaVu Sans"]

# NOTE ON GLYPHS: Arial Rounded Bold has no MIDDLE DOT (U+00B7). Passing "·" into a
# matplotlib title renders an empty box. Inside figures use "|" or "-" as separators;
# "·" is fine in Streamlit markdown, which uses the browser's fonts.


# ----- pitch + primitives -------------------------------------------------
def direction_of_play(ax, y=86.5, x0=48, x1=72):
    """A left-to-right arrow under the pitch.

    Every map in this project is drawn in StatsBomb's normalised frame, where the
    team in possession always attacks towards x=120. A reader cannot know that by
    looking, so each pitch says so explicitly.
    """
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#9aa3d1", lw=1.6,
                                mutation_scale=14), annotation_clip=False, zorder=10)
    ax.text((x0 + x1) / 2, y - 2.4, "Direction of play", color="#9aa3d1",
            fontsize=9.5, family=FONT_STACK, ha="center", va="bottom",
            zorder=10, clip_on=False)


def new_pitch(figsize=(11, 7.4), half=False, direction=True):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(BG); ax.set_facecolor(BG)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG, line_color=PITCH_LINE,
                  linewidth=1.4, line_zorder=2, half=half, pad_top=6, pad_bottom=10)
    pitch.draw(ax=ax)
    if direction:
        direction_of_play(ax)
    return fig, ax, pitch


def comet(pitch, ax, x, y, ex, ey, colour, lw=3.0, alpha=1.0, z=4, dot=55):
    """A PASS: tapering comet line with a dot on the receiving end."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return
    pitch.lines(x, y, ex, ey, lw=lw, comet=True, transparent=True,
                color=colour, ax=ax, zorder=z, alpha=alpha)
    if dot > 0:
        pitch.scatter(ex, ey, s=dot, marker="o", edgecolors="none", c=colour,
                      ax=ax, zorder=z + 1, alpha=alpha)


def dotted(pitch, ax, x, y, ex, ey, colour, lw=2.3, alpha=0.95, z=4, dot=45):
    """A CARRY: dotted line with a dot on the end."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return
    for x0, y0, x1, y1 in zip(x, np.asarray(y, float), np.asarray(ex, float), np.asarray(ey, float)):
        ax.plot([x0, x1], [y0, y1], color=colour, lw=lw, ls=(0, (1.6, 2.2)),
                alpha=alpha, zorder=z, solid_capstyle="round")
    pitch.scatter(ex, ey, s=dot, marker="o", edgecolors="none", c=colour,
                  ax=ax, zorder=z + 1, alpha=alpha)


def marks(pitch, ax, x, y, colour, marker="o", s=110, z=5, hollow=False, lw=1.6, alpha=1.0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return
    if hollow:
        pitch.scatter(x, y, s=s, marker=marker, edgecolors=colour, c="none",
                      lw=lw, ax=ax, zorder=z, alpha=alpha)
    else:
        pitch.scatter(x, y, s=s, marker=marker, c=colour, edgecolors=BG, lw=.6,
                      ax=ax, zorder=z, alpha=alpha)


def legend(ax, items, loc="upper center"):
    """items: list of (label, colour, kind) with kind in comet|dotted|mark."""
    if not items:
        return
    handles = []
    for label, colour, kind in items:
        if kind == "mark":
            handles.append(Line2D([], [], marker="o", color=colour, lw=0, markersize=9, label=label))
        elif kind == "dotted":
            handles.append(Line2D([], [], color=colour, lw=2.4, ls=(0, (1.6, 2.2)), label=label))
        else:
            handles.append(Line2D([], [], color=colour, lw=3.2, label=label))
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=(0.5, -0.01),
                    ncol=len(handles), frameon=False, fontsize=11,
                    labelcolor="#c8c8c8", handlelength=2.0, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_family(FONT_STACK)


def title(fig, name, subtitle):
    """Name over subtitle.

    Long names (e.g. "Francisco Antonio Machado Mota de Castro Trincao") wrap to a
    second line, which used to run straight into the subtitle. The subtitle now
    sits low enough to clear a two-line name, and the name shrinks a little once
    it gets long.
    """
    size = 20 if len(str(name)) <= 26 else 17
    fig.text(0.5, 0.975, name, color="white", fontsize=size, fontweight="bold",
             family=FONT_STACK, ha="center", va="top")
    if subtitle:
        fig.text(0.5, 0.912, subtitle, color="#c8c8c8", fontsize=12,
                 family=FONT_STACK, ha="center", va="top")


# ----- surfaces + freeze-frames ------------------------------------------
def surface(ax, gx, gy, surf, cmap="magma", alpha=0.72, levels=16):
    ax.contourf(gx, gy, surf, levels=levels, cmap=cmap, alpha=alpha, zorder=1)


def option_overlay(pitch, ax, origin, options, chosen=None, best=None):
    """
    Every teammate as a passing OPTION, drawn as a comet whose width and colour
    carry its value -- the "who is actually a good option, and why" read.
    """
    if options is None or len(options) == 0:
        return
    vmax = max(float(options["option_value"].max()), 1e-9)
    for _, o in options.iterrows():
        frac = float(o["option_value"]) / vmax
        col = PROG_PASS if frac > .66 else AMBER if frac > .33 else PASS_FAIL
        comet(pitch, ax, [origin[0]], [origin[1]], [o["x"]], [o["y"]], col,
              lw=0.9 + 4.2 * frac, alpha=.35 + .55 * frac, z=4, dot=18 + 60 * frac)


def freeze_players(pitch, ax, mates, defs, ball=None, s=210):
    if defs is not None and len(defs):
        pitch.scatter(defs[:, 0], defs[:, 1], ax=ax, s=s, color=DEF,
                      edgecolors="#1a0f0f", lw=1.0, zorder=6)
    if mates is not None and len(mates):
        pitch.scatter(mates[:, 0], mates[:, 1], ax=ax, s=s, color=ATK,
                      edgecolors="#12102a", lw=1.0, zorder=6)
    if ball is not None:
        pitch.scatter([ball[0]], [ball[1]], ax=ax, s=90, color=BALL,
                      edgecolors="#5a4a00", lw=1.0, zorder=8)


def run_explorer(mates_rel, defs_rel, mates_rec, defs_rec, ball_before,
                 origin, receipt, label=None, subtitle=None):
    """
    ONE run, told properly: the two freeze-frames overlaid.

    faded    = where everyone stood when the pass was STRUCK
    solid    = where they stood when it was RECEIVED
    comet    = the pass (ball's path)          -- house convention for a pass
    dotted   = the RUN (the player's path)     -- house convention for movement
    """
    fig, ax, pitch = new_pitch(figsize=(11, 7.2))

    # --- release frame: ghosted underneath -------------------------------
    if defs_rel is not None and len(defs_rel):
        pitch.scatter(defs_rel[:, 0], defs_rel[:, 1], ax=ax, s=150, color=DEF,
                      alpha=.22, edgecolors="none", zorder=3)
    if mates_rel is not None and len(mates_rel):
        pitch.scatter(mates_rel[:, 0], mates_rel[:, 1], ax=ax, s=150, color=ATK,
                      alpha=.22, edgecolors="none", zorder=3)

    # --- receipt frame: solid on top -------------------------------------
    if defs_rec is not None and len(defs_rec):
        pitch.scatter(defs_rec[:, 0], defs_rec[:, 1], ax=ax, s=200, color=DEF,
                      edgecolors="#1a0f0f", lw=1.0, zorder=6)
    if mates_rec is not None and len(mates_rec):
        pitch.scatter(mates_rec[:, 0], mates_rec[:, 1], ax=ax, s=200, color=ATK,
                      edgecolors="#12102a", lw=1.0, zorder=6)

    # --- the pass: a comet from the passer to the reception --------------
    comet(pitch, ax, [ball_before[0]], [ball_before[1]],
          [receipt[0]], [receipt[1]], GOAL, lw=3.6, z=7, dot=90)
    # --- the run: dotted, origin -> reception ----------------------------
    dotted(pitch, ax, [origin[0]], [origin[1]], [receipt[0]], [receipt[1]],
           PROG_PASS, lw=3.0, z=8, dot=120)
    # start of the run, hollow, so the movement reads directionally
    pitch.scatter([origin[0]], [origin[1]], ax=ax, s=190, facecolors="none",
                  edgecolors=PROG_PASS, lw=2.2, zorder=9)
    # the ball at release
    pitch.scatter([ball_before[0]], [ball_before[1]], ax=ax, s=110, color=BALL,
                  edgecolors="#5a4a00", lw=1.0, zorder=10)

    legend(ax, [("pass", GOAL, "comet"), ("the run", PROG_PASS, "dotted"),
                ("ghosted = at pass release", "#6a6a72", "mark")])
    if label:
        title(fig, label, subtitle or "")
    return fig
