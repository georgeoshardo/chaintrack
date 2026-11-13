"""chaintrack: a mother-machine trench tracked as a chain of cells judged by size and spacing.

The trench is an ordered chain of cells from the closed end. Between frames:
  - each cell's length grows by the fitted rate; it divides into two near-halves with a
    probability that rises with its length (a sizer); otherwise its length cannot change
    by more than max_length_change;
  - the gap above each cell persists, with small noise, and rarely jumps (a cell lysed
    above it, a cell squeezed in, part of the column slid);
  - the top cell stays against the closed end, and rarely jumps with the whole column;
  - a cell may lyse, leave past the bottom once growth has pushed it there, or go one to
    three frames without a mask; the cells below keep their gap to the last cell above with
    a mask, so a lysed or unseen cell's space is expected to stay, and the column closing
    over it is priced as a jump, not forbidden;
  - no new cells appear after the first frame.
The masks (shapes only, no image) are observations, cleaned first: any mask shorter
than the smallest plausible newborn, (div_mu - 3 div_sd) / 2, is removed whole, and with
the recipe's crop_bottom_um so is a mask with more than half its area in those bottom rows (for
crops that reach the halo at the trench opening; PyMMM's crops stop above it, so 0 by default),
and with clear_bottom_um any mask reaching those bottom rows (a cell cut off by the crop's edge;
segmentation often stops such a mask a few pixels short of the edge); the masks kept are kept whole. What is left is: one mask per cell, one cell
cut into two masks, one mask over two cells. Each frame the predicted chain is aligned to
the masks in order; a cell is judged by its own length and by the gap between its mask
and the mask of the last cell above that has one, so errors never build up down the column.
A beam of lineage histories lets the frames after an event decide what it was.
At the open end and in lysis waves, three switches in the recipe (all on in the example recipe):
  - a cell cut off by the bottom edge has an unknown length, but no shorter than predicted: it can
    show whole again only that long (not as a short mask, which is its upper daughter, nor shared
    with its sister in one short mask). The recipe's trunc_bound;
  - a cell may divide with only its upper daughter in view, the lower one too far past the bottom
    to leave a mask; the graph shows the mother with one child, and the GEFF's extra "div_one"
    lists those daughters' first nodes. The recipe's div_one;
  - a cell that went unseen comes back within revival_um_per_min per unseen minute of where it was
    predicted, never by a jump (no lysed cell taking a survivor's mask). The recipe's no_revival.
Cells off the centre line (side cells, beside the column) are tracked first and kept out of
the chain: each follows the nearest similar mask, reaching the centre line only by drifting.
Where the base merges a side cell into a chain cell and an alternative segmentation draws them
apart, that cut is used (see split_side). Their tracks are not written unless the recipe's
write_side is on.

The constants set by hand are a recipe (a toml; tests/data/recipe.toml is a complete one, each
key explained), in µm and minutes, so one recipe serves every magnification and
frame rate: chaintrack reads the store's pixel size and frame interval and works in pixels and
frames (RECIPE_KEYS says how each key is converted; lengths counted in pixels become whole
pixels, a probability per minute becomes the probability over one frame, and the segmentation
error probabilities stay per frame). Every key is required, and unknown keys are refused.
Only `fit` reads the recipe file (fit --recipe FILE): it freezes the recipe in the store's fit
record, and `run` and `follow` use that copy, so editing the file for another experiment cannot
change this one's tracks. Another recipe for this experiment is another fit, under its own name
(--fit NAME). A fit that a run has used is never overwritten, and a run is continued only with
the fit, segmentation and alternatives it was started with.

Fitted without labels: growth, length noise, length at division, jump size. Each store
has its own fit: `fit` measures them on that store's trenches and records them in the store,
attrs.pymmm.fits[<fit>] (in µm and minutes, with the trenches used, the recipe and chaintrack's
commit; --fit NAME, default the segmentation's name), and `run` reads them from there. A store's
first fit starts from the recipe's [fit_start], a later one from the store's last.
The gap noise (how much the gap above a cell changes from one frame to the next) is not fitted:
it is the recipe's sigma_gap_um. Gap changes are whole pixels and more than half of them are 0,
so the robust spread the fit uses reads them as no noise at all, at any magnification; as a
plain spread they were about 0.07 µm in both the 40x and the 100x data chaintrack was built on.

The masks come from the experiment group PyMMM extracts (--store PATH: one OME-Zarr image per trench,
<experiment>.zarr; see pymmm.store.experiment_group): a trench's segmentation <seg> is its label image
labels/<seg>.<hypothesis>, one (t, y, x) label image per hypothesis, the base first. `run` writes each trench's tracks
beside it, as the run <out> (default chain): the GEFF tracks/<out>.geff and the masks it chose,
labels/tracks.<out>. A tracker never writes curation/ (mm-track makes the curated copy at the first
edit), so re-running replaces tracker output only. `follow` does the same for each trench as soon
as its segmentation is finished, while PyMMM's Segmenter is still segmenting the others.
A run's record in the store (attrs.pymmm.tracks[<out>]) holds the fit, the fit's recipe, the
parameters as used (pixels and frames), the store's calibration and chaintrack's commit.

Trenches are independent, so `fit`, `run` and `follow` track several at once, each in its own
process: as many as the cores and the free memory allow, or --workers N. A trench's tracks do not
depend on how many are tracked at once.

    S="--store experiment.zarr --seg <segmentation>"
    python chain.py fit 0-7 $S --recipe tests/data/recipe.toml
    python chain.py run 0-99 $S
    python chain.py follow 0-99 $S --out chain_live
    python chain.py run 0-9 $S --base-only --out chain_base

A run with other constants (a wider beam, clear_bottom_um, the open-end switches off) needs a fit
made with another recipe file:

    python chain.py fit 0-7 $S --recipe wide.toml --fit wide
    python chain.py run 0-9 $S --fit wide --out chain_wide
"""

import itertools
import json
import os
import subprocess
import sys
import time
import tomllib
from functools import lru_cache
from pathlib import Path

import numba
import numpy as np
from pymmm.store.experiment_group import calibration, open_experiment, records
from scipy import ndimage as ndi
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
TRACKER = "chain"


def option(flag):
    """The value after ``flag`` on the command line, or None."""
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


USAGE = (
    "chain.py fit <trenches> --store <experiment>.zarr --seg <segmentation> --recipe <file> [--fit NAME] [--workers N]\n"
    "chain.py run|follow <trenches> --store <experiment>.zarr --seg <segmentation> [--fit NAME] [--out RUN] [--workers N]"
)
# what to do: fit, run or follow
COMMAND = sys.argv[1] if len(sys.argv) > 1 else ""
if COMMAND not in ("fit", "run", "follow"):
    sys.exit(USAGE)
# the experiment group to track: one OME-Zarr image per trench (--store PATH)
STORE = option("--store")
# which segmentation's masks to track (--seg NAME)
SEG = option("--seg")
# the constants set by hand, in µm and minutes (--recipe FILE). Only `fit` reads the file: it freezes
# the recipe in the store's fit record, and `run` and `follow` take it from there, so a recipe file
# edited later for another experiment cannot change this experiment's tracks
RECIPE_FILE = option("--recipe")
for _flag, _value in (("--store", STORE), ("--seg", SEG)) + (
    (("--recipe", RECIPE_FILE),) if COMMAND == "fit" else ()
):
    if _value is None:
        sys.exit(f"chain.py: give {_flag}\n{USAGE}")
if COMMAND != "fit" and RECIPE_FILE is not None:
    sys.exit(
        f"chain.py: {COMMAND} takes no --recipe: it uses the recipe frozen in the store's fit. For another "
        "recipe, fit again under a new name (fit ... --recipe <file> --fit NAME) and track with --fit NAME"
    )
STORE = Path(STORE)
# the fit to use, or to make: the store's record attrs.pymmm.fits[<fit>] (--fit NAME, default the segmentation's name)
FIT = option("--fit") or SEG
# a segmentation with several hypotheses: the first is the base, the others its alternatives (which
# hypotheses those are is in the store's record of the segmentation); --alts a,b
# uses only those, --base-only the base alone
ALTS = option("--alts").split(",") if "--alts" in sys.argv else None
BASE_ONLY = "--base-only" in sys.argv
OUT = option("--out") or TRACKER
# how many trenches to track at once (--workers N); unset, as many as the cores and the free memory allow
WORKERS = option("--workers")
if WORKERS is not None and not (WORKERS.isdigit() and int(WORKERS) >= 1):
    sys.exit(f"chain.py: --workers takes a whole number of at least 1, not {WORKERS!r}")

# The recipe's keys: the parameter each becomes (in pixels and frames, as the tracker uses them) and how.
#   same          copied                      flag    copied, true or false       count   copied, a whole number
#   px            µm, as pixels               whole px    µm, as whole pixels     whole px2   µm², as a number of pixels
#   frames        minutes, as whole frames    px per frame    µm per minute
#   per frame     a probability per minute, as the probability over one frame
#   log per frame a log growth per minute, over one frame
RECIPE_KEYS = {
    "beam": ("beam", "count"),
    "max_length_change": ("max_length_change", "same"),
    "sigma_daughter": ("sigma_daughter", "same"),
    "sigma_gap_um": ("sigma_gap", "px"),
    "p_jump_per_min": ("p_jump", "per frame"),
    "p_slide_per_min": ("p_slide", "per frame"),
    "p_lysis_per_min": ("p_lysis", "per frame"),
    "appear_cost": ("appear_cost", "same"),
    "p_over": ("p_over", "same"),
    "p_under": ("p_under", "same"),
    "p_under_sisters": ("p_under_sisters", "same"),
    "sisters_min": ("sisters_frames", "frames"),
    "p_alt": ("p_alt", "same"),
    "p_miss": ("p_miss", "same"),
    "max_miss_frames": ("max_miss", "count"),
    "centre_um": ("centre_px", "whole px"),
    "exit_visible_um": ("exit_visible_px", "whole px"),
    "max_gap_um": ("max_gap", "whole px"),
    "cut_off_um": ("cut_off_px", "whole px"),
    "crop_bottom_um": ("crop_bottom_px", "whole px"),
    "clear_bottom_um": ("clear_bottom_px", "whole px"),
    "trunc_bound": ("trunc_bound", "flag"),
    "div_one": ("div_one", "flag"),
    "no_revival": ("no_revival", "flag"),
    "revival_um_per_min": ("revival_px", "px per frame"),
    "write_side": ("write_side", "flag"),
    "side_sigma_y_um": ("side_sigma_y", "px"),
    "side_sigma_x_um": ("side_sigma_x", "px"),
    "side_sigma_area": ("side_sigma_area", "same"),
    "side_max_cost": ("side_max_cost", "same"),
    "side_min_overlap": ("side_min_overlap", "same"),
    "side_min_piece_um2": ("side_min_piece_px", "whole px2"),
    "side_alt_cut": ("side_alt_cut", "flag"),
    "side_alt_piece": ("side_alt_piece", "flag"),
    "side_window_um": ("side_window_px", "whole px"),
}
# The fitted values, as a store's fit record and the recipe's [fit_start] hold them
FIT_KEYS = {
    "growth_per_min": ("growth", "log per frame"),  # log length per frame
    "sigma_end_um": (
        "sigma_end",
        "px",
    ),  # noise of each end of a mask (so a mask's length is noisy by about sqrt(2) of this)
    "div_um": ("div_mu", "px"),  # length at division
    "div_sd_um": ("div_sd", "px"),
    "jump_scale_um": ("jump_scale_px", "px"),  # typical size of a jump
}


def check_recipe(constants, start, where):
    """Stop unless a recipe is one this chaintrack can use: every key there, none unknown, each a
    number (or true/false for a flag, a whole number for a count). start: its [fit_start] values,
    or None for a recipe that carries none (one frozen in a store).
    """

    def number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    right = {
        "flag": lambda v: isinstance(v, bool),
        "count": lambda v: isinstance(v, int) and not isinstance(v, bool),
    }
    problems = []
    for prefix, table, keys in (("", constants, RECIPE_KEYS),) + (
        (("fit_start.", start, FIT_KEYS),) if start is not None else ()
    ):
        problems += [f"unknown key {prefix}{k}" for k in table if k not in keys]
        problems += [f"missing key {prefix}{k}" for k in keys if k not in table]
        problems += [
            f"{prefix}{k} = {v!r}: not a {'number' if keys[k][1] not in right else keys[k][1]}"
            for k, v in table.items()
            if k in keys and not right.get(keys[k][1], number)(v)
        ]
    if problems:
        sys.exit(f"chain.py: {where}:\n  " + "\n  ".join(problems))


def load_recipe(path):
    """The recipe file's constants and its [fit_start] values, checked."""
    with open(path, "rb") as file:
        raw = tomllib.load(file)
    start = raw.pop("fit_start", {})
    check_recipe(raw, start, f"recipe {path}")
    return raw, start


def to_pixels(value, how):
    """A recipe or fit value in the store's pixels and frames."""
    if how == "px":
        return value / PX
    if how == "whole px":
        return round(value / PX)
    if how == "whole px2":
        return round(value / PX**2)
    if how == "frames":
        return round(value / DT)
    if how == "px per frame":
        return value * DT / PX
    if how == "per frame":
        return 1.0 - (1.0 - value) ** DT
    if how == "log per frame":
        return value * DT
    return value


def fitted():
    """The fitted parameters as a fit record holds them: µm and minutes."""
    return {
        k: float(P[p] / DT if how == "log per frame" else P[p] * PX)
        for k, (p, how) in FIT_KEYS.items()
    }


@lru_cache(maxsize=1)
def code_version():
    """The commit of chaintrack, and whether tracked files had uncommitted changes.

    None outside git.
    """

    def git(*args):
        try:
            return subprocess.run(  # noqa: S603 - fixed git arguments
                ["git", "-C", str(HERE), *args],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


PX, DT = calibration(open_experiment(STORE))  # µm per pixel, minutes per frame
# the store's record of the fit: None until `fit` has run on this store
FIT_RECORD = records(open_experiment(STORE), "fits").get(FIT)
if COMMAND == "fit":
    RECIPE, FIT_START = load_recipe(RECIPE_FILE)
else:
    # tracking needs the store's own fit, and uses the recipe frozen in it, never a file
    if FIT_RECORD is None:
        sys.exit(
            f"chain.py: {STORE.name} has no fit {FIT!r} (its fits: {sorted(records(open_experiment(STORE), 'fits')) or 'none'}): "
            "run `chain.py fit <trenches> ... --recipe <file>` on it first"
        )
    if not FIT_RECORD.get("recipe"):
        sys.exit(
            f"chain.py: the fit {FIT!r} of {STORE.name} holds no recipe: fit again, under a new name"
        )
    RECIPE, FIT_START = dict(FIT_RECORD["recipe"]), None
    check_recipe(
        RECIPE,
        None,
        f"the recipe frozen in the fit {FIT!r} of {STORE.name} is not one this chaintrack can use "
        f"(the fit was made by chaintrack {(FIT_RECORD.get('chaintrack') or {}).get('commit')}); fit again, under a new name",
    )
# The parameters as the tracker uses them, in pixels and frames: the recipe's constants, and the
# fitted ones from the store's fit (from the recipe's [fit_start] until there is one)
P = {p: to_pixels(RECIPE[k], how) for k, (p, how) in RECIPE_KEYS.items()}
P.update(
    {
        p: to_pixels((FIT_RECORD["values"] if FIT_RECORD else FIT_START)[k], how)
        for k, (p, how) in FIT_KEYS.items()
    }
)

MOVE = {
    "match": 0,
    "div": 1,
    "over": 2,
    "under": 3,
    "miss": 4,
    "lysis": 5,
    "exit": 6,
    "appear": 7,
    "div_one": 8,
}
INF = 1e18


@numba.njit(cache=True)
def mix(r, sigma, p, scale):
    """-log of: no jump (Gaussian noise), or a jump, small ones commoner than big ones."""
    g = (1.0 - p) * np.exp(-0.5 * (r / sigma) ** 2) / (sigma * 2.5066282746310002)
    return -np.log(g + p * np.exp(-abs(r) / scale) / (2.0 * scale))


@numba.njit(cache=True)
def align(
    sm,
    sd,
    so,
    su,
    nodiv,
    divc,
    underc,
    overc,
    missc,
    lysisc,
    ptop,
    plen,
    otop,
    obot,
    H,
    sig_gap,
    p_jump,
    p_slide,
    rng,
    vis,
    sptr,
    sidx,
    pptr,
    pj2,
    end_ok,
    sd1,
    maxr,
    appear,
    force,
):
    """Order-preserving alignment of n predicted cells to a frame's masks. The masks form a graph
    (see Frame): state s = 0 before any mask, s = j + 1 once mask j is taken; sidx[sptr[s]:sptr[s + 1]]
    are the masks that can come next, pj2[pptr[j]:pptr[j + 1]] the masks that can follow mask j
    (for a division or a cell cut into two masks; sd and so are indexed by that position), and
    end_ok[s] says whether the frame's masks may end there. Search state (i cells, s, a = 1 + the
    last cell given a mask, 0 if none yet). Spacing is judged against the mask of that cell (or
    the closed end); size by the move's own terms.
    """
    n = sm.shape[0]
    S = end_ok.shape[0]
    dp = np.full((n + 1, S, n + 1), INF)
    bc = np.full((n + 1, S, n + 1), -1, np.int64)
    bi = np.zeros((n + 1, S, n + 1), np.int64)
    bs = np.zeros((n + 1, S, n + 1), np.int64)
    ba = np.zeros((n + 1, S, n + 1), np.int64)
    bm = np.full((n + 1, S, n + 1), -1, np.int64)
    bm2 = np.full((n + 1, S, n + 1), -1, np.int64)
    dp[0, 0, 0] = 0.0
    for i in range(n + 1):
        for s in range(S):
            for a in range(n + 1):
                c0 = dp[i, s, a]
                if c0 >= INF / 2:
                    continue
                if i < n:
                    # where cell i would start: after the last mask taken, plus the space between
                    if a == 0:
                        pt = ptop[i]
                    else:
                        pt = obot[s - 1] + 1 + (ptop[i] - (ptop[a - 1] + plen[a - 1]))
                    f = force[i]
                    gone = pt >= H - vis
                    for q in range(sptr[s], sptr[s + 1]):
                        j = sidx[q]
                        # spacing cost of putting cell i's top at mask j's top
                        if a == 0:
                            sp = mix(otop[j] - ptop[i], sig_gap, p_slide, rng)
                            if abs(otop[j] - ptop[i]) > maxr[i]:
                                sp = INF
                        else:
                            sp = mix(otop[j] - pt, sig_gap, p_jump, rng)
                            if abs(otop[j] - pt) > maxr[i]:
                                sp = INF
                        # match
                        if f < 0 or f == 0:
                            c = c0 + sp + sm[i, j] + nodiv[i]
                            if c < dp[i + 1, j + 1, i + 1]:
                                dp[i + 1, j + 1, i + 1] = c
                                bc[i + 1, j + 1, i + 1] = 0
                                bi[i + 1, j + 1, i + 1] = i
                                bs[i + 1, j + 1, i + 1] = s
                                ba[i + 1, j + 1, i + 1] = a
                                bm[i + 1, j + 1, i + 1] = j
                        # divide / cut into two masks
                        for r in range(pptr[j], pptr[j + 1]):
                            j2 = pj2[r]
                            if f < 0 or f == 1:
                                c = c0 + sp + sd[i, r] + divc[i]
                                if c < dp[i + 1, j2 + 1, i + 1]:
                                    dp[i + 1, j2 + 1, i + 1] = c
                                    bc[i + 1, j2 + 1, i + 1] = 1
                                    bi[i + 1, j2 + 1, i + 1] = i
                                    bs[i + 1, j2 + 1, i + 1] = s
                                    ba[i + 1, j2 + 1, i + 1] = a
                                    bm[i + 1, j2 + 1, i + 1] = j
                                    bm2[i + 1, j2 + 1, i + 1] = j2
                            if f < 0 or f == 2:
                                c = c0 + sp + so[i, r] + overc[i] + nodiv[i]
                                if c < dp[i + 1, j2 + 1, i + 1]:
                                    dp[i + 1, j2 + 1, i + 1] = c
                                    bc[i + 1, j2 + 1, i + 1] = 2
                                    bi[i + 1, j2 + 1, i + 1] = i
                                    bs[i + 1, j2 + 1, i + 1] = s
                                    ba[i + 1, j2 + 1, i + 1] = a
                                    bm[i + 1, j2 + 1, i + 1] = j
                                    bm2[i + 1, j2 + 1, i + 1] = j2
                        # divide, the lower daughter unseen past the bottom
                        if (f < 0 or f == 1) and sd1[i, j] < INF:
                            c = c0 + sp + sd1[i, j] + divc[i]
                            if c < dp[i + 1, j + 1, i + 1]:
                                dp[i + 1, j + 1, i + 1] = c
                                bc[i + 1, j + 1, i + 1] = 8
                                bi[i + 1, j + 1, i + 1] = i
                                bs[i + 1, j + 1, i + 1] = s
                                ba[i + 1, j + 1, i + 1] = a
                                bm[i + 1, j + 1, i + 1] = j
                        # two cells in one mask
                        if (
                            i + 1 < n
                            and (f < 0 or f == 3)
                            and (force[i + 1] < 0 or force[i + 1] == 3)
                        ):
                            c = c0 + sp + su[i, j] + underc[i] + nodiv[i] + nodiv[i + 1]
                            if c < dp[i + 2, j + 1, i + 2]:
                                dp[i + 2, j + 1, i + 2] = c
                                bc[i + 2, j + 1, i + 2] = 3
                                bi[i + 2, j + 1, i + 2] = i
                                bs[i + 2, j + 1, i + 2] = s
                                ba[i + 2, j + 1, i + 2] = a
                                bm[i + 2, j + 1, i + 2] = j
                    # no mask for cell i
                    if not gone:
                        if f < 0 or f == 4:
                            c = c0 + missc[i] + nodiv[i]
                            if c < dp[i + 1, s, a]:
                                dp[i + 1, s, a] = c
                                bc[i + 1, s, a] = 4
                                bi[i + 1, s, a] = i
                                bs[i + 1, s, a] = s
                                ba[i + 1, s, a] = a
                    near_bottom = (
                        ptop[i] + 2 * plen[i] >= H
                    )  # its own prediction, not one pushed by a jump above
                    if (not gone or not near_bottom) and (f < 0 or f == 5):
                        c = c0 + lysisc
                        if c < dp[i + 1, s, a]:
                            dp[i + 1, s, a] = c
                            bc[i + 1, s, a] = 5
                            bi[i + 1, s, a] = i
                            bs[i + 1, s, a] = s
                            ba[i + 1, s, a] = a
                    if (f < 0 or f == 6) and near_bottom:
                        if gone:
                            ec = 0.0
                        elif pt + plen[i] >= H:
                            ec = 1.0 if H - pt < 0.6 * plen[i] else 3.0
                        else:
                            ec = INF
                        if ec < INF:
                            c = c0 + ec
                            if c < dp[i + 1, s, a]:
                                dp[i + 1, s, a] = c
                                bc[i + 1, s, a] = 6
                                bi[i + 1, s, a] = i
                                bs[i + 1, s, a] = s
                                ba[i + 1, s, a] = a
                for q in range(sptr[s], sptr[s + 1]):  # a mask no cell explains
                    j = sidx[q]
                    c = c0 + appear
                    if c < dp[i, j + 1, a]:
                        dp[i, j + 1, a] = c
                        bc[i, j + 1, a] = 7
                        bi[i, j + 1, a] = i
                        bs[i, j + 1, a] = s
                        ba[i, j + 1, a] = a
                        bm[i, j + 1, a] = j
    best, best_s, best_a = INF, 0, 0
    for s in range(S):
        if end_ok[s]:
            for a in range(n + 1):
                if dp[n, s, a] < best:
                    best, best_s, best_a = dp[n, s, a], s, a
    rows = np.full((n + S + 1, 4), -1, np.int64)
    k = 0
    i, s, a = n, best_s, best_a
    while (i > 0 or s > 0) and bc[i, s, a] >= 0:
        code = bc[i, s, a]
        rows[k, 0] = code
        rows[k, 1] = -1 if code == 7 else bi[i, s, a]
        rows[k, 2] = bm[i, s, a]
        rows[k, 3] = bm2[i, s, a]
        k += 1
        i, s, a = bi[i, s, a], bs[i, s, a], ba[i, s, a]
    return best, rows[:k][::-1]


class Hyp:
    __slots__ = (
        "born",
        "ids",
        "length",
        "parent",
        "rec",
        "score",
        "top",
        "trunc",
        "unseen",
    )

    def __init__(self, ids, top, length, parent, born, unseen, trunc, score, rec):
        self.ids, self.top, self.length, self.parent = ids, top, length, parent
        self.born, self.unseen, self.trunc, self.score, self.rec = (
            born,
            unseen,
            trunc,
            score,
            rec,
        )


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return self.n


def frame_masks(frame, H):
    W = frame.shape[1]
    flat = frame.ravel()
    out = []
    for l in np.unique(flat):
        if l == 0:
            continue
        pix = np.flatnonzero(flat == l)
        rows, cols = pix // W, pix % W
        out.append(
            (
                int(rows.min()),
                int(rows.max()),
                int(l),
                bool(rows.max() >= H - P["cut_off_px"]),
                float(rows.mean()),
                len(pix),
                float(cols.mean()),
            )
        )
    return out


def side_cost(o, last):
    """How unlike the off-centre cell `last` the mask `o` is: it can move along the trench a fair
    way, but keeps its size and stays near its own sideways position.
    """
    return (
        ((o[4] - last[4]) / P["side_sigma_y"]) ** 2
        + (np.log(o[5] / last[5]) / P["side_sigma_area"]) ** 2
        + ((o[6] - last[6]) / P["side_sigma_x"]) ** 2
    )


def may_follow(o, frame, previous, centre):
    """An off-centre cell may continue on a mask on the centre line only by drifting (or turning)
    onto it, so that the mask overlaps its last one, not by jumping to a chain cell next to it.
    previous: (frame, label) of the cell's last mask.
    """
    if abs(o[6] - centre) > P["centre_px"]:
        return True
    a = np.flatnonzero(frame.ravel() == o[2])
    b = np.flatnonzero(previous[0].ravel() == previous[1])
    return np.intersect1d(a, b, assume_unique=True).size >= P["side_min_overlap"] * min(
        len(a), len(b)
    )


def mask_stats(pix, W):
    rows, cols = pix // W, pix % W
    return (
        int(rows.min()),
        int(rows.max()),
        float(rows.mean()),
        len(pix),
        float(cols.mean()),
    )


def alt_side_piece(frame, alt_frames, last, centre, claimed):
    """For an off-centre cell with no base mask of its own this frame: the piece of an unclaimed
    base mask that an alternative draws as a mask of its own, off-centre and like the cell (the
    base merged the cell into a chain cell). (host label, flat pixels of the piece) or None.
    """
    W = frame.shape[1]
    flat = frame.ravel()
    lo, hi = (
        max(0, int(last[0]) - P["side_window_px"]),
        int(last[1]) + P["side_window_px"] + 1,
    )
    best = None
    for alt in alt_frames:
        window = alt[lo:hi]
        for q in np.unique(window[window > 0]):
            pix = np.flatnonzero(alt.ravel() == q)
            top, bottom, row, area, col = mask_stats(pix, W)
            if abs(col - centre) <= P["centre_px"]:
                continue
            o = (top, bottom, 0, False, row, area, col)
            c = side_cost(o, last)
            if c > P["side_max_cost"] or (best is not None and c >= best[0]):
                continue
            hosts = flat[pix]
            hosts = hosts[hosts > 0]
            if not len(hosts):
                continue
            host = int(np.bincount(hosts).argmax())
            if host in claimed:
                continue
            host_pix = np.flatnonzero(flat == host)
            piece = np.intersect1d(pix, host_pix, assume_unique=True)
            if (
                len(piece) < 0.5 * len(pix)
                or len(host_pix) - len(piece) < P["side_min_piece_px"]
                or len(piece) < P["side_min_piece_px"]
            ):
                continue
            best = (c, host, piece)
    return None if best is None else best[1:]


def alt_centre_cut(frame, alt_frames, host, centre):
    """For a new off-centre base mask: the off-centre piece that alternatives cut out of it, leaving
    a piece on the centre line (the base merged a side cell into a chain cell), if at least two
    alternatives cut it alike (IoU >= 0.8). Flat pixels of the piece, or None.
    """
    W = frame.shape[1]
    host_pix = np.flatnonzero(frame.ravel() == host)
    found = []  # (piece, votes)
    for alt in alt_frames:
        inside = alt.ravel()[host_pix]
        centred, off = False, []
        for q in np.unique(inside[inside > 0]):
            pix = np.flatnonzero(alt.ravel() == q)
            piece = np.intersect1d(pix, host_pix, assume_unique=True)
            if len(piece) < 0.5 * len(pix) or len(piece) < P["side_min_piece_px"]:
                continue
            if abs(mask_stats(piece, W)[4] - centre) <= P["centre_px"]:
                centred = True
            else:
                off.append(piece)
        if (
            not (centred and len(off) == 1)
            or len(host_pix) - len(off[0]) < P["side_min_piece_px"]
        ):
            continue
        for k, (piece, votes) in enumerate(found):
            inter = np.intersect1d(piece, off[0], assume_unique=True).size
            if inter / (len(piece) + len(off[0]) - inter) >= 0.8:
                found[k] = (piece, votes + 1)
                break
        else:
            found.append((off[0], 1))
    found = [f for f in found if f[1] >= 2]
    return max(found, key=lambda f: f[1])[0] if found else None


def split_side(lab, centre, H, alts=()):
    """Chain masks per frame, and off-centre cells tracked first (an off-centre cell keeps its
    identity through the nearest similar mask, so it survives drifting onto the centre line, but it
    cannot jump across onto a chain cell). With alternative segmentations (alts, label stacks like
    lab), a side cell the base merges into a chain cell continues on the piece an alternative draws
    for it, and a new off-centre mask that alternatives cut into a chain piece and a side piece is
    cut: lab is relabelled in place there (the piece gets a new label, the rest keeps the mask's).
    """
    obs, tracks, active, nid, side_labels = [], {}, [], 0, []

    def last_mask(tid):
        t_last, label = tracks[tid][-1]
        return lab[t_last], label

    for t in range(lab.shape[0]):
        frame = lab[t]
        alt_t = [a[t] for a in alts]
        masks = frame_masks(frame, H)
        claimed, nxt, waiting = set(), [], []
        for tid, last, missed in active:
            best = None
            for o in masks:
                if o[2] in claimed or not may_follow(o, frame, last_mask(tid), centre):
                    continue
                c = side_cost(o, last)
                if c <= P["side_max_cost"] and (best is None or c < best[0]):
                    best = (c, o)
            if best is not None:
                o = best[1]
                claimed.add(o[2])
                tracks[tid].append((t, o[2]))
                nxt.append((tid, o, 0))
            else:
                waiting.append((tid, last, missed))
        still = []
        for tid, last, missed in waiting:
            cut = (
                alt_side_piece(frame, alt_t, last, centre, claimed)
                if alt_t and P["side_alt_piece"]
                else None
            )
            if cut is not None:
                _, piece = cut
                new = int(frame.max()) + 1
                frame.ravel()[piece] = new
                masks = frame_masks(frame, H)
                o = next(m for m in masks if m[2] == new)
                claimed.add(new)
                tracks[tid].append((t, new))
                nxt.append((tid, o, 0))
            else:
                still.append((tid, last, missed))
        for label in [
            o[2]
            for o in masks
            if o[2] not in claimed and abs(o[6] - centre) > P["centre_px"]
        ]:
            piece = (
                alt_centre_cut(frame, alt_t, label, centre)
                if alt_t and P["side_alt_cut"]
                else None
            )
            if piece is not None:
                label = int(frame.max()) + 1
                frame.ravel()[piece] = label
                masks = frame_masks(frame, H)
            o = next(m for m in masks if m[2] == label)
            # a piece just cut out may be the side cell that lost its mask to the merge
            best = None
            if piece is not None:
                for k, (tid, last, _missed) in enumerate(still):
                    c = side_cost(o, last)
                    if (
                        c <= P["side_max_cost"]
                        and may_follow(o, frame, last_mask(tid), centre)
                        and (best is None or c < best[0])
                    ):
                        best = (c, k)
            if best is not None:
                tid = still.pop(best[1])[0]
                tracks[tid].append((t, o[2]))
                claimed.add(o[2])
                nxt.append((tid, o, 0))
            else:
                nid += 1
                tracks[-nid] = [(t, o[2])]
                claimed.add(o[2])
                nxt.append((-nid, o, 0))
        for tid, last, missed in still:
            if (
                missed < P["max_miss"]
            ):  # no mask this frame, like a chain cell: wait for it
                nxt.append((tid, last, missed + 1))
        active = nxt
        obs.append(sorted(o[:6] for o in masks if o[2] not in claimed))
        side_labels.append({o[2] for o in masks if o[2] in claimed})
    return obs, tracks, side_labels


@numba.njit(cache=True)
def n_common(a, b):
    """How many values two sorted arrays of distinct values share."""
    i = j = n = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            n += 1
            i += 1
            j += 1
    return n


def label_groups(frame):
    """(label, flat pixel indices) for each label of one frame."""
    flat = frame.ravel()
    nz = np.flatnonzero(flat)
    if not len(nz):
        return []
    lab = flat[nz]
    order = np.argsort(lab, kind="stable")
    lab, nz = lab[order], nz[order]
    ids, first = np.unique(lab, return_index=True)
    bounds = [*list(first), len(lab)]
    return [(int(l), nz[bounds[k] : bounds[k + 1]]) for k, l in enumerate(ids)]


class Frame:
    """One frame's masks as the tracker sees them. With alternative segmentations (other
    checkpoints of the same model), masks of different hypotheses that overlap substantially
    form a region, and each region offers its distinct cuts: the base hypothesis's cut, and each
    alternative cut that covers at least half of every base mask in the region (an alternative
    may cut what the base sees differently, or add a mask where the base has none, but never
    drop a base mask). A mask matching a base mask (IoU >= 0.8) is that base mask. The tracker
    picks one cut per region while it aligns the cells; how many hypotheses give a cut does not
    count, and taking one costs the prior that the base is wrong there (p_alt). Alternatives only
    split base masks or add masks: a mask merging base masks is not offered (the move "one cell
    cut into two masks" already covers that). Divisions follow the base: two pieces another hypothesis cuts out of one base mask can be two cells already in
    the chain, never a mother's two daughters (other checkpoints split a constricting cell earlier
    than the base, whose split is when the curator calls the division). With the base alone, the
    graph is the base masks in order.

    masks: (top, bottom, candidate, cut off, mean row, area) per mask, in graph order;
    pix: flat pixel indices per candidate; base: the base masks' indices; sptr/sidx: the masks
    that can come next from each state (0 = before any mask, j + 1 = after mask j); pptr/pj/pj2:
    the pairs of masks that can follow each other (no_div: not as a mother's daughters);
    end_ok: states the frame may end at.
    """

    __slots__ = (
        "alt_first",
        "base",
        "end_ok",
        "masks",
        "no_div",
        "pix",
        "pj",
        "pj2",
        "pptr",
        "sidx",
        "sptr",
    )

    def __init__(self, base_obs, base_frame, alt_frames, side_labels, centre, H):
        W = base_frame.shape[1]
        groups = dict(label_groups(base_frame))
        info, pix = [], []
        for o in base_obs:
            info.append((o[0], o[1], len(info), o[3], o[4], o[5]))
            pix.append(groups[o[2]])
        nb = len(info)
        base_area = np.array([len(p) for p in pix], np.int64)
        idimg = np.zeros(base_frame.size, np.int64)
        for b in range(nb):
            idimg[pix[b]] = b + 1
        side = np.isin(base_frame.ravel(), list(side_labels)) if side_labels else None
        extras, cuts = [], []
        for fr in alt_frames:
            mine = []
            for _, q in label_groups(fr):
                cols = q % W
                if abs(cols.mean() - centre) > P["centre_px"] or (
                    side is not None and side[q].mean() > 0.5
                ):
                    continue
                same = -1
                cnt = np.bincount(idimg[q], minlength=nb + 1)
                cnt[0] = 0
                if np.count_nonzero(cnt[1:] >= 0.3 * base_area) >= 2:
                    continue  # a merge of base masks: "one cell cut into two masks" covers it
                b = int(np.argmax(cnt)) - 1
                if b >= 0 and cnt[b + 1] / (len(q) + len(pix[b]) - cnt[b + 1]) >= 0.8:
                    same = b
                for e in extras if same < 0 else ():
                    inter = n_common(q, pix[e])
                    if inter / (len(q) + len(pix[e]) - inter) >= 0.8:
                        same = e
                        break
                if same < 0:
                    rows = q // W
                    same = len(info)
                    info.append(
                        (
                            int(rows.min()),
                            int(rows.max()),
                            same,
                            bool(rows.max() >= H - P["cut_off_px"]),
                            float(rows.mean()),
                            len(q),
                        )
                    )
                    pix.append(q)
                    extras.append(same)
                mine.append(same)
            cuts.append(mine)
        # regions: candidates that overlap by at least 30% of the smaller one
        root = list(range(len(info)))

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        cover = {}
        for k, e in enumerate(extras):
            cnt = np.bincount(idimg[pix[e]], minlength=nb + 1)
            cover[e] = cnt
            for b in np.flatnonzero(cnt[1:]):
                if cnt[b + 1] >= 0.3 * min(len(pix[e]), len(pix[b])):
                    root[find(e)] = find(b)
            for e2 in extras[:k]:
                inter = n_common(pix[e], pix[e2])
                if inter >= 0.3 * min(len(pix[e]), len(pix[e2])):
                    root[find(e)] = find(e2)
        members = {}
        for c in range(len(info)):
            members.setdefault(find(c), []).append(c)

        def key(c):
            return (info[c][0], info[c][1], c)

        regions = []
        for mem in sorted(members.values(), key=lambda m: min(key(c) for c in m)):
            inside = set(mem)
            base_cut = tuple(sorted((c for c in mem if c < nb), key=key))
            options = [base_cut]
            for mine in cuts:
                cut = tuple(sorted((c for c in mine if c in inside), key=key))
                if not cut or cut in options:
                    continue
                got = np.zeros(nb + 1)
                for c in cut:
                    if c < nb:
                        got[c + 1] += len(pix[c])
                    else:
                        got += cover[c]
                if all(got[b + 1] >= 0.5 * len(pix[b]) for b in base_cut):
                    options.append(cut)
            regions.append(options)
        # the graph
        masks, opt_masks = [], []
        for options in regions:
            om = []
            for cut in options:
                om.append(list(range(len(masks), len(masks) + len(cut))))
                masks += [info[c] for c in cut]
            opt_masks.append(om)
        R, M = len(regions), len(masks)
        empty = [any(not cut for cut in options) for options in regions]
        tail = [True] * (R + 1)
        entry = [[] for _ in range(R + 1)]
        for r in range(R - 1, -1, -1):
            tail[r] = empty[r] and tail[r + 1]
            entry[r] = [om[0] for om in opt_masks[r] if om] + (
                entry[r + 1] if empty[r] else []
            )
        succ = [entry[0]] + [[] for _ in range(M)]
        end_ok = np.zeros(M + 1, bool)
        end_ok[0] = tail[0]
        for r in range(R):
            for om in opt_masks[r]:
                for k, j in enumerate(om):
                    if k < len(om) - 1:
                        succ[j + 1] = [om[k + 1]]
                    else:
                        succ[j + 1] = entry[r + 1]
                        end_ok[j + 1] = tail[r + 1]
        pairs = [(j, j2) for j in range(M) for j2 in succ[j + 1]]
        self.masks, self.pix, self.end_ok = masks, pix, end_ok
        # taking another hypothesis's cut of a region is a claim that the base got it wrong
        self.alt_first = np.zeros(M, bool)
        for om in opt_masks:
            for k, idx in enumerate(om):
                if k > 0 and idx:
                    self.alt_first[idx[0]] = True
        self.base = [j for om in opt_masks for j in om[0]]
        self.sptr = np.cumsum([0] + [len(x) for x in succ]).astype(np.int64)
        self.sidx = np.array([j for x in succ for j in x], np.int64)
        self.pj = np.array([a for a, _ in pairs], np.int64)
        self.pj2 = np.array([b for _, b in pairs], np.int64)
        self.pptr = np.searchsorted(self.pj, np.arange(M + 1)).astype(np.int64)
        # the base mask holding most of each candidate (itself for a base mask)
        within = [
            c
            if c < nb
            else (
                int(np.argmax(cover[c][1:]))
                if nb and cover[c][1:].max() >= 0.5 * len(pix[c])
                else -1
            )
            for c in range(len(info))
        ]
        cid = [m[2] for m in masks]
        self.no_div = np.array(
            [
                cid[a] >= nb
                and cid[b] >= nb
                and within[cid[a]] >= 0
                and within[cid[a]] == within[cid[b]]
                for a, b in pairs
            ],
            bool,
        )


def hazards(length, grown, trunc):
    mu, sd = P["div_mu"], P["div_sd"]
    a, b = ndtr((length - mu) / sd), ndtr((grown - mu) / sd)
    p = np.clip((b - a) / np.maximum(1 - a, 1e-9), 1e-6, 0.95)
    return -np.log(p), np.where(trunc, 0.0, -np.log1p(-p))


def predict(h):
    """Predicted lengths, and tops laid out from the top cell through the persisting gaps."""
    length = np.asarray(h.length, float)
    grown = length * np.exp(P["growth"])
    top = np.asarray(h.top, float)
    n = len(top)
    ptop = np.zeros(n)
    if n:
        ptop[0] = top[0]
        for i in range(1, n):
            gap = top[i] - (top[i - 1] + length[i - 1])
            ptop[i] = ptop[i - 1] + grown[i - 1] + gap
    return ptop, grown


def size_costs(h, frame, grown, t, H):
    obs = frame.masks
    n, m = len(h.ids), len(obs)
    otop = np.array([o[0] for o in obs], float)
    obot = np.array([o[1] for o in obs], float)
    otr = np.array([o[3] for o in obs], bool)
    trunc = np.asarray(h.trunc, bool)
    lim = np.log(P["max_length_change"])
    sl = np.sqrt(2.0) * P["sigma_end"]

    def fit(L, pred, cut, free):
        x = np.log(L[None, :] / pred[:, None])
        r = L[None, :] - pred[:, None]  # px: both ends' noise, whatever the length
        c = 0.5 * (r / sl) ** 2
        c = np.where(
            cut[None, :] & (r < 0), 0.0, c
        )  # cut off by the crop: only an upper bound
        ok = (x <= lim) & ((x >= -lim) | cut[None, :])
        if P["trunc_bound"]:
            # a cell cut off at the bottom (free): its length is unknown but no shorter than predicted,
            # so a whole mask may be longer than the prediction, not shorter
            cf = np.where(r >= 0, 0.0, c)
            okf = (x >= -lim) | cut[None, :]
            return np.where(free[:, None], np.where(okf, cf, INF), np.where(ok, c, INF))
        return np.where(ok | free[:, None], np.where(free[:, None], 0.0, c), INF)

    olen = obot - otop + 1
    sm = fit(olen, grown, otr, trunc) if n and m else np.full((n, m), INF)
    pj, pj2 = frame.pj, frame.pj2
    npairs = len(pj)
    sd = np.full((n, max(npairs, 1)), INF)
    so = np.full((n, max(npairs, 1)), INF)
    if n and npairs:
        span = obot[pj2] - otop[pj] + 1
        whole = fit(span, grown, otr[pj2], trunc)
        so[:, :npairs] = whole
        d1 = np.log(olen[None, pj] / (grown[:, None] / 2))
        d2 = np.log(olen[None, pj2] / (grown[:, None] / 2))
        d2 = np.where(otr[None, pj2] & (d2 < 0), 0.0, d2)
        sd[:, :npairs] = (
            whole
            + 0.5 * (d1 / P["sigma_daughter"]) ** 2
            + 0.5 * (d2 / P["sigma_daughter"]) ** 2
        )
    su = np.full((max(n - 1, 1), m), INF)
    underc = np.zeros(max(n - 1, 1))
    if n >= 2 and m:
        su[: n - 1] = fit(olen, grown[:-1] + grown[1:], otr, trunc[:-1] | trunc[1:])
        sisters = np.array(
            [
                h.parent[i] >= 0
                and h.parent[i] == h.parent[i + 1]
                and t - h.born[i] <= P["sisters_frames"]
                for i in range(n - 1)
            ]
        )
        underc[: n - 1] = -np.log(np.where(sisters, P["p_under_sisters"], P["p_under"]))
    # a division with only the upper daughter in view: mask j is about half the mother, and the
    # lower daughter (the rest of the mother below it) would be cut off so far that no mask of it is
    # kept: in view by less than the shortest mask kept, or reaching the rows cleared at the bottom
    sd1 = np.full((n, max(m, 1)), INF)
    if P["div_one"] and n and m:
        half = grown[:, None] / 2
        d1 = np.log(olen[None, :] / half)
        lower_top = obot[None, :] + 1
        lower_bot = otop[None, :] + grown[:, None] - 1
        shortest = (P["div_mu"] - 3 * P["div_sd"]) / 2
        cleared = P["clear_bottom_px"] > 0
        unseen = (H - lower_top < shortest) | (cleared & (lower_bot >= H))
        ok = (
            unseen
            & (lower_bot >= H)
            & ~otr[None, :]
            & (np.abs(d1) <= np.log(P["max_length_change"]) + 3 * P["sigma_daughter"])
        )
        sd1[:, :m] = np.where(ok, 0.5 * (d1 / P["sigma_daughter"]) ** 2, INF)
    return sm, sd, so, su, underc, otop, obot, sd1


def costs(h, frame, H, t):
    ptop, grown = predict(h)
    sm, sd, so, su, underc, otop, obot, sd1 = size_costs(h, frame, grown, t, H)
    altc = np.where(frame.alt_first, -np.log(P["p_alt"]), 0.0)
    sm = sm + altc[None, :]
    if sd1.shape[1] == len(altc):
        sd1 = sd1 + altc[None, :]
    su = su + altc[None, :]
    if len(frame.pj):
        sd = sd.copy()
        so = so.copy()
        pairc = altc[frame.pj] + altc[frame.pj2]
        sd[:, : len(pairc)] += pairc[None, :]
        so[:, : len(pairc)] += pairc[None, :]
        sd[:, : len(pairc)][:, frame.no_div] = INF
    divc, nodiv = (
        hazards(np.asarray(h.length, float), grown, np.asarray(h.trunc, bool))
        if h.ids
        else (np.zeros(0), np.zeros(0))
    )
    missc = (
        np.where(np.asarray(h.unseen) < P["max_miss"], -np.log(P["p_miss"]), INF)
        if h.ids
        else np.zeros(0)
    )
    overc = np.full(len(grown), -np.log(P["p_over"]))
    unseen = np.asarray(h.unseen, float) if h.ids else np.zeros(0)
    maxr = np.where(
        (unseen > 0) & P["no_revival"],
        4 * P["sigma_gap"] + P["revival_px"] * unseen,
        INF,
    )
    args = (
        sm,
        sd,
        so,
        su,
        nodiv,
        divc,
        underc,
        overc,
        missc,
        -np.log(P["p_lysis"]),
        ptop,
        grown,
        otop,
        obot,
        float(H),
        P["sigma_gap"],
        P["p_jump"],
        P["p_slide"],
        P["jump_scale_px"],
        float(P["exit_visible_px"]),
        frame.sptr,
        frame.sidx,
        frame.pptr,
        frame.pj2,
        frame.end_ok,
        sd1,
        maxr,
        float(P["appear_cost"]),
    )
    return args, ptop, grown


def apply(h, rows, frame, ptop, grown, t, cost, new_id, args=None):
    obs = frame.masks
    ids, top, length, parent, born, unseen, trunc = [], [], [], [], [], [], []
    entries, divisions, lone = [], [], []
    events = {
        "div": 0,
        "div_one": 0,
        "over": 0,
        "under": 0,
        "miss": 0,
        "lysis": 0,
        "exit": 0,
        "appear": 0,
        "jump": 0,
        "slide": 0,
    }
    lens, gaps, divlen, growth, jumps = [], [], [], [], []
    ref_bot, ref_cell = None, None  # mask bottom and cell of the last cell with a mask

    def keep(cid, tp, L, par, b, u, tr):
        ids.append(cid)
        top.append(float(tp))
        length.append(max(float(L), 1.0))
        parent.append(par)
        born.append(b)
        unseen.append(u)
        trunc.append(tr)

    def where(i):
        if ref_cell is None:
            return ptop[i]
        return ref_bot + 1 + (ptop[i] - (ptop[ref_cell] + grown[ref_cell]))

    def spacing(i, otop_):
        r = otop_ - where(i)
        sig = P["sigma_gap"]
        if abs(r) > 4 * sig:
            events["slide" if ref_cell is None else "jump"] += 1
            jumps.append(abs(r))
        else:
            gaps.append(r)

    for code, i, j, j2 in rows:
        code, i, j, j2 = int(code), int(i), int(j), int(j2)
        if code == 0:
            o = obs[j]
            L = o[1] - o[0] + 1
            spacing(i, o[0])
            keep(
                h.ids[i],
                o[0],
                max(L, grown[i]) if o[3] else L,
                h.parent[i],
                h.born[i],
                0,
                o[3],
            )
            entries.append((h.ids[i], "mask", (o[2],)))
            if not o[3] and not h.trunc[i]:
                lens.append(L - grown[i])
                growth.append(np.log(L / h.length[i]))
            ref_bot, ref_cell = o[1], i
        elif code == 1:
            spacing(i, obs[j][0])
            for oo in (obs[j], obs[j2]):
                c = new_id()
                L = oo[1] - oo[0] + 1
                keep(
                    c,
                    oo[0],
                    max(L, grown[i] / 2) if oo[3] else L,
                    h.ids[i],
                    t,
                    0,
                    oo[3],
                )
                entries.append((c, "mask", (oo[2],)))
                divisions.append((h.ids[i], c))
            events["div"] += 1
            if not h.trunc[i] and not obs[j2][3]:
                divlen.append(grown[i])
            ref_bot, ref_cell = obs[j2][1], i
        elif code == 2:
            a, b = obs[j], obs[j2]
            spacing(i, a[0])
            L = b[1] - a[0] + 1
            keep(
                h.ids[i],
                a[0],
                max(L, grown[i]) if b[3] else L,
                h.parent[i],
                h.born[i],
                0,
                b[3],
            )
            entries.append((h.ids[i], "union", (a[2], b[2])))
            events["over"] += 1
            ref_bot, ref_cell = b[1], i
        elif code == 3:
            o = obs[j]
            spacing(i, o[0])
            frac = grown[i] / (grown[i] + grown[i + 1])
            cut = round(o[0] + frac * (o[1] - o[0] + 1))
            keep(h.ids[i], o[0], cut - o[0], h.parent[i], h.born[i], 0, False)
            entries.append((h.ids[i], "upper", (o[2], cut)))
            L2 = o[1] - cut + 1
            keep(
                h.ids[i + 1],
                cut,
                max(L2, grown[i + 1]) if o[3] else L2,
                h.parent[i + 1],
                h.born[i + 1],
                0,
                o[3],
            )
            entries.append((h.ids[i + 1], "lower", (o[2], cut)))
            events["under"] += 1
            ref_bot, ref_cell = o[1], i + 1
        elif code == 4:
            keep(
                h.ids[i],
                where(i),
                grown[i],
                h.parent[i],
                h.born[i],
                h.unseen[i] + 1,
                h.trunc[i],
            )
            events["miss"] += 1
        elif code == 5:
            events["lysis"] += 1
        elif code == 6:
            events["exit"] += 1
        elif code == 7:
            o = obs[j]
            c = new_id()
            keep(c, o[0], o[1] - o[0] + 1, -1, t, 0, o[3])
            entries.append((c, "mask", (o[2],)))
            events["appear"] += 1
        elif code == 8:
            o = obs[j]
            spacing(i, o[0])
            c = new_id()
            keep(c, o[0], o[1] - o[0] + 1, h.ids[i], t, 0, False)
            entries.append((c, "mask", (o[2],)))
            divisions.append((h.ids[i], c))
            lone.append(c)
            events["div_one"] += 1
            # the cells below are placed after the lower daughter, past the mask
            ref_bot, ref_cell = o[0] + grown[i] - 1, i
    # keep the chain in order (an appeared mask is placed by its position)
    order = np.argsort(top, kind="stable")
    cols = [ids, top, length, parent, born, unseen, trunc]
    ids, top, length, parent, born, unseen, trunc = (
        [c[k] for k in order] for c in cols
    )
    rec = (
        h.rec,
        t,
        entries,
        divisions,
        events,
        lens,
        gaps,
        divlen,
        growth + [("jump", j) for j in jumps],
        (np.array(rows), ptop, grown, [o[:2] for o in obs], list(h.length), args, cost)
        if P.get("debug")
        else None,
        lone,
    )
    return Hyp(ids, top, length, parent, born, unseen, trunc, h.score + cost, rec)


DEBUG_ARGS = {}
ALT = {1: 2, 2: 1, 4: 5}  # division <-> cut into two masks, no mask <-> lysed


def step(hyps, frame, H, t, new_id):
    children = []
    for h in hyps:
        args, ptop, grown = costs(h, frame, H, t)
        if P.get("debug"):
            DEBUG_ARGS["args"] = args
        force = np.full(len(h.ids), -1, np.int64)
        cost, rows = align(*args, force)
        if cost >= INF / 2:
            continue
        children.append(apply(h, rows, frame, ptop, grown, t, cost, new_id, args))
        for row in rows:
            alt = ALT.get(int(row[0]))
            if alt is None:
                continue
            force = np.full(len(h.ids), -1, np.int64)
            force[int(row[1])] = alt
            c2, r2 = align(*args, force)
            if c2 < INF / 2:
                children.append(apply(h, r2, frame, ptop, grown, t, c2, new_id, args))
        missing = [int(row[1]) for row in rows if int(row[0]) == 4]
        if len(missing) >= 2:  # a lysis wave: all of them at once
            force = np.full(len(h.ids), -1, np.int64)
            force[missing] = 5
            c2, r2 = align(*args, force)
            if c2 < INF / 2:
                children.append(apply(h, r2, frame, ptop, grown, t, c2, new_id, args))
    children.sort(key=lambda c: c.score)
    out, seen = [], set()
    for c in children:
        key = (tuple(c.ids), tuple(round(x) for x in c.top))
        if key not in seen:
            seen.add(key)
            out.append(c)
            if len(out) >= P["beam"]:
                break
    return out or hyps


def visible_end(lab):
    """Where this segmentation's masks stop at the open end: the median lowest mask row over the
    frames whose lowest mask is within a cell length of the crop's bottom. That is the crop's
    bottom for the mCherry masks; PC masks stop a few px short of the trench opening, and a mask
    ending there is a cell cut off by the edge of what is seen, not a short cell.
    """
    H = lab.shape[1]
    low = np.array([np.flatnonzero(f.any(1)).max() for f in lab if f.any()])
    low = low[low >= H - P["div_mu"]]
    return int(np.median(low)) + 1 if len(low) else H


@numba.njit(cache=True)
def _clean_frames(lab, cut, edge, shortest):
    """clean()'s work, frame by frame: each mask's top and bottom rows, its area and its area
    from row `cut` down, then the masks to drop set to 0, in place.
    """
    T, H, W = lab.shape
    n = int(lab.max()) + 1
    top = np.zeros(n, np.int64)
    bottom = np.zeros(n, np.int64)
    area = np.zeros(n, np.int64)
    below = np.zeros(n, np.int64)
    drop = np.zeros(n, np.bool_)
    seen = np.zeros(n, np.int64)
    for t in range(T):
        k = 0
        for y in range(H):
            for x in range(W):
                m = lab[t, y, x]
                if m == 0:
                    continue
                if area[m] == 0:
                    top[m] = y
                    seen[k] = m
                    k += 1
                bottom[m] = y
                area[m] += 1
                if y >= cut:
                    below[m] += 1
        any_drop = False
        for i in range(k):
            m = seen[i]
            if (
                (bottom[m] - top[m] + 1) < shortest
                or 2 * below[m] > area[m]
                or bottom[m] >= edge
            ):
                drop[m] = True
                any_drop = True
        if any_drop:
            for y in range(H):
                for x in range(W):
                    if drop[lab[t, y, x]]:
                        lab[t, y, x] = 0
        for i in range(k):
            m = seen[i]
            area[m] = 0
            below[m] = 0
            drop[m] = False


def clean(lab):
    """The masks the tracker sees: whole masks only, without those that have more than half
    their area in the bottom crop_bottom_px rows, without those reaching the bottom clear_bottom_px
    rows, and without those shorter than the smallest plausible newborn, (div_mu - 3 div_sd) / 2.
    Cleans lab in place (copied first if it is read only) and returns it.
    """
    lab = np.asarray(lab)
    if not lab.flags.writeable:
        lab = lab.copy()
    cut = lab.shape[1] - P["crop_bottom_px"]
    edge = lab.shape[1] - P["clear_bottom_px"] if P["clear_bottom_px"] else lab.shape[1]
    shortest = (P["div_mu"] - 3 * P["div_sd"]) / 2
    if lab.size:
        _clean_frames(lab, cut, edge, float(shortest))
    return lab


def merge_pair(a, b, W):
    """One cell cut into two masks: their union, with the gap between them closed."""
    rows = np.r_[a, b] // W
    top, bot = rows.min(), rows.max()
    crop = np.zeros((bot - top + 1, W), bool)
    crop.ravel()[np.r_[a, b] - top * W] = True
    g = P["max_gap"]
    closed = ndi.binary_closing(crop, structure=np.ones((2 * g + 1, 3))) | crop
    return np.flatnonzero(closed.ravel()) + top * W


class Compressed:
    """A label stack kept compressed frame by frame (zstd) and read a frame at a time, as
    stack[t]. Masks are mostly background, so this holds a hypothesis in about 1% of its plain
    size. Frames read from it are read only.
    """

    def __init__(self, lab):
        from numcodecs import Zstd

        self._codec = Zstd(1)
        self.shape, self.dtype = lab.shape, lab.dtype
        self._frames = [self._codec.encode(np.ascontiguousarray(f)) for f in lab]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, t):
        frame = np.frombuffer(self._codec.decode(self._frames[t]), self.dtype)
        return frame.reshape(self.shape[1:])


def track(labs):
    """labs: cleaned label stacks, the base hypothesis first (a plain array), then any
    alternatives (anything that gives a frame as stack[t], such as Compressed).
    """
    lab = labs[0]
    T, _, W = lab.shape
    H = visible_end(lab)
    allc = np.nonzero(lab)[2]
    centre = float(np.median(allc)) if len(allc) else W / 2
    obs, side, side_labels = split_side(lab, centre, H, labs[1:])
    frames = [
        Frame(obs[t], lab[t], [l[t] for l in labs[1:]], side_labels[t], centre, H)
        for t in range(T)
    ]
    new_id = Counter()
    first = next((t for t in range(T) if obs[t]), None)
    if first is None:
        return [], side, frames
    o0 = [frames[first].masks[j] for j in frames[first].base]
    ids = [new_id() for _ in o0]
    rec = (
        None,
        first,
        [(c, "mask", (o[2],)) for c, o in zip(ids, o0)],
        [],
        {},
        [],
        [],
        [],
        [],
        None,
        [],
    )
    hyps = [
        Hyp(
            ids,
            [float(o[0]) for o in o0],
            [float(o[1] - o[0] + 1) for o in o0],
            [-1] * len(o0),
            [first] * len(o0),
            [0] * len(o0),
            [o[3] for o in o0],
            0.0,
            rec,
        )
    ]
    for t in range(first + 1, T):
        hyps = step(hyps, frames[t], H, t, new_id)
    best = min(hyps, key=lambda x: x.score)
    recs, r = [], best.rec
    while r is not None:
        recs.append(r)
        r = r[0]
    return recs[::-1], side, frames


def stats_of(recs):
    tot = {
        "div": 0,
        "div_one": 0,
        "over": 0,
        "under": 0,
        "miss": 0,
        "lysis": 0,
        "exit": 0,
        "appear": 0,
        "jump": 0,
        "slide": 0,
    }
    lens, gaps, divlen, growth, jumps = [], [], [], [], []
    for r in recs[1:]:
        for k, v in r[4].items():
            tot[k] += v
        lens += r[5]
        gaps += r[6]
        divlen += r[7]
        growth += [g for g in r[8] if not isinstance(g, tuple)]
        jumps += [g[1] for g in r[8] if isinstance(g, tuple)]
    return tot, lens, gaps, divlen, growth, jumps


_POOL = None  # the worker processes, started at the first use
_WORKERS = (
    None  # how many trenches `run` and `follow` track at once, set at the first batch
)
_ROOT = None  # this process's handle on the store, for writing a trench's tracks


def workers_for(tasks, bytes_each):
    """How many trenches to track at once: --workers, or as many as the cores and the free memory
    allow when each needs about `bytes_each` (never more than there are trenches).
    """
    if WORKERS is not None:
        return max(1, min(int(WORKERS), tasks))
    try:
        with open("/proc/meminfo") as f:
            free = next(
                int(line.split()[1]) * 1024
                for line in f
                if line.startswith("MemAvailable")
            )
    except (OSError, StopIteration):
        return 1  # free memory unknown: one at a time unless --workers says otherwise
    return max(1, min(os.cpu_count() or 1, int(0.8 * free / bytes_each), tasks))


def in_parallel(function, tasks, workers):
    """function(task) for each task, in the tasks' order, `workers` at a time. Each worker is a
    fresh process that imports this file with the same command line, so it holds the same store,
    fit and parameters. The processes are started at the first call and serve every later one,
    so the first call's `workers` stands.

    A worker that dies (killed when the machine runs out of memory, say) raises BrokenProcessPool
    here and stops the others: a ProcessPoolExecutor notices, where multiprocessing.Pool would
    wait for that trench for ever.
    """
    global _POOL
    if workers <= 1 and _POOL is None:
        return map(function, tasks)
    if _POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")
        )
    return _POOL.map(function, tasks)


def fit_track(task):
    """One trench's statistics for a round of the fit, tracked with that round's parameters."""
    lab, parameters = task
    P.update(parameters)
    return stats_of(track([lab])[0])


def fit(trenches, rounds=3):
    """Measure the fitted parameters on these trenches' own tracks (the base hypothesis, tracked
    with the current values, `rounds` times over), and record them in the store as its fit FIT.
    """
    from pymmm.store.experiment_group import segmentation, update_experiment

    root = open_experiment(STORE)
    # runs made before fits were recorded used the fit named after their segmentation
    used_by = [
        name
        for name, rec in records(root, "tracks").items()
        if (rec.get("fit") or {}).get("name", rec.get("segmentation")) == FIT
    ]
    if FIT_RECORD is not None and used_by:
        sys.exit(
            f"chain.py: the fit {FIT!r} of {STORE.name} was used by the run(s) {', '.join(used_by)}, so it stays as it is: "
            "give this fit another name with --fit"
        )
    started = (
        f"the store's fit of {FIT_RECORD['created']}"
        if FIT_RECORD
        else "the recipe's fit_start"
    )
    print(
        f"fit {FIT} of {STORE.name} ({PX:.4f} µm per pixel, {DT:.4f} min per frame), starting from {started}",
        flush=True,
    )
    used, labs = [], []
    for t in trenches:
        lab = np.asarray(segmentation(root, t, SEG)[1][0][...])
        if lab.any():
            used.append(int(t))
            labs.append(clean(lab))

    def mad(v):
        return 1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v))))

    # allow each trench 4 times its masks' size (a guess: the fit reads one hypothesis, and its
    # memory was not measured)
    workers = workers_for(len(labs), 4 * max((lab.nbytes for lab in labs), default=1))
    for r in range(rounds):
        tot = {
            "div": 0,
            "div_one": 0,
            "over": 0,
            "under": 0,
            "miss": 0,
            "lysis": 0,
            "exit": 0,
            "appear": 0,
            "jump": 0,
            "slide": 0,
        }
        lens, gaps, divlen, growth, jumps = [], [], [], [], []
        start = time.perf_counter()
        # the trenches' statistics are gathered in the trenches' order, however many are tracked at once
        for a, l, g, d, gr, ju in in_parallel(
            fit_track, [(lab, dict(P)) for lab in labs], workers
        ):
            for k in tot:
                tot[k] += a[k]
            lens += l
            gaps += g
            divlen += d
            growth += gr
            jumps += ju
        if len(growth) > 100:
            P["growth"] = float(np.median(growth))
        if len(lens) > 100:
            P["sigma_end"] = max(mad(lens) / np.sqrt(2.0), 0.5)
        if len(divlen) > 20:
            P["div_mu"] = float(np.median(divlen))
            P["div_sd"] = max(mad(divlen), 0.05 * P["div_mu"])
        if len(jumps) > 20:
            P["jump_scale_px"] = float(np.mean(jumps))
        print(
            f"round {r + 1}: {len(labs)} trenches, {time.perf_counter() - start:.0f} s; events {tot}; "
            f"growth {P['growth']:.4f}, end noise {P['sigma_end']:.2f} px, gap noise {P['sigma_gap']:.2f} px (the recipe's), "
            f"division {P['div_mu']:.1f} ± {P['div_sd']:.1f} px, jump scale {P['jump_scale_px']:.1f} px",
            flush=True,
        )
    # a fresh handle, so records written to the store while this ran are kept
    update_experiment(
        open_experiment(STORE, mode="r+"),
        "fits",
        FIT,
        {
            "tracker": TRACKER,
            "segmentation": SEG,
            "values": fitted(),
            "trenches": used,
            "frames": "all",
            "rounds": rounds,
            "started_from": started,
            "recipe": RECIPE,
            "fit_start": FIT_START,
            "pixel_size_um": PX,
            "frame_interval_min": DT,
            "made_by": "external/chaintrack/chain.py fit",
            "chaintrack": code_version(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(
        f"fit recorded in {STORE.name}, attrs.pymmm.fits[{FIT!r}]: "
        + ", ".join(f"{k} {v:.4g}" for k, v in fitted().items()),
        flush=True,
    )


def write(root, tid, lab, recs, side, frames):
    from pymmm.store.geff import tracklets_and_lineages, write_tracks

    _T, _H, W = lab.shape
    seg = np.zeros(lab.shape, np.uint16)
    nodes, nlab, divisions, lone = [], {}, [], set()

    def add(t, cell, pix):
        if len(pix) == 0:
            return
        nlab[t] = nlab.get(t, 0) + 1
        seg[t].ravel()[pix] = nlab[t]
        nodes.append(
            (t, cell, float((pix // W).mean()), float((pix % W).mean()), nlab[t])
        )

    for r in recs:
        _, t, entries, divs = r[:4]
        cand = frames[t].pix
        for cell, kind, data in entries:
            if kind == "mask":
                pix = cand[data[0]]
            elif kind == "union":
                pix = merge_pair(cand[data[0]], cand[data[1]], W)
            else:
                p = cand[data[0]]
                pix = (
                    p[(p // W) < data[1]] if kind == "upper" else p[(p // W) >= data[1]]
                )
            add(t, cell, pix)
        divisions += divs
        lone |= set(r[10])
    for sid, seq in side.items() if P["write_side"] else ():
        for t, l in seq:
            add(t, ("side", sid), np.flatnonzero(lab[t].ravel() == l))
    by_cell = {}
    for k, (t, cell, *_r) in enumerate(nodes):
        by_cell.setdefault(cell, []).append((t, k))
    edges = []
    for seq in by_cell.values():
        seq.sort()
        edges += [(a[1], b[1]) for a, b in itertools.pairwise(seq)]
    for mother, daughter in divisions:
        ms, ds = by_cell.get(mother), by_cell.get(daughter)
        if ms and ds:
            edges.append((max(ms)[1], min(ds)[1]))
    e = np.array(edges, np.int64).reshape(-1, 2)
    tt = np.array([n[0] for n in nodes], np.int32)
    track_id, lineage_id = tracklets_and_lineages(len(nodes), tt, e)
    props = {
        "t": tt,
        "y": np.array([n[2] for n in nodes], np.float32),
        "x": np.array([n[3] for n in nodes], np.float32),
        "seg_id": np.array([n[4] for n in nodes], np.int32),
        "track_id": track_id,
        "lineage_id": lineage_id,
    }
    # daughters whose sister went past the bottom unseen: their first nodes (GEFF ids: position + 1);
    # the mother's link to such a daughter is a division, though the graph shows a single child
    first = {cell: min(seq)[1] for cell, seq in by_cell.items()}
    div_one = sorted(int(first[c]) + 1 for c in lone if c in first)
    write_tracks(
        root,
        tid,
        OUT,
        props,
        e,
        np.full(len(e), np.nan, np.float32),
        seg,
        extra={
            "tracker": TRACKER,
            "segmentation": SEG,
            "params": json.dumps(P, default=str),
            "fit": FIT,
            "chaintrack_commit": code_version()["commit"],
            "channels": [str(c) for c in root["C"][:]],
            "div_one": div_one,
        },
    )


def alternatives(names):
    """The hypotheses (indices into names; names[0] is the base) used as alternatives."""
    if BASE_ONLY:
        return []
    if ALTS is not None:
        return [names.index(n) for n in ALTS if n != names[0]]
    return list(range(1, len(names)))


def record_run(root):
    """Note this run in the experiment group: its segmentation, the fit it used, that fit's recipe
    (µm and minutes), and the parameters as the tracker used them (pixels and frames).

    A run already in the store is continued only with the same fit (and so the same recipe), the
    same segmentation and the same alternatives, so that one run is never tracked two ways; its
    record is kept and the continuation noted in it.
    """
    from pymmm.store.experiment_group import update_experiment

    fit = {
        "name": FIT,
        "values": {k: FIT_RECORD["values"][k] for k in FIT_KEYS},
        "created": FIT_RECORD.get("created"),
    }
    alts = "none" if BASE_ONLY else (ALTS or "all")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    old = records(root, "tracks").get(OUT)
    if old is not None:
        old_fit = old.get("fit") or {}
        same = {
            "fit": old_fit.get("name") == FIT
            and old_fit.get("created") == fit["created"],
            "segmentation": old.get("segmentation") == SEG,
            "alternatives": old.get("alternatives") == alts,
        }
        if not all(same.values()):
            sys.exit(
                f"chain.py: the run {OUT!r} of {STORE.name} was made with another "
                f"{', '.join(k for k, v in same.items() if not v)}: give this run another name with --out"
            )
        update_experiment(
            root,
            "tracks",
            OUT,
            {
                **old,
                "continued": [
                    *old.get("continued", []),
                    {"at": now, "chaintrack": code_version()},
                ],
            },
        )
        return
    update_experiment(
        root,
        "tracks",
        OUT,
        {
            "tracker": TRACKER,
            "segmentation": SEG,
            "params": json.loads(json.dumps(P, default=str)),
            "recipe": RECIPE,
            "fit": fit,
            "pixel_size_um": PX,
            "frame_interval_min": DT,
            "alternatives": alts,
            "made_by": "external/chaintrack/chain.py",
            "chaintrack": code_version(),
            "created": now,
        },
    )


def track_trench(tid):
    """Track one trench and write its tracks into the store: the line to print, and the error if
    it failed. A trench writes only inside its own folder, so several can be written at once.
    """
    from pymmm.store.experiment_group import segmentation

    global _ROOT
    try:
        if _ROOT is None:
            _ROOT = open_experiment(STORE, mode="r+")
        names, labels = segmentation(_ROOT, tid, SEG)
        use = [0, *alternatives(names)]
        # the base stays a plain array: it is relabelled in place and written. The alternatives
        # are read a frame at a time, so they are kept compressed.
        labs = [clean(np.asarray(labels[use[0]][...]))]
        labs += [Compressed(clean(np.asarray(labels[h][...]))) for h in use[1:]]
        start = time.perf_counter()
        recs, side, frames = track(labs)
        seconds = time.perf_counter() - start
        write(_ROOT, tid, labs[0], recs, side, frames)
        return (
            f"trench {tid}: {len(use)} hypotheses, {seconds:.0f} s; events {stats_of(recs)[0]}",
            None,
        )
    except Exception:
        import traceback

        return f"trench {tid} failed", traceback.format_exc()


def track_all(trenches, root, of=None):
    """Track these trenches several at a time, printing each one's line in the trenches' order;
    the trenches that failed. `of`: how many trenches the whole job has, when these are only the
    first of them (`follow`), so that enough workers are started for the rest.
    """
    from pymmm.store.experiment_group import segmentation

    global _WORKERS
    if not trenches:
        return []
    if _WORKERS is None:
        # A worker needs about 0.5 GB plus 6 times the size of one hypothesis's masks, however many
        # alternatives it reads, since those are kept compressed (measured: at most 2.0 GB for
        # 14 hypotheses of 1029 x 960 x 128 uint16, a worker tracking several trenches in turn)
        _names, labels = segmentation(root, trenches[0], SEG)
        each = 0.5e9 + 6 * labels[0].nbytes
        _WORKERS = workers_for(of or len(trenches), each)
        print(
            f"tracking {_WORKERS} trenches at a time (about {each / 1e9:.1f} GB each)",
            flush=True,
        )
    workers = _WORKERS
    failed = []
    for tid, (line, error) in zip(
        trenches, in_parallel(track_trench, trenches, workers)
    ):
        print(line if error is None else f"{line}:\n{error}", flush=True)
        if error is not None:
            failed.append(tid)
    return failed


def run(trenches):
    root = open_experiment(STORE, mode="r+")
    record_run(root)
    failed = track_all(trenches, root)
    print(f"tracks: {STORE}/trench_NNNN/tracks/{OUT}.geff", flush=True)
    if failed:
        sys.exit(
            f"chain.py: {len(failed)} of {len(trenches)} trenches failed: {' '.join(map(str, failed))}"
        )


def follow(trenches):
    """Track each trench as soon as its segmentation is finished (the Segmenter still running),
    until every one is tracked.
    """
    from pymmm.store.experiment_group import has_segmentation

    root = open_experiment(STORE, mode="r+")
    record_run(root)
    left = list(trenches)
    while left:
        ready = [t for t in left if has_segmentation(root, t, SEG)]
        track_all(ready, root, of=len(trenches))
        left = [t for t in left if t not in ready]
        if left and not ready:
            time.sleep(30)


def parse(args):
    out = []
    for a in args:
        if a.startswith("--"):
            sys.exit(f"chain.py: unknown option {a}")
        if "-" in a:
            lo, hi = map(int, a.split("-"))
            out += list(range(lo, hi + 1))
        else:
            out.append(int(a))
    return out


if __name__ == "__main__":
    args = sys.argv[2:]
    for flag, takes in (
        ("--store", 1),
        ("--recipe", 1),
        ("--seg", 1),
        ("--fit", 1),
        ("--out", 1),
        ("--alts", 1),
        ("--base-only", 0),
        ("--workers", 1),
    ):
        if flag in args:
            k = args.index(flag)
            args = args[:k] + args[k + 1 + takes :]
    from concurrent.futures.process import BrokenProcessPool

    try:
        {"fit": fit, "run": run, "follow": follow}[COMMAND](parse(args))
    except BrokenProcessPool:
        # the trenches being tracked at that moment were stopped part-way through writing
        left = (
            "No fit was recorded"
            if COMMAND == "fit"
            else "The trenches printed above are written; run again for the others"
        )
        sys.exit(
            "chain.py: a worker process died, most likely killed because the machine ran out of memory, so "
            f"this stopped. {left}. Give a smaller --workers if memory was short"
        )
    finally:
        if (
            _POOL is not None
        ):  # trenches not started yet are dropped; those being tracked finish first
            _POOL.shutdown(wait=True, cancel_futures=True)
