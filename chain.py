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
The masks (S1 only, shapes only, no image) are observations, cleaned first: a mask with more
than half its area in the bottom crop_bottom_px rows (the halo at the trench opening) is
removed whole, and so is any mask shorter than the smallest plausible newborn,
(div_mu - 3 div_sd) / 2; the masks kept are kept whole. What is left is: one mask per cell, one cell
cut into two masks, one mask over two cells. Each frame the predicted chain is aligned to
the masks in order; a cell is judged by its own length and by the gap between its mask
and the mask of the last cell above that has one, so errors never build up down the column.
A beam of lineage histories lets the frames after an event decide what it was.
Cells off the centre line (side cells, beside the column) are tracked first and kept out of
the chain: each follows the nearest similar mask, reaching the centre line only by drifting.
Where the base merges a side cell into a chain cell and an alternative segmentation draws them
apart, that cut is used (see split_side).

Fitted without labels: growth, length noise, gap noise, length at division, jump size.
Each segmentation has its own fit (params/<segmentation>.json).

The masks come from the experiment group PyMMM extracts (one OME-Zarr image per trench,
<experiment>.zarr; see pymmm.trench_images): a trench's segmentation <seg> is its label image
labels/<seg>.<hypothesis>, one (t, y, x) label image per hypothesis, the base first. `run` writes each trench's tracks
beside it, as the run <out> (default chain_S1): the GEFF tracks/<out>.geff and the masks it chose,
labels/tracks.<out>. A tracker never writes curation/ (mm-track makes the curated copy at the first
edit), so re-running replaces tracker output only. `follow` does the same for each trench as soon
as its segmentation is finished, while tools/candidates.py is still segmenting the others.

    cd /data/phage-signatures
    pixi run -e seg python external/chaintrack/chain.py fit 2-9 --seg pc_affinity_ensemble
    pixi run -e seg python external/chaintrack/chain.py run 0-9 --seg pc_affinity_ensemble
    pixi run -e seg python external/chaintrack/chain.py follow 0-492 --seg pc_affinity_ensemble --out chain_alt_S1
"""
import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/phage-signatures")
EXPERIMENT = "20261005_pbs_psc_yfp_ty013"
# the experiment group: one OME-Zarr image per trench (--group PATH for another, e.g. a scratch copy)
GROUP = Path(sys.argv[sys.argv.index("--group") + 1]) if "--group" in sys.argv else ROOT / f"{EXPERIMENT}.zarr"
# which segmentation's masks to track (--seg NAME, default the mCherry one); each has its own fit
SEG = sys.argv[sys.argv.index("--seg") + 1] if "--seg" in sys.argv else "mcherry"
PARAMS = HERE / "params" / f"{SEG}.json"
TRACKER = "chain"
# a segmentation with several hypotheses: the first is the base, the others its alternatives (which
# checkpoints those are is decided where they are made, tools/candidates.py); --alts e050,e150
# uses only those, --base-only the base alone
ALTS = sys.argv[sys.argv.index("--alts") + 1].split(",") if "--alts" in sys.argv else None
BASE_ONLY = "--base-only" in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{TRACKER}_S1"

P = dict(
    beam=40,
    # fitted without labels
    growth=0.029,              # log length per frame
    sigma_end=1.0,             # px, noise of each end of a mask (so a mask's length is noisy by about sqrt(2) of this)
    sigma_gap=2.0,             # px, noise of the gap between neighbours
    div_mu=57.6, div_sd=6.1,   # px, length at division
    # priors
    max_length_change=1.25,    # outside a division, a cell's masks fit its predicted length within this
    sigma_daughter=0.15,       # log size of each daughter relative to half the mother
    p_jump=0.005,              # the gap above a cell jumps this frame
    p_slide=0.005,             # the top cell (and so the column) jumps this frame
    p_over=0.01,               # one cell cut into two masks
    p_alt=0.01,                # the base segmentation is wrong at a spot, so another hypothesis's cut is right
    p_under=0.01,              # one mask over two cells
    p_under_sisters=0.2,       # ... sisters born in the last few frames
    sisters_frames=3,
    p_miss=0.01, max_miss=3,
    p_lysis=0.003,
    jump_scale_px=20.0,        # typical size of a jump (fitted)
    centre_px=6,               # masks further than this from the centre line are off the chain
    side_sigma_y=15.0, side_sigma_area=0.1, side_sigma_x=4.0,   # an off-centre cell between frames
    side_max_cost=16.0,        # ... anything worse is not the same cell (4 sigma)
    side_min_overlap=0.3,      # an off-centre cell reaches the centre line only by drifting: its mask there overlaps its last one by this much of the smaller
    side_min_piece_px=30,      # an alternative separates a side cell from a chain cell only into pieces at least this big
    # a side cell the base merged into a chain cell continues on an alternative's piece: off, as a
    # decision made before the chain it can shift the chain (trench 1 t40); it belongs in the beam
    side_alt_piece=False,
    side_alt_cut=True,         # a new off-centre mask that alternatives cut into a chain and a side piece is cut
    exit_visible_px=8,
    crop_bottom_px=30,         # masks mostly in these bottom rows (the halo at the trench opening) are removed
    max_gap=8,                 # px: an over-cut cell's two masks are joined across a gap up to this
)
FITTED = ("growth", "sigma_end", "sigma_gap", "div_mu", "div_sd", "jump_scale_px")
if PARAMS.is_file():
    P.update(json.loads(PARAMS.read_text()))

MOVE = dict(match=0, div=1, over=2, under=3, miss=4, lysis=5, exit=6, appear=7)
INF = 1e18
APPEAR_COST = 40.0  # no new cells: only when nothing else can explain a mask


@numba.njit(cache=True)
def mix(r, sigma, p, scale):
    """-log of: no jump (Gaussian noise), or a jump, small ones commoner than big ones."""
    g = (1.0 - p) * np.exp(-0.5 * (r / sigma) ** 2) / (sigma * 2.5066282746310002)
    return -np.log(g + p * np.exp(-abs(r) / scale) / (2.0 * scale))


@numba.njit(cache=True)
def align(sm, sd, so, su, nodiv, divc, underc, overc, missc, lysisc, ptop, plen, otop, obot,
          H, sig_gap, p_jump, p_slide, rng, vis, sptr, sidx, pptr, pj2, end_ok, force):
    """Order-preserving alignment of n predicted cells to a frame's masks. The masks form a graph
    (see Frame): state s = 0 before any mask, s = j + 1 once mask j is taken; sidx[sptr[s]:sptr[s + 1]]
    are the masks that can come next, pj2[pptr[j]:pptr[j + 1]] the masks that can follow mask j
    (for a division or a cell cut into two masks; sd and so are indexed by that position), and
    end_ok[s] says whether the frame's masks may end there. Search state (i cells, s, a = 1 + the
    last cell given a mask, 0 if none yet). Spacing is judged against the mask of that cell (or
    the closed end); size by the move's own terms."""
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
                        else:
                            sp = mix(otop[j] - pt, sig_gap, p_jump, rng)
                        # match
                        if f < 0 or f == 0:
                            c = c0 + sp + sm[i, j] + nodiv[i]
                            if c < dp[i + 1, j + 1, i + 1]:
                                dp[i + 1, j + 1, i + 1] = c; bc[i + 1, j + 1, i + 1] = 0; bi[i + 1, j + 1, i + 1] = i
                                bs[i + 1, j + 1, i + 1] = s; ba[i + 1, j + 1, i + 1] = a; bm[i + 1, j + 1, i + 1] = j
                        # divide / cut into two masks
                        for r in range(pptr[j], pptr[j + 1]):
                            j2 = pj2[r]
                            if f < 0 or f == 1:
                                c = c0 + sp + sd[i, r] + divc[i]
                                if c < dp[i + 1, j2 + 1, i + 1]:
                                    dp[i + 1, j2 + 1, i + 1] = c; bc[i + 1, j2 + 1, i + 1] = 1; bi[i + 1, j2 + 1, i + 1] = i
                                    bs[i + 1, j2 + 1, i + 1] = s; ba[i + 1, j2 + 1, i + 1] = a
                                    bm[i + 1, j2 + 1, i + 1] = j; bm2[i + 1, j2 + 1, i + 1] = j2
                            if f < 0 or f == 2:
                                c = c0 + sp + so[i, r] + overc[i] + nodiv[i]
                                if c < dp[i + 1, j2 + 1, i + 1]:
                                    dp[i + 1, j2 + 1, i + 1] = c; bc[i + 1, j2 + 1, i + 1] = 2; bi[i + 1, j2 + 1, i + 1] = i
                                    bs[i + 1, j2 + 1, i + 1] = s; ba[i + 1, j2 + 1, i + 1] = a
                                    bm[i + 1, j2 + 1, i + 1] = j; bm2[i + 1, j2 + 1, i + 1] = j2
                        # two cells in one mask
                        if i + 1 < n and (f < 0 or f == 3) and (force[i + 1] < 0 or force[i + 1] == 3):
                            c = c0 + sp + su[i, j] + underc[i] + nodiv[i] + nodiv[i + 1]
                            if c < dp[i + 2, j + 1, i + 2]:
                                dp[i + 2, j + 1, i + 2] = c; bc[i + 2, j + 1, i + 2] = 3; bi[i + 2, j + 1, i + 2] = i
                                bs[i + 2, j + 1, i + 2] = s; ba[i + 2, j + 1, i + 2] = a; bm[i + 2, j + 1, i + 2] = j
                    # no mask for cell i
                    if not gone:
                        if f < 0 or f == 4:
                            c = c0 + missc[i] + nodiv[i]
                            if c < dp[i + 1, s, a]:
                                dp[i + 1, s, a] = c; bc[i + 1, s, a] = 4; bi[i + 1, s, a] = i; bs[i + 1, s, a] = s; ba[i + 1, s, a] = a
                    near_bottom = ptop[i] + 2 * plen[i] >= H   # its own prediction, not one pushed by a jump above
                    if (not gone or not near_bottom) and (f < 0 or f == 5):
                        c = c0 + lysisc
                        if c < dp[i + 1, s, a]:
                            dp[i + 1, s, a] = c; bc[i + 1, s, a] = 5; bi[i + 1, s, a] = i; bs[i + 1, s, a] = s; ba[i + 1, s, a] = a
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
                                dp[i + 1, s, a] = c; bc[i + 1, s, a] = 6; bi[i + 1, s, a] = i; bs[i + 1, s, a] = s; ba[i + 1, s, a] = a
                for q in range(sptr[s], sptr[s + 1]):  # a mask no cell explains
                    j = sidx[q]
                    c = c0 + APPEAR_COST
                    if c < dp[i, j + 1, a]:
                        dp[i, j + 1, a] = c; bc[i, j + 1, a] = 7; bi[i, j + 1, a] = i; bs[i, j + 1, a] = s; ba[i, j + 1, a] = a
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
        rows[k, 0] = code; rows[k, 1] = -1 if code == 7 else bi[i, s, a]
        rows[k, 2] = bm[i, s, a]; rows[k, 3] = bm2[i, s, a]
        k += 1
        i, s, a = bi[i, s, a], bs[i, s, a], ba[i, s, a]
    return best, rows[:k][::-1]


class Hyp:
    __slots__ = ("ids", "top", "length", "parent", "born", "unseen", "trunc", "score", "rec")

    def __init__(self, ids, top, length, parent, born, unseen, trunc, score, rec):
        self.ids, self.top, self.length, self.parent = ids, top, length, parent
        self.born, self.unseen, self.trunc, self.score, self.rec = born, unseen, trunc, score, rec


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
        out.append((int(rows.min()), int(rows.max()), int(l), bool(rows.max() >= H - 2), float(rows.mean()),
                    len(pix), float(cols.mean())))
    return out


def side_cost(o, last):
    """How unlike the off-centre cell `last` the mask `o` is: it can move along the trench a fair
    way, but keeps its size and stays near its own sideways position."""
    return (((o[4] - last[4]) / P["side_sigma_y"]) ** 2 + (np.log(o[5] / last[5]) / P["side_sigma_area"]) ** 2
            + ((o[6] - last[6]) / P["side_sigma_x"]) ** 2)


def may_follow(o, frame, previous, centre):
    """An off-centre cell may continue on a mask on the centre line only by drifting (or turning)
    onto it, so that the mask overlaps its last one, not by jumping to a chain cell next to it.
    previous: (frame, label) of the cell's last mask."""
    if abs(o[6] - centre) > P["centre_px"]:
        return True
    a = np.flatnonzero(frame.ravel() == o[2])
    b = np.flatnonzero(previous[0].ravel() == previous[1])
    return np.intersect1d(a, b, assume_unique=True).size >= P["side_min_overlap"] * min(len(a), len(b))


def mask_stats(pix, W):
    rows, cols = pix // W, pix % W
    return int(rows.min()), int(rows.max()), float(rows.mean()), len(pix), float(cols.mean())


def alt_side_piece(frame, alt_frames, last, centre, claimed):
    """For an off-centre cell with no base mask of its own this frame: the piece of an unclaimed
    base mask that an alternative draws as a mask of its own, off-centre and like the cell (the
    base merged the cell into a chain cell). (host label, flat pixels of the piece) or None."""
    W = frame.shape[1]
    flat = frame.ravel()
    lo, hi = max(0, int(last[0]) - 40), int(last[1]) + 41
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
            if len(piece) < 0.5 * len(pix) or len(host_pix) - len(piece) < P["side_min_piece_px"] \
                    or len(piece) < P["side_min_piece_px"]:
                continue
            best = (c, host, piece)
    return None if best is None else best[1:]


def alt_centre_cut(frame, alt_frames, host, centre):
    """For a new off-centre base mask: the off-centre piece that alternatives cut out of it, leaving
    a piece on the centre line (the base merged a side cell into a chain cell), if at least two
    alternatives cut it alike (IoU >= 0.8). Flat pixels of the piece, or None."""
    W = frame.shape[1]
    host_pix = np.flatnonzero(frame.ravel() == host)
    found = []   # (piece, votes)
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
        if not (centred and len(off) == 1) or len(host_pix) - len(off[0]) < P["side_min_piece_px"]:
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
    cut: lab is relabelled in place there (the piece gets a new label, the rest keeps the mask's)."""
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
                claimed.add(o[2]); tracks[tid].append((t, o[2])); nxt.append((tid, o, 0))
            else:
                waiting.append((tid, last, missed))
        still = []
        for tid, last, missed in waiting:
            cut = alt_side_piece(frame, alt_t, last, centre, claimed) if alt_t and P["side_alt_piece"] else None
            if cut is not None:
                _, piece = cut
                new = int(frame.max()) + 1
                frame.ravel()[piece] = new
                masks = frame_masks(frame, H)
                o = next(m for m in masks if m[2] == new)
                claimed.add(new); tracks[tid].append((t, new)); nxt.append((tid, o, 0))
            else:
                still.append((tid, last, missed))
        for label in [o[2] for o in masks if o[2] not in claimed and abs(o[6] - centre) > P["centre_px"]]:
            piece = alt_centre_cut(frame, alt_t, label, centre) if alt_t and P["side_alt_cut"] else None
            if piece is not None:
                label = int(frame.max()) + 1
                frame.ravel()[piece] = label
                masks = frame_masks(frame, H)
            o = next(m for m in masks if m[2] == label)
            # a piece just cut out may be the side cell that lost its mask to the merge
            best = None
            if piece is not None:
                for k, (tid, last, missed) in enumerate(still):
                    c = side_cost(o, last)
                    if c <= P["side_max_cost"] and may_follow(o, frame, last_mask(tid), centre) and (best is None or c < best[0]):
                        best = (c, k)
            if best is not None:
                tid = still.pop(best[1])[0]
                tracks[tid].append((t, o[2])); claimed.add(o[2]); nxt.append((tid, o, 0))
            else:
                nid += 1
                tracks[-nid] = [(t, o[2])]; claimed.add(o[2]); nxt.append((-nid, o, 0))
        for tid, last, missed in still:
            if missed < P["max_miss"]:  # no mask this frame, like a chain cell: wait for it
                nxt.append((tid, last, missed + 1))
        active = nxt
        obs.append(sorted(o[:6] for o in masks if o[2] not in claimed))
        side_labels.append({o[2] for o in masks if o[2] in claimed})
    return obs, tracks, side_labels


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
    bounds = list(first) + [len(lab)]
    return [(int(l), nz[bounds[k]:bounds[k + 1]]) for k, l in enumerate(ids)]


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
    end_ok: states the frame may end at."""
    __slots__ = ("masks", "pix", "base", "sptr", "sidx", "pptr", "pj", "pj2", "end_ok", "alt_first", "no_div")

    def __init__(self, base_obs, base_frame, alt_frames, side_labels, centre, H):
        W = base_frame.shape[1]
        groups = dict(label_groups(base_frame))
        info, pix = [], []
        for o in base_obs:
            info.append((o[0], o[1], len(info), o[3], o[4], o[5])); pix.append(groups[o[2]])
        nb = len(info)
        idimg = np.zeros(base_frame.size, np.int64)
        for b in range(nb):
            idimg[pix[b]] = b + 1
        side = np.isin(base_frame.ravel(), list(side_labels)) if side_labels else None
        extras, cuts = [], []
        for fr in alt_frames:
            mine = []
            for _, q in label_groups(fr):
                cols = q % W
                if abs(cols.mean() - centre) > P["centre_px"] or (side is not None and side[q].mean() > 0.5):
                    continue
                same = -1
                cnt = np.bincount(idimg[q], minlength=nb + 1); cnt[0] = 0
                if sum(cnt[b + 1] >= 0.3 * len(pix[b]) for b in range(nb)) >= 2:
                    continue   # a merge of base masks: "one cell cut into two masks" covers it
                b = int(np.argmax(cnt)) - 1
                if b >= 0 and cnt[b + 1] / (len(q) + len(pix[b]) - cnt[b + 1]) >= 0.8:
                    same = b
                for e in extras if same < 0 else ():
                    inter = np.intersect1d(q, pix[e], assume_unique=True).size
                    if inter / (len(q) + len(pix[e]) - inter) >= 0.8:
                        same = e
                        break
                if same < 0:
                    rows = q // W
                    same = len(info)
                    info.append((int(rows.min()), int(rows.max()), same, bool(rows.max() >= H - 2),
                                 float(rows.mean()), len(q)))
                    pix.append(q); extras.append(same)
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
                inter = np.intersect1d(pix[e], pix[e2], assume_unique=True).size
                if inter >= 0.3 * min(len(pix[e]), len(pix[e2])):
                    root[find(e)] = find(e2)
        members = {}
        for c in range(len(info)):
            members.setdefault(find(c), []).append(c)
        key = lambda c: (info[c][0], info[c][1], c)
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
            entry[r] = [om[0] for om in opt_masks[r] if om] + (entry[r + 1] if empty[r] else [])
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
        within = [c if c < nb else (int(np.argmax(cover[c][1:])) if nb and cover[c][1:].max() >= 0.5 * len(pix[c]) else -1)
                  for c in range(len(info))]
        cid = [m[2] for m in masks]
        self.no_div = np.array([cid[a] >= nb and cid[b] >= nb and within[cid[a]] >= 0
                                and within[cid[a]] == within[cid[b]] for a, b in pairs], bool)


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


def size_costs(h, frame, grown, t):
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
        r = L[None, :] - pred[:, None]                           # px: both ends' noise, whatever the length
        c = 0.5 * (r / sl) ** 2
        c = np.where(cut[None, :] & (r < 0), 0.0, c)             # cut off by the crop: only an upper bound
        ok = (x <= lim) & ((x >= -lim) | cut[None, :])
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
        sd[:, :npairs] = whole + 0.5 * (d1 / P["sigma_daughter"]) ** 2 + 0.5 * (d2 / P["sigma_daughter"]) ** 2
    su = np.full((max(n - 1, 1), m), INF)
    underc = np.zeros(max(n - 1, 1))
    if n >= 2 and m:
        su[:n - 1] = fit(olen, grown[:-1] + grown[1:], otr, trunc[:-1] | trunc[1:])
        sisters = np.array([h.parent[i] >= 0 and h.parent[i] == h.parent[i + 1]
                            and t - h.born[i] <= P["sisters_frames"] for i in range(n - 1)])
        underc[:n - 1] = -np.log(np.where(sisters, P["p_under_sisters"], P["p_under"]))
    return sm, sd, so, su, underc, otop, obot


def costs(h, frame, H, t):
    ptop, grown = predict(h)
    sm, sd, so, su, underc, otop, obot = size_costs(h, frame, grown, t)
    altc = np.where(frame.alt_first, -np.log(P["p_alt"]), 0.0)
    sm = sm + altc[None, :]
    su = su + altc[None, :]
    if len(frame.pj):
        sd = sd.copy(); so = so.copy()
        pairc = altc[frame.pj] + altc[frame.pj2]
        sd[:, :len(pairc)] += pairc[None, :]
        so[:, :len(pairc)] += pairc[None, :]
        sd[:, :len(pairc)][:, frame.no_div] = INF
    divc, nodiv = hazards(np.asarray(h.length, float), grown, np.asarray(h.trunc, bool)) if h.ids else (np.zeros(0), np.zeros(0))
    missc = np.where(np.asarray(h.unseen) < P["max_miss"], -np.log(P["p_miss"]), INF) if h.ids else np.zeros(0)
    overc = np.full(len(grown), -np.log(P["p_over"]))
    args = (sm, sd, so, su, nodiv, divc, underc, overc, missc, -np.log(P["p_lysis"]),
            ptop, grown, otop, obot, float(H), P["sigma_gap"], P["p_jump"], P["p_slide"], P["jump_scale_px"],
            float(P["exit_visible_px"]), frame.sptr, frame.sidx, frame.pptr, frame.pj2, frame.end_ok)
    return args, ptop, grown


def apply(h, rows, frame, ptop, grown, t, cost, new_id, args=None):
    obs = frame.masks
    ids, top, length, parent, born, unseen, trunc = [], [], [], [], [], [], []
    entries, divisions = [], []
    events = dict(div=0, over=0, under=0, miss=0, lysis=0, exit=0, appear=0, jump=0, slide=0)
    lens, gaps, divlen, growth, jumps = [], [], [], [], []
    ref_bot, ref_cell = None, None  # mask bottom and cell of the last cell with a mask

    def keep(cid, tp, L, par, b, u, tr):
        ids.append(cid); top.append(float(tp)); length.append(max(float(L), 1.0)); parent.append(par)
        born.append(b); unseen.append(u); trunc.append(tr)

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
            keep(h.ids[i], o[0], max(L, grown[i]) if o[3] else L, h.parent[i], h.born[i], 0, o[3])
            entries.append((h.ids[i], "mask", (o[2],)))
            if not o[3] and not h.trunc[i]:
                lens.append(L - grown[i]); growth.append(np.log(L / h.length[i]))
            ref_bot, ref_cell = o[1], i
        elif code == 1:
            spacing(i, obs[j][0])
            for oo in (obs[j], obs[j2]):
                c = new_id()
                L = oo[1] - oo[0] + 1
                keep(c, oo[0], max(L, grown[i] / 2) if oo[3] else L, h.ids[i], t, 0, oo[3])
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
            keep(h.ids[i], a[0], max(L, grown[i]) if b[3] else L, h.parent[i], h.born[i], 0, b[3])
            entries.append((h.ids[i], "union", (a[2], b[2])))
            events["over"] += 1
            ref_bot, ref_cell = b[1], i
        elif code == 3:
            o = obs[j]
            spacing(i, o[0])
            frac = grown[i] / (grown[i] + grown[i + 1])
            cut = int(round(o[0] + frac * (o[1] - o[0] + 1)))
            keep(h.ids[i], o[0], cut - o[0], h.parent[i], h.born[i], 0, False)
            entries.append((h.ids[i], "upper", (o[2], cut)))
            L2 = o[1] - cut + 1
            keep(h.ids[i + 1], cut, max(L2, grown[i + 1]) if o[3] else L2, h.parent[i + 1], h.born[i + 1], 0, o[3])
            entries.append((h.ids[i + 1], "lower", (o[2], cut)))
            events["under"] += 1
            ref_bot, ref_cell = o[1], i + 1
        elif code == 4:
            keep(h.ids[i], where(i), grown[i], h.parent[i], h.born[i], h.unseen[i] + 1, h.trunc[i])
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
    # keep the chain in order (an appeared mask is placed by its position)
    order = np.argsort(top, kind="stable")
    cols = [ids, top, length, parent, born, unseen, trunc]
    ids, top, length, parent, born, unseen, trunc = ([c[k] for k in order] for c in cols)
    rec = (h.rec, t, entries, divisions, events, lens, gaps, divlen, growth + [("jump", j) for j in jumps],
           (np.array(rows), ptop, grown, [o[:2] for o in obs], list(h.length), args, cost) if P.get("debug") else None)
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
        key = (tuple(c.ids), tuple(int(round(x)) for x in c.top))
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
    ending there is a cell cut off by the edge of what is seen, not a short cell."""
    H = lab.shape[1]
    low = np.array([np.flatnonzero(f.any(1)).max() for f in lab if f.any()])
    low = low[low >= H - P["div_mu"]]
    return int(np.median(low)) + 1 if len(low) else H


def clean(lab):
    """The masks the tracker sees: whole masks only, without those that have more than half
    their area in the bottom crop_bottom_px rows, and without those shorter than the smallest
    plausible newborn, (div_mu - 3 div_sd) / 2."""
    lab = np.array(lab)
    cut = lab.shape[1] - P["crop_bottom_px"]
    shortest = (P["div_mu"] - 3 * P["div_sd"]) / 2
    for f in lab:
        ids = np.unique(f)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        rows, cols = np.nonzero(f)
        which = f[rows, cols]
        lo = np.asarray(ndi.minimum(rows, which, ids))
        hi = np.asarray(ndi.maximum(rows, which, ids))
        below = np.asarray(ndi.mean(rows >= cut, which, ids))
        drop = ids[((hi - lo + 1) < shortest) | (below > 0.5)]
        if len(drop):
            f[np.isin(f, drop)] = 0
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


def track(labs):
    """labs: cleaned label stacks, the base hypothesis first, then any alternatives."""
    lab = labs[0]
    T, _, W = lab.shape
    H = visible_end(lab)
    allc = np.nonzero(lab)[2]
    centre = float(np.median(allc)) if len(allc) else W / 2
    obs, side, side_labels = split_side(lab, centre, H, labs[1:])
    frames = [Frame(obs[t], lab[t], [l[t] for l in labs[1:]], side_labels[t], centre, H) for t in range(T)]
    new_id = Counter()
    first = next((t for t in range(T) if obs[t]), None)
    if first is None:
        return [], side, frames
    o0 = [frames[first].masks[j] for j in frames[first].base]
    ids = [new_id() for _ in o0]
    rec = (None, first, [(c, "mask", (o[2],)) for c, o in zip(ids, o0)], [], {}, [], [], [], [], None)
    hyps = [Hyp(ids, [float(o[0]) for o in o0], [float(o[1] - o[0] + 1) for o in o0], [-1] * len(o0),
                [first] * len(o0), [0] * len(o0), [o[3] for o in o0], 0.0, rec)]
    for t in range(first + 1, T):
        hyps = step(hyps, frames[t], H, t, new_id)
    best = min(hyps, key=lambda x: x.score)
    recs, r = [], best.rec
    while r is not None:
        recs.append(r)
        r = r[0]
    return recs[::-1], side, frames


def stats_of(recs):
    tot = dict(div=0, over=0, under=0, miss=0, lysis=0, exit=0, appear=0, jump=0, slide=0)
    lens, gaps, divlen, growth, jumps = [], [], [], [], []
    for r in recs[1:]:
        for k, v in r[4].items():
            tot[k] += v
        lens += r[5]; gaps += r[6]; divlen += r[7]
        growth += [g for g in r[8] if not isinstance(g, tuple)]
        jumps += [g[1] for g in r[8] if isinstance(g, tuple)]
    return tot, lens, gaps, divlen, growth, jumps


def fit(trenches, rounds=3):
    from pymmm.trench_images import open_experiment, segmentation

    root = open_experiment(GROUP)
    labs = [clean(l) for l in (np.asarray(segmentation(root, t, SEG)[1][0][...]) for t in trenches) if l.any()]
    mad = lambda v: 1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v))))
    for r in range(rounds):
        tot = dict(div=0, over=0, under=0, miss=0, lysis=0, exit=0, appear=0, jump=0, slide=0)
        lens, gaps, divlen, growth, jumps = [], [], [], [], []
        start = time.perf_counter()
        for lab in labs:
            a, l, g, d, gr, ju = stats_of(track([lab])[0])
            for k in tot:
                tot[k] += a[k]
            lens += l; gaps += g; divlen += d; growth += gr; jumps += ju
        if len(growth) > 100:
            P["growth"] = float(np.median(growth))
        if len(lens) > 100:
            P["sigma_end"] = max(mad(lens) / np.sqrt(2.0), 0.5)
        if len(gaps) > 100:
            P["sigma_gap"] = max(mad(gaps), 0.5)
        if len(divlen) > 20:
            P["div_mu"] = float(np.median(divlen)); P["div_sd"] = max(mad(divlen), 0.05 * P["div_mu"])
        if len(jumps) > 20:
            P["jump_scale_px"] = float(np.mean(jumps))
        print(f"round {r + 1}: {len(labs)} trenches, {time.perf_counter() - start:.0f} s; events {tot}; "
              f"growth {P['growth']:.4f}, end noise {P['sigma_end']:.2f} px, gap noise {P['sigma_gap']:.2f} px, "
              f"division {P['div_mu']:.1f} ± {P['div_sd']:.1f} px, jump scale {P['jump_scale_px']:.1f} px", flush=True)
    PARAMS.write_text(json.dumps({k: P[k] for k in FITTED}, indent=1))


def write(root, tid, lab, recs, side, frames):
    from pymmm.tracks import tracklets_and_lineages
    from pymmm.trench_images import write_tracks

    T, H, W = lab.shape
    seg = np.zeros(lab.shape, np.uint16)
    nodes, nlab, divisions = [], {}, []

    def add(t, cell, pix):
        if len(pix) == 0:
            return
        nlab[t] = nlab.get(t, 0) + 1
        seg[t].ravel()[pix] = nlab[t]
        nodes.append((t, cell, float((pix // W).mean()), float((pix % W).mean()), nlab[t]))

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
                pix = p[(p // W) < data[1]] if kind == "upper" else p[(p // W) >= data[1]]
            add(t, cell, pix)
        divisions += divs
    for sid, seq in side.items():
        for t, l in seq:
            add(t, ("side", sid), np.flatnonzero(lab[t].ravel() == l))
    by_cell = {}
    for k, (t, cell, *_r) in enumerate(nodes):
        by_cell.setdefault(cell, []).append((t, k))
    edges = []
    for cell, seq in by_cell.items():
        seq.sort()
        edges += [(a[1], b[1]) for a, b in zip(seq, seq[1:])]
    for mother, daughter in divisions:
        ms, ds = by_cell.get(mother), by_cell.get(daughter)
        if ms and ds:
            edges.append((max(ms)[1], min(ds)[1]))
    e = np.array(edges, np.int64).reshape(-1, 2)
    tt = np.array([n[0] for n in nodes], np.int32)
    track_id, lineage_id = tracklets_and_lineages(len(nodes), tt, e)
    props = {"t": tt, "y": np.array([n[2] for n in nodes], np.float32), "x": np.array([n[3] for n in nodes], np.float32),
             "seg_id": np.array([n[4] for n in nodes], np.int32), "track_id": track_id, "lineage_id": lineage_id}
    write_tracks(root, tid, OUT, props, e, np.full(len(e), np.nan, np.float32), seg,
                 extra={"tracker": TRACKER, "segmentation": SEG, "params": json.dumps(P, default=str),
                        "channels": [str(c) for c in root["C"][:]]})


def alternatives(names):
    """The hypotheses (indices into names; names[0] is the base) used as alternatives."""
    if BASE_ONLY:
        return []
    if ALTS is not None:
        return [names.index(n) for n in ALTS if n != names[0]]
    return list(range(1, len(names)))


def record_run(root):
    """Note this run (its segmentation and settings) in the experiment group."""
    from pymmm.trench_images import update_experiment

    update_experiment(root, "tracks", OUT, {
        "tracker": TRACKER, "segmentation": SEG, "params": json.loads(json.dumps(P, default=str)),
        "alternatives": "none" if BASE_ONLY else (ALTS or "all"), "made_by": "external/chaintrack/chain.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def run(trenches, root=None):
    from pymmm.trench_images import open_experiment, segmentation

    if root is None:
        root = open_experiment(GROUP, mode="r+")
        record_run(root)
    for tid in trenches:
        names, labels = segmentation(root, tid, SEG)
        use = [0] + alternatives(names)
        labs = [clean(np.asarray(labels[h][...])) for h in use]
        start = time.perf_counter()
        recs, side, frames = track(labs)
        print(f"trench {tid}: {len(use)} hypotheses, {time.perf_counter() - start:.0f} s; events {stats_of(recs)[0]}", flush=True)
        write(root, tid, labs[0], recs, side, frames)
    print(f"tracks: {GROUP}/trench_NNNN/tracks/{OUT}.geff", flush=True)


def follow(trenches):
    """Track each trench as soon as its segmentation is finished (candidates.py still running),
    until every one is tracked."""
    from pymmm.trench_images import has_segmentation, open_experiment

    root = open_experiment(GROUP, mode="r+")
    record_run(root)
    left = list(trenches)
    while left:
        ready = [t for t in left if has_segmentation(root, t, SEG)]
        for t in ready:
            try:
                run([t], root)
            except Exception:   # noqa: BLE001 - one bad trench must not stop the others
                import traceback
                print(f"trench {t} failed:\n{traceback.format_exc()}", flush=True)
        left = [t for t in left if t not in ready]
        if left and not ready:
            time.sleep(30)


def parse(args):
    out = []
    for a in args:
        if "-" in a:
            lo, hi = map(int, a.split("-"))
            out += list(range(lo, hi + 1))
        else:
            out.append(int(a))
    return out


if __name__ == "__main__":
    args = sys.argv[2:]
    for flag, takes in (("--seg", 1), ("--out", 1), ("--alts", 1), ("--base-only", 0), ("--group", 1)):
        if flag in args:
            k = args.index(flag); args = args[:k] + args[k + 1 + takes:]
    cmd, rest = sys.argv[1], parse(args)
    {"fit": fit, "run": run, "follow": follow}[cmd](rest)
