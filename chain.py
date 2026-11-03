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
The masks (S1 only, shapes only, no image) are observations, cleaned first: the bottom
crop_bottom_px rows (the halo at the trench opening) are removed, and so is any mask shorter
than the smallest plausible newborn, (div_mu - 3 div_sd) / 2. What is left is: one mask per cell, one cell
cut into two masks, one mask over two cells. Each frame the predicted chain is aligned to
the masks in order; a cell is judged by its own length and by the gap between its mask
and the mask of the last cell above that has one, so errors never build up down the column.
A beam of lineage histories lets the frames after an event decide what it was.

Fitted without labels: growth, length noise, gap noise, length at division, jump size.
Each segmentation has its own fit (params/<segmentation>.json).

`run` writes the tracks to the experiment's tracks store as the collection chain_S1 (the
mCherry masks: <experiment>.tracks.zarr; another segmentation <seg>: <experiment>.<seg>.tracks.zarr)
and copies them to the matching curation store; a trench with edits there is never replaced.

    cd /data/phage-signatures
    pixi run -e seg python external/chaintrack/chain.py fit 2-9 --seg pc_affinity_e250
    pixi run -e seg python external/chaintrack/chain.py run 0-9 --seg pc_affinity_e250
    pixi run curate 0 "--experiment cellpose_finetune/experiments/pc_affinity_e250.toml --tracker chain"
"""
import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
import zarr
from scipy import ndimage as ndi
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/phage-signatures")
EXPERIMENT = "20261005_pbs_psc_yfp_ty013"
STORE = ROOT / f"{EXPERIMENT}.trenches.zarr"
# which segmentation's masks to track (--seg NAME, default the mCherry one); each has its own fit
SEG = sys.argv[sys.argv.index("--seg") + 1] if "--seg" in sys.argv else "mcherry"
PARAMS = HERE / "params" / f"{SEG}.json"
TRACKER = "chain"
_stem = EXPERIMENT if SEG == "mcherry" else f"{EXPERIMENT}.{SEG}"
TRACKS = ROOT / f"{_stem}.tracks.zarr" / f"{TRACKER}_S1"
CURATION = ROOT / f"{_stem}.curation.zarr" / f"{TRACKER}_S1"

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
    p_under=0.01,              # one mask over two cells
    p_under_sisters=0.2,       # ... sisters born in the last few frames
    sisters_frames=3,
    p_miss=0.01, max_miss=3,
    p_lysis=0.003,
    jump_scale_px=20.0,        # typical size of a jump (fitted)
    centre_px=6,               # masks further than this from the centre line are off the chain
    side_sigma_y=15.0, side_sigma_area=0.1, side_sigma_x=4.0,   # an off-centre cell between frames
    side_max_cost=16.0,        # ... anything worse is not the same cell (4 sigma)
    exit_visible_px=8,
    crop_bottom_px=30,         # rows removed at the trench opening, where the halo makes masks unreliable
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
          H, sig_gap, p_jump, p_slide, rng, vis, force):
    """Order-preserving alignment of n predicted cells to m masks. State (i cells, j masks,
    a = 1 + the last cell given a mask, 0 if none yet). Spacing is judged against the mask
    of that cell (or the closed end); size by the move's own terms."""
    n, m = sm.shape
    dp = np.full((n + 1, m + 1, n + 1), INF)
    bc = np.full((n + 1, m + 1, n + 1), -1, np.int64)
    bi = np.zeros((n + 1, m + 1, n + 1), np.int64)
    bj = np.zeros((n + 1, m + 1, n + 1), np.int64)
    ba = np.zeros((n + 1, m + 1, n + 1), np.int64)
    dp[0, 0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            for a in range(n + 1):
                c0 = dp[i, j, a]
                if c0 >= INF / 2:
                    continue
                # where cell i would start: after the last cell with a mask, plus the space between
                pt = 0.0
                if i < n:
                    if a == 0:
                        pt = ptop[i]
                    else:
                        pt = obot[j - 1] + 1 + (ptop[i] - (ptop[a - 1] + plen[a - 1]))
                    f = force[i]
                    gone = pt >= H - vis
                    sp = 0.0
                    # spacing cost of putting cell i's top at mask j's top
                    if j < m:
                        if a == 0:
                            sp = mix(otop[j] - ptop[i], sig_gap, p_slide, rng)
                        else:
                            sp = mix(otop[j] - pt, sig_gap, p_jump, rng)
                    # match
                    if j < m and (f < 0 or f == 0):
                        c = c0 + sp + sm[i, j] + nodiv[i]
                        if c < dp[i + 1, j + 1, i + 1]:
                            dp[i + 1, j + 1, i + 1] = c; bc[i + 1, j + 1, i + 1] = 0
                            bi[i + 1, j + 1, i + 1] = i; bj[i + 1, j + 1, i + 1] = j; ba[i + 1, j + 1, i + 1] = a
                    # divide / cut into two masks
                    if j + 1 < m:
                        if f < 0 or f == 1:
                            c = c0 + sp + sd[i, j] + divc[i]
                            if c < dp[i + 1, j + 2, i + 1]:
                                dp[i + 1, j + 2, i + 1] = c; bc[i + 1, j + 2, i + 1] = 1
                                bi[i + 1, j + 2, i + 1] = i; bj[i + 1, j + 2, i + 1] = j; ba[i + 1, j + 2, i + 1] = a
                        if f < 0 or f == 2:
                            c = c0 + sp + so[i, j] + overc[i] + nodiv[i]
                            if c < dp[i + 1, j + 2, i + 1]:
                                dp[i + 1, j + 2, i + 1] = c; bc[i + 1, j + 2, i + 1] = 2
                                bi[i + 1, j + 2, i + 1] = i; bj[i + 1, j + 2, i + 1] = j; ba[i + 1, j + 2, i + 1] = a
                    # two cells in one mask
                    if i + 1 < n and j < m and (f < 0 or f == 3) and (force[i + 1] < 0 or force[i + 1] == 3):
                        c = c0 + sp + su[i, j] + underc[i] + nodiv[i] + nodiv[i + 1]
                        if c < dp[i + 2, j + 1, i + 2]:
                            dp[i + 2, j + 1, i + 2] = c; bc[i + 2, j + 1, i + 2] = 3
                            bi[i + 2, j + 1, i + 2] = i; bj[i + 2, j + 1, i + 2] = j; ba[i + 2, j + 1, i + 2] = a
                    # no mask for cell i
                    if not gone:
                        if f < 0 or f == 4:
                            c = c0 + missc[i] + nodiv[i]
                            if c < dp[i + 1, j, a]:
                                dp[i + 1, j, a] = c; bc[i + 1, j, a] = 4; bi[i + 1, j, a] = i; bj[i + 1, j, a] = j; ba[i + 1, j, a] = a
                    near_bottom = ptop[i] + 2 * plen[i] >= H   # its own prediction, not one pushed by a jump above
                    if (not gone or not near_bottom) and (f < 0 or f == 5):
                        c = c0 + lysisc
                        if c < dp[i + 1, j, a]:
                            dp[i + 1, j, a] = c; bc[i + 1, j, a] = 5; bi[i + 1, j, a] = i; bj[i + 1, j, a] = j; ba[i + 1, j, a] = a
                    if (f < 0 or f == 6) and near_bottom:
                        if gone:
                            ec = 0.0
                        elif pt + plen[i] >= H:
                            ec = 1.0 if H - pt < 0.6 * plen[i] else 3.0
                        else:
                            ec = INF
                        if ec < INF:
                            c = c0 + ec
                            if c < dp[i + 1, j, a]:
                                dp[i + 1, j, a] = c; bc[i + 1, j, a] = 6; bi[i + 1, j, a] = i; bj[i + 1, j, a] = j; ba[i + 1, j, a] = a
                if j < m:  # a mask no cell explains
                    c = c0 + APPEAR_COST
                    if c < dp[i, j + 1, a]:
                        dp[i, j + 1, a] = c; bc[i, j + 1, a] = 7; bi[i, j + 1, a] = i; bj[i, j + 1, a] = j; ba[i, j + 1, a] = a
    best_a, best = 0, INF
    for a in range(n + 1):
        if dp[n, m, a] < best:
            best, best_a = dp[n, m, a], a
    rows = np.full((n + m + 1, 3), -1, np.int64)
    k = 0
    i, j, a = n, m, best_a
    while (i > 0 or j > 0) and bc[i, j, a] >= 0:
        code = bc[i, j, a]
        pi, pj, pa = bi[i, j, a], bj[i, j, a], ba[i, j, a]
        cell = -1 if code == 7 else pi
        mask = pj if (code <= 3 or code == 7) else -1
        rows[k, 0] = code; rows[k, 1] = cell; rows[k, 2] = mask
        k += 1
        i, j, a = pi, pj, pa
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


def split_side(lab, centre, H):
    """Chain masks per frame, and off-centre cells tracked first (an off-centre cell keeps its
    identity through the nearest similar mask, so it survives drifting onto the centre line)."""
    obs, tracks, active, nid = [], {}, [], 0
    for t in range(lab.shape[0]):
        masks = frame_masks(lab[t], H)
        claimed, nxt = set(), []
        for tid, last, missed in active:
            # the same cell: it can move along the trench a fair way, but keeps its size and
            # stays near its own sideways position
            best = None
            for o in masks:
                if o[2] in claimed:
                    continue
                c = (((o[4] - last[4]) / P["side_sigma_y"]) ** 2 + (np.log(o[5] / last[5]) / P["side_sigma_area"]) ** 2
                     + ((o[6] - last[6]) / P["side_sigma_x"]) ** 2)
                if c <= P["side_max_cost"] and (best is None or c < best[0]):
                    best = (c, o)
            if best is not None:
                o = best[1]
                claimed.add(o[2]); tracks[tid].append((t, o[2])); nxt.append((tid, o, 0))
            elif missed < P["max_miss"]:  # no mask this frame, like a chain cell: wait for it
                nxt.append((tid, last, missed + 1))
        for o in masks:
            if o[2] not in claimed and abs(o[6] - centre) > P["centre_px"]:
                nid += 1
                tracks[-nid] = [(t, o[2])]; claimed.add(o[2]); nxt.append((-nid, o, 0))
        active = nxt
        obs.append(sorted(o[:6] for o in masks if o[2] not in claimed))
    return obs, tracks


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


def size_costs(h, obs, grown, t):
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
    sd = np.full((n, max(m - 1, 1)), INF)
    so = np.full((n, max(m - 1, 1)), INF)
    if n and m >= 2:
        span = obot[1:] - otop[:-1] + 1
        whole = fit(span, grown, otr[1:], trunc)
        so[:, :m - 1] = whole
        d1 = np.log(olen[None, :-1] / (grown[:, None] / 2))
        d2 = np.log(olen[None, 1:] / (grown[:, None] / 2))
        d2 = np.where(otr[None, 1:] & (d2 < 0), 0.0, d2)
        sd[:, :m - 1] = whole + 0.5 * (d1 / P["sigma_daughter"]) ** 2 + 0.5 * (d2 / P["sigma_daughter"]) ** 2
    su = np.full((max(n - 1, 1), m), INF)
    underc = np.zeros(max(n - 1, 1))
    if n >= 2 and m:
        su[:n - 1] = fit(olen, grown[:-1] + grown[1:], otr, trunc[:-1] | trunc[1:])
        sisters = np.array([h.parent[i] >= 0 and h.parent[i] == h.parent[i + 1]
                            and t - h.born[i] <= P["sisters_frames"] for i in range(n - 1)])
        underc[:n - 1] = -np.log(np.where(sisters, P["p_under_sisters"], P["p_under"]))
    return sm, sd, so, su, underc, otop, obot


def costs(h, obs, H, t):
    ptop, grown = predict(h)
    sm, sd, so, su, underc, otop, obot = size_costs(h, obs, grown, t)
    divc, nodiv = hazards(np.asarray(h.length, float), grown, np.asarray(h.trunc, bool)) if h.ids else (np.zeros(0), np.zeros(0))
    missc = np.where(np.asarray(h.unseen) < P["max_miss"], -np.log(P["p_miss"]), INF) if h.ids else np.zeros(0)
    overc = np.full(len(grown), -np.log(P["p_over"]))
    args = (sm, sd, so, su, nodiv, divc, underc, overc, missc, -np.log(P["p_lysis"]),
            ptop, grown, otop, obot, float(H), P["sigma_gap"], P["p_jump"], P["p_slide"], P["jump_scale_px"],
            float(P["exit_visible_px"]))
    return args, ptop, grown


def apply(h, rows, obs, ptop, grown, t, cost, new_id, args=None):
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

    for code, i, j in rows:
        code, i, j = int(code), int(i), int(j)
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
            for oo in (obs[j], obs[j + 1]):
                c = new_id()
                L = oo[1] - oo[0] + 1
                keep(c, oo[0], max(L, grown[i] / 2) if oo[3] else L, h.ids[i], t, 0, oo[3])
                entries.append((c, "mask", (oo[2],)))
                divisions.append((h.ids[i], c))
            events["div"] += 1
            if not h.trunc[i] and not obs[j + 1][3]:
                divlen.append(grown[i])
            ref_bot, ref_cell = obs[j + 1][1], i
        elif code == 2:
            a, b = obs[j], obs[j + 1]
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


def step(hyps, obs, H, t, new_id):
    children = []
    for h in hyps:
        args, ptop, grown = costs(h, obs, H, t)
        if P.get("debug"):
            DEBUG_ARGS["args"] = args
        force = np.full(len(h.ids), -1, np.int64)
        cost, rows = align(*args, force)
        if cost >= INF / 2:
            continue
        children.append(apply(h, rows, obs, ptop, grown, t, cost, new_id, args))
        for row in rows:
            alt = ALT.get(int(row[0]))
            if alt is None:
                continue
            force = np.full(len(h.ids), -1, np.int64)
            force[int(row[1])] = alt
            c2, r2 = align(*args, force)
            if c2 < INF / 2:
                children.append(apply(h, r2, obs, ptop, grown, t, c2, new_id, args))
        missing = [int(row[1]) for row in rows if int(row[0]) == 4]
        if len(missing) >= 2:  # a lysis wave: all of them at once
            force = np.full(len(h.ids), -1, np.int64)
            force[missing] = 5
            c2, r2 = align(*args, force)
            if c2 < INF / 2:
                children.append(apply(h, r2, obs, ptop, grown, t, c2, new_id, args))
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
    """The masks the tracker sees: without the bottom crop_bottom_px rows, and without masks
    shorter than the smallest plausible newborn, (div_mu - 3 div_sd) / 2."""
    lab = np.array(lab)
    lab[:, lab.shape[1] - P["crop_bottom_px"]:] = 0
    shortest = (P["div_mu"] - 3 * P["div_sd"]) / 2
    for f in lab:
        ids = np.unique(f)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        rows = np.nonzero(f)[0]
        lo = ndi.minimum(rows, f[np.nonzero(f)], ids)
        hi = ndi.maximum(rows, f[np.nonzero(f)], ids)
        small = ids[(np.asarray(hi) - np.asarray(lo) + 1) < shortest]
        if len(small):
            f[np.isin(f, small)] = 0
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


def track(lab):
    T, _, W = lab.shape
    H = visible_end(lab)
    allc = np.nonzero(lab)[2]
    centre = float(np.median(allc)) if len(allc) else W / 2
    obs, side = split_side(lab, centre, H)
    new_id = Counter()
    first = next((t for t in range(T) if obs[t]), None)
    if first is None:
        return [], side
    o0 = obs[first]
    ids = [new_id() for _ in o0]
    rec = (None, first, [(c, "mask", (o[2],)) for c, o in zip(ids, o0)], [], {}, [], [], [], [], None)
    hyps = [Hyp(ids, [float(o[0]) for o in o0], [float(o[1] - o[0] + 1) for o in o0], [-1] * len(o0),
                [first] * len(o0), [0] * len(o0), [o[3] for o in o0], 0.0, rec)]
    for t in range(first + 1, T):
        hyps = step(hyps, obs[t], H, t, new_id)
    best = min(hyps, key=lambda x: x.score)
    recs, r = [], best.rec
    while r is not None:
        recs.append(r)
        r = r[0]
    return recs[::-1], side


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
    store = zarr.open_group(str(STORE), mode="r")
    labs = [clean(l) for l in (np.asarray(store[f"segmentation/{SEG}/labels"][0, t]) for t in trenches) if l.any()]
    mad = lambda v: 1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v))))
    for r in range(rounds):
        tot = dict(div=0, over=0, under=0, miss=0, lysis=0, exit=0, appear=0, jump=0, slide=0)
        lens, gaps, divlen, growth, jumps = [], [], [], [], []
        start = time.perf_counter()
        for lab in labs:
            a, l, g, d, gr, ju = stats_of(track(lab)[0])
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


def write(collection, tid, lab, recs, side):
    from pymmm.tracks import tracklets_and_lineages, write_trench_geff

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
        flat = lab[t].ravel()
        for cell, kind, data in entries:
            if kind == "mask":
                pix = np.flatnonzero(flat == data[0])
            elif kind == "union":
                pix = merge_pair(np.flatnonzero(flat == data[0]), np.flatnonzero(flat == data[1]), W)
            else:
                p = np.flatnonzero(flat == data[0])
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
    store = zarr.open_group(str(STORE), mode="r")
    write_trench_geff(collection, tid, props, e, np.full(len(e), np.nan, np.float32), seg,
                      pixel_size=float(store["acquisition/pixel_size"][()]),
                      interval_min=float(store["acquisition/time_interval"][()]) / 60000,
                      extra={"tracker": TRACKER, "segmentation": SEG, "trench_store": str(STORE), "params": json.dumps(P, default=str),
                             "channels": [str(c) for c in store["C"][:]]})


def run(trenches):
    import shutil
    from pymmm.tracks import is_edited, prepare_curation

    store = zarr.open_group(str(STORE), mode="r")
    for tid in trenches:
        lab = clean(np.asarray(store[f"segmentation/{SEG}/labels"][0, tid]))
        start = time.perf_counter()
        recs, side = track(lab)
        print(f"trench {tid}: {time.perf_counter() - start:.0f} s; events {stats_of(recs)[0]}", flush=True)
        write(TRACKS, tid, lab, recs, side)
    # refresh the curation copies of these trenches unless they have been edited
    for tid in trenches:
        copy = CURATION / f"trench_{tid:04d}"
        if copy.exists() and not is_edited(copy):
            shutil.rmtree(copy)
    print(f"tracks: {TRACKS}")
    print(f"curation: {CURATION}", prepare_curation(TRACKS, CURATION))


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
    if "--seg" in args:
        k = args.index("--seg"); args = args[:k] + args[k + 2:]
    cmd, rest = sys.argv[1], parse(args)
    fit(rest) if cmd == "fit" else run(rest)
