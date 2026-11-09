"""Score tracking runs against the user's Reviewed curations (experiment group, same crops, all
frames), tolerant of what the curation cannot or need not pin down:

- Masks are paired per frame by IoU >= 0.65 (a mask over two cells pairs with neither). Only chain
  masks above the bottom are judged: a mask reaching the bottom EDGE rows (cut off by the crop, never
  in the curation) or off the centre line is not judged as a cell, but a link may pass through it.
- Each judged curated mask b, paired with run mask b': its predecessor is its nearest ancestor that
  is paired and judged (walking up through masks that are not, so a cell hidden at the bottom for a
  few frames, or a division drawn a frame earlier or later on other masks, is not a link error).
  The run agrees on b when b's predecessor and b''s predecessor are partners (or both have none).
  Disagreements are classed:
    shift     the run's predecessor is a different cell in the frame the curation's comes from
    root      the run starts a new cell where the curation continues one
    extra     the run continues a cell where the curation starts one
    skip      the run's predecessor is from another frame (a frame skipped, or not skipped)
    other
- Masks: curated judged masks with no run partner (missed), run judged masks with no curated
  partner (extra), and run masks off the centre line (side; the user deleted them all).
- Divisions: a curated division (a judged mask with two judged children, after walking through
  unjudged masks) is found if the run divides the same cell (two children, or a daughter listed in
  the GEFF's extra "div_one") within about DIV_TOL frames along the curated lineage; a curated mother
  with one child at the bottom is not counted as a division. A run division of two judged daughters
  with no curated division near it is extra.

    pixi run -e seg python external/chaintrack/tools/curation_score.py chain_S1,chain_fix_S1 [--v]
    (--curation RUN: the curated run, default chain_clearbottom_endfix_S1; --trenches 0,2,3: default
    the Reviewed ones; a run as GROUP:RUN is read from another experiment group, e.g. a scratch copy)
"""
import sys
from collections import Counter, defaultdict

import numpy as np
import zarr

S = "/data/phage-signatures/20261005_pbs_psc_yfp_ty013.zarr"
RUNS = sys.argv[1].split(",")
CURRUN = sys.argv[sys.argv.index("--curation") + 1] if "--curation" in sys.argv else "chain_clearbottom_endfix_S1"
GROUP = sys.argv[sys.argv.index("--group") + 1] if "--group" in sys.argv else S
VERBOSE = "--v" in sys.argv
EDGE = 6          # masks reaching the bottom EDGE rows are cut off by the crop: not judged
CENTRE_PX = 6     # masks further than this from the centre line are side objects
DIV_TOL = 2
PAIR_IOU = 0.65  # a merge of two cells has IoU about 0.5 with each: it pairs with neither


def graph(path):
    g = zarr.open_group(path, mode="r")
    ids = np.asarray(g["nodes/ids"][:])
    p = g["nodes/props"]
    t = np.asarray(p["t/values"][:]).astype(int); s = np.asarray(p["seg_id/values"][:]).astype(int)
    pos = {int(i): k for k, i in enumerate(ids)}
    e = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    e = [(pos[int(a)], pos[int(b)]) for a, b in e]
    extra = dict(g.attrs).get("geff", {}).get("extra", {})
    return dict(t=t, s=s, e=e, extra=extra)


def boxes(seg):
    out = {}
    for f in range(seg.shape[0]):
        fr = seg[f]
        ys, xs = np.nonzero(fr)
        if not len(ys):
            continue
        v = fr[ys, xs]
        order = np.argsort(v, kind="stable")
        v, ys, xs = v[order], ys[order], xs[order]
        ls, first = np.unique(v, return_index=True)
        bounds = list(first) + [len(v)]
        for k, l in enumerate(ls):
            yy, xx = ys[bounds[k]:bounds[k + 1]], xs[bounds[k]:bounds[k + 1]]
            out[(f, int(l))] = (int(yy.min()), int(yy.max()), len(yy), float(xx.mean()))
    return out


def match(segA, segB):
    out = {}
    for f in range(segA.shape[0]):
        a, b = segA[f].ravel().astype(np.int64), segB[f].ravel().astype(np.int64)
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        aa, bb = np.bincount(a), np.bincount(b)
        pairs, inter = np.unique(a[both] * 100000 + b[both], return_counts=True)
        cand = []
        for p, i in zip(pairs, inter):
            la, lb = divmod(int(p), 100000)
            iou = i / (aa[la] + bb[lb] - i)
            if iou >= PAIR_IOU:
                cand.append((iou, la, lb))
        ua, ub = set(), set()
        for iou, la, lb in sorted(cand, reverse=True):
            if la in ua or lb in ub:
                continue
            out[(f, la)] = (f, lb); ua.add(la); ub.add(lb)
    return out


class G:
    """A run or curation of one trench: nodes keyed (t, label), parents, children, boxes."""

    def __init__(self, geff, labels):
        g = graph(geff)
        self.extra = g["extra"]
        self.seg = np.asarray(zarr.open_array(labels, mode="r")[:])
        self.T, self.H, self.W = self.seg.shape
        self.box = boxes(self.seg)
        key = [(int(t), int(s)) for t, s in zip(g["t"], g["s"])]
        self.nodes = [k for k in key if k in self.box]
        ok = set(self.nodes)
        self.parent, self.children = {}, defaultdict(list)
        # daughters whose sister went past the bottom unseen (chaintrack's one-daughter divisions): GEFF ids
        lone_ids = set(self.extra.get("div_one", []))
        gz = zarr.open_group(geff, mode="r")
        gid = np.asarray(gz["nodes/ids"][:])
        self.lone = {key[k] for k in range(len(key)) if int(gid[k]) in lone_ids and key[k] in self.box}
        for a, b in g["e"]:
            ka, kb = key[a], key[b]
            if ka in ok and kb in ok:
                self.parent[kb] = ka
                self.children[ka].append(kb)


def judged(g, k, centre):
    top, bot, area, col = g.box[k]
    return bot < g.H - EDGE and abs(col - centre) <= CENTRE_PX


def score(run, cur, centre):
    m = match(run.seg, cur.seg)                     # run -> cur
    minv = {v: k for k, v in m.items()}
    jr = {k for k in run.nodes if judged(run, k, centre)}
    jc = {k for k in cur.nodes if judged(cur, k, centre)}
    paired_c = {k for k in jc if k in minv and minv[k] in jr}

    def pred(g, k, ok):
        p = g.parent.get(k)
        while p is not None and p not in ok:
            p = g.parent.get(p)
        return p

    ok_r = {minv[k] for k in paired_c}
    out = Counter(); events = []
    for b in sorted(paired_c):
        pc = pred(cur, b, paired_c)
        pr = pred(run, minv[b], ok_r)
        pr_c = m.get(pr) if pr is not None else None
        if pc == pr_c:
            out["agree"] += 1
            continue
        if pc is not None and pr_c is None:
            kind = "root"
        elif pc is None:
            kind = "extra"
        elif pr_c[0] == pc[0]:
            kind = "shift"
        elif pr_c[0] != pc[0]:
            kind = "skip"
        else:
            kind = "other"
        out[kind] += 1
        events.append((b[0], kind, pc, pr_c, b))
    out["missed"] = len(jc - paired_c)
    out["extra_masks"] = len([k for k in jr if k not in m or m[k] not in jc])
    out["side"] = len([k for k in run.nodes if abs(run.box[k][3] - centre) > CENTRE_PX])
    # divisions: judged mothers with two judged children (walking down through unjudged masks)

    def kids(g, k, ok):
        found, todo = [], list(g.children.get(k, []))
        while todo:
            c = todo.pop()
            if c in ok:
                found.append(c)
            else:
                todo += g.children.get(c, [])
        return found

    div_r = set()
    for k in run.nodes:
        mother = len(run.children.get(k, [])) == 2 or any(c in run.lone for c in run.children.get(k, []))
        if not mother:
            continue
        # the division, as the curated masks it lands on: the mother, its parent, its children
        for x in [k, run.parent.get(k)] + list(run.children.get(k, [])):
            if x is not None and x in m:
                div_r.add(m[x])
    cur_divs = [k for k in jc if len(kids(cur, k, jc)) == 2]
    found = 0
    for k in cur_divs:
        # the run divides the same cell within DIV_TOL frames: follow the curated cell up and down
        near = {k}
        p = k
        for _ in range(DIV_TOL):
            p = cur.parent.get(p)
            if p is None:
                break
            near.add(p)
        todo = [(c, 1) for c in cur.children.get(k, [])]
        while todo:
            c, d = todo.pop()
            if d > DIV_TOL:
                continue
            near.add(c)
            todo += [(x, d + 1) for x in cur.children.get(c, [])]
        if near & div_r:
            found += 1
        else:
            events.append((k[0], "division-missed", k, None, k))
    out["divisions"] = len(cur_divs); out["div_found"] = found
    # run divisions with no curated division nearby
    cur_div_near = set()
    for k in cur_divs:
        p = k
        for _ in range(DIV_TOL + 1):
            if p is None:
                break
            cur_div_near.add(p)
            p = cur.parent.get(p)
        todo = [(c, 1) for c in cur.children.get(k, [])]
        while todo:
            c, d = todo.pop()
            if d > DIV_TOL:
                continue
            cur_div_near.add(c)
            todo += [(x, d + 1) for x in cur.children.get(c, [])]
    for k in run.nodes:
        if len(run.children.get(k, [])) == 2 and judged(run, k, centre):
            kids_judged = [c for c in run.children[k] if judged(run, c, centre)]
            if len(kids_judged) < 2:
                continue        # a division at the bottom (a daughter cut off): not judged
            if not (k in m and m[k] in cur_div_near):
                out["div_extra"] += 1
                events.append((k[0], "division-extra", m.get(k), None, k))
    return out, events


def centre_of(tid):
    raw = np.asarray(zarr.open_array(f"{S}/trench_{tid:04d}/labels/pc_affinity_ensemble.e250/0", mode="r")[:])
    return float(np.median(np.nonzero(raw)[2]))


def reviewed_trenches():
    import glob
    import re
    out = []
    for p in sorted(glob.glob(f"{S}/trench_*/curation/{CURRUN}.geff")):
        tid = int(re.search(r"trench_(\d+)", p).group(1))
        if graph(p)["extra"].get("reviewed"):
            out.append(tid)
    return out


if __name__ == "__main__":
    trenches = ([int(x) for x in sys.argv[sys.argv.index("--trenches") + 1].split(",")]
                if "--trenches" in sys.argv else reviewed_trenches())
    cols = ["agree", "shift", "root", "extra", "skip", "missed", "extra_masks", "side", "divisions", "div_found", "div_extra"]
    tot = {r: Counter() for r in RUNS}
    for tid in trenches:
        cur = G(f"{S}/trench_{tid:04d}/curation/{CURRUN}.geff", f"{S}/trench_{tid:04d}/labels/curation.{CURRUN}/0")
        centre = centre_of(tid)
        for r in RUNS:
            grp, _, name = r.rpartition(":")
            grp = grp or GROUP
            run = G(f"{grp}/trench_{tid:04d}/tracks/{name}.geff", f"{grp}/trench_{tid:04d}/labels/tracks.{name}/0")
            out, ev = score(run, cur, centre)
            tot[r].update(out)
            print(f"trench {tid} {r.rpartition(':')[2]:40s} " + " ".join(f"{c}={out[c]}" for c in cols), flush=True)
            if VERBOSE:
                for t, kind, pc, pr, b in sorted(ev, key=lambda e: e[0]):
                    fb = lambda k, g: f"t{k[0]} {g.box[k][0]}-{g.box[k][1]}" if k is not None and k in g.box else "-"
                    print(f"    t{t:3d} {kind:16s} cur pred {fb(pc, cur)} | run pred {fb(pr, cur)} | at {fb(b, cur) if b in cur.box else fb(b, run)}")
    print()
    for r in RUNS:
        t = tot[r]
        bad = t["shift"] + t["root"] + t["extra"] + t["skip"]
        print(f"{r.rpartition(':')[2]:40s} link errors {bad:4d} (shift {t['shift']}, root {t['root']}, extra {t['extra']}, skip {t['skip']}) "
              f"of {bad + t['agree']}; masks missed {t['missed']}, extra {t['extra_masks']}, side {t['side']}; "
              f"divisions found {t['div_found']}/{t['divisions']}, extra {t['div_extra']}")
