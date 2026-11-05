"""Lineage disagreements of two runs with a curation, matching cells by overlap (read-only), so a
cell drawn with a different mask but linked the same way is not a disagreement. Each run cell is
matched to the curated cell in its frame with the best IoU (>= 0.5, one to one). A run link
disagrees when its two cells are not matched to a curated link; a curated link disagrees when no
run link realises it. The new run's new disagreements (absent from the old run's) are grouped
into events (frames within 2) and sorted: shift (two or more cells in one frame pair linked to a
neighbour's curated child), skip (a run link over more than one frame), division (a division in
one and not the other), bottom (every cell involved within 60 px of the crop line), other.

    pixi run -e seg python external/chaintrack/tools/lineage_diff.py <old run> <new run> <curated> 0,1,3 [crop line]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S


def rows_of(g, i):
    r = np.flatnonzero((g["seg"][g["g"]["t"][i]] == g["g"]["seg_id"][i]).any(1))
    return int(r.min()), int(r.max())


def match(run, cur):
    """run node -> curated node, by IoU >= 0.5 in the same frame, best first, one to one."""
    out = {}
    for f in range(cur["seg"].shape[0]):
        a, b = run["seg"][f].ravel().astype(np.int64), cur["seg"][f].ravel().astype(np.int64)
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        aa, bb = np.bincount(a), np.bincount(b)
        pairs, inter = np.unique(a[both] * 100000 + b[both], return_counts=True)
        cand = []
        for p, i in zip(pairs, inter):
            la, lb = divmod(int(p), 100000)
            iou = i / (aa[la] + bb[lb] - i)
            if iou >= 0.5:
                cand.append((iou, la, lb))
        used_a, used_b = set(), set()
        for iou, la, lb in sorted(cand, reverse=True):
            if la in used_a or lb in used_b:
                continue
            ra, rb = run["key"].get((f, la)), cur["key"].get((f, lb))
            if ra is not None and rb is not None:
                out[ra] = rb; used_a.add(la); used_b.add(lb)
    return out


def disagreements(run, cur, m):
    """Keys of disagreeing links: curated node ids where matched, ('run', t, top, bottom) otherwise."""
    name = lambda i: m[i] if i in m else ("run", int(run["g"]["t"][i])) + rows_of(run, i)
    cl = {(int(p), int(c)) for p, c in cur["g"]["edges"]}
    out, realised = {}, set()
    for p, c in run["g"]["edges"]:
        k = (name(p), name(c))
        if k in cl:
            realised.add(k)
        else:
            out[k] = ("run", int(p), int(c))
    for k in cl - realised:
        out[k] = ("curated",) + k
    return out


old, new, cur = (Path(a) for a in sys.argv[1:4])
trenches = [int(x) for x in sys.argv[4].split(",")]
line = int(sys.argv[5]) if len(sys.argv) > 5 else 625
summary = defaultdict(int)
for tid in trenches:
    name = f"trench_{tid:04d}"
    C, O, N = S.load(cur / name), S.load(old / name), S.load(new / name)
    dO = disagreements(O, C, match(O, C))
    mN = match(N, C)
    dN = disagreements(N, C, mN)
    fresh = {k: v for k, v in dN.items() if k not in dO}
    fixed = [k for k in dO if k not in dN]
    summary["fixed"] += len(fixed); summary["new"] += len(fresh)
    print(f"{name}: old {len(dO)} disagreeing links, new {len(dN)}; fixed {len(fixed)}, new {len(fresh)}")
    if not fresh:
        continue
    # each fresh disagreement as (frame, rows of the cells involved, is skip, is run link, parent/child)
    items = []
    for k, v in fresh.items():
        if v[0] == "run":
            p, c = v[1], v[2]
            t0, t1 = int(N["g"]["t"][p]), int(N["g"]["t"][c])
            items.append(dict(t=t0, rows=[rows_of(N, p), rows_of(N, c)], skip=t1 - t0 > 1, run=True,
                              div=len(N["children"][p]) == 2, p=p, c=c))
        else:
            p, c = v[1], v[2]
            items.append(dict(t=int(C["g"]["t"][p]), rows=[rows_of(C, p), rows_of(C, c)], skip=False, run=False,
                              div=len(C["children"][p]) == 2))
    items.sort(key=lambda d: d["t"])
    events, cur_ev = [], []
    for d in items:
        if cur_ev and d["t"] - cur_ev[-1]["t"] > 2:
            events.append(cur_ev); cur_ev = []
        cur_ev.append(d)
    events.append(cur_ev)
    cur_parent = {int(c): int(p) for p, c in C["g"]["edges"]}
    for ev in events:
        centres = [(r[0] + r[1]) / 2 for d in ev for r in d["rows"]]
        shifted = defaultdict(int)
        for d in ev:
            if d["run"] and d["p"] in mN and d["c"] in mN:
                cp = cur_parent.get(mN[d["c"]])
                if cp is not None and cp != mN[d["p"]] and not d["div"]:
                    shifted[d["t"]] += 1
        skips = sum(d["skip"] for d in ev)
        if max(shifted.values(), default=0) >= 2:
            kind = "shift"
        elif skips:
            kind = "skip"
        elif any(d["div"] for d in ev):
            kind = "division"
        elif all(c >= line - 60 for c in centres):
            kind = "bottom"
        else:
            kind = "other"
        summary[kind] += 1
        lo, hi = min(r[0] for d in ev for r in d["rows"]), max(r[1] for d in ev for r in d["rows"])
        print(f"   t{ev[0]['t']}-{ev[-1]['t']}: {kind:9s} {len(ev):2d} links, rows {lo}-{hi}")
print("all:", dict(summary))
