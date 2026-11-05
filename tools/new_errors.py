"""The new run's new disagreements with a curation, grouped into events and sorted by kind
(read-only). New = disagreements the old run did not have. An event is a run of frames within 2
of each other. Kinds:
  shift      in one frame pair, two or more cells each linked to its neighbour's curated child
  division   a division in one and not the other within 3 frames of a curated division
  bottom     every mask involved has its centre within 60 px of the crop line or below
  other      anything else
    pixi run -e seg python external/chaintrack/tools/new_errors.py <old run> <new run> <curated> 0,1,3 [crop line]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S


def load(group):
    g = S.load(group)
    t, sid = g["g"]["t"], g["g"]["seg_id"]
    key = {}
    for i in range(g["n"]):
        r = np.flatnonzero((g["seg"][t[i]] == sid[i]).any(1))
        key[i] = (int(t[i]), int(r.min()), int(r.max()))
    links = {(key[p], key[c]) for p, c in g["g"]["edges"]}
    kids, parents = defaultdict(list), defaultdict(list)
    for a, b in links:
        kids[a].append(b); parents[b].append(a)
    return links, kids, parents


old, new, cur = (Path(a) for a in sys.argv[1:4])
trenches = [int(x) for x in sys.argv[4].split(",")]
line = int(sys.argv[5]) if len(sys.argv) > 5 else 625
summary = defaultdict(int)
for tid in trenches:
    name = f"trench_{tid:04d}"
    C, ck, cp = load(cur / name)
    O, _, _ = load(old / name)
    N, nk, npar = load(new / name)
    fresh = (N ^ C) - (O ^ C)
    if not fresh:
        continue
    curated_divs = sorted({a[0] for a in ck if len(ck[a]) == 2})
    frames = sorted({a[0] for a, b in fresh})
    events, cur_ev = [], []
    for f in frames:
        if cur_ev and f - cur_ev[-1] > 2:
            events.append(cur_ev); cur_ev = []
        cur_ev.append(f)
    if cur_ev:
        events.append(cur_ev)
    for ev in events:
        links = [(a, b) for a, b in fresh if ev[0] <= a[0] <= ev[-1]]
        masks = {m for l in links for m in l}
        centre = lambda m: (m[1] + m[2]) / 2
        # shift: new-run links p->c where the curation gives c to p's neighbour, for 2+ cells in one frame pair
        shifted = defaultdict(int)
        for a, b in links:
            if (a, b) in N and (a, b) not in C and cp.get(b) and ck.get(a):
                if all(abs(centre(x) - centre(a)) > 3 for x in cp[b]):
                    shifted[a[0]] += 1
        div_n = any(len(nk[a]) == 2 for a, b in links if (a, b) in N)
        div_c = any(len(ck[a]) == 2 for a, b in links if (a, b) in C)
        near_div = any(abs(f - d) <= 3 for f in ev for d in curated_divs)
        if max(shifted.values(), default=0) >= 2 and not all(centre(m) >= line - 60 for m in masks):
            kind = "shift"
        elif all(centre(m) >= line - 60 for m in masks):
            kind = "bottom"
        elif (div_n or div_c) and near_div:
            kind = "division"
        else:
            kind = "other"
        summary[kind] += 1
        span = (min(m[1] for m in masks), max(m[2] for m in masks))
        skips = sum(1 for a, b in links if (a, b) in N and b[0] - a[0] > 1)
        summary["new skip links"] += skips
        print(f"{name} t{ev[0]}-{ev[-1]}: {kind:9s} {len(links):2d} links, rows {span[0]}-{span[1]}"
              + (f", {skips} skipping frames" if skips else ""))
print("events by kind:", dict(summary))
