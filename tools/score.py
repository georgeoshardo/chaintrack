"""Score tracks against the curated trenches (read-only), as in the aligntrack comparison,
plus how many curated masks the tracker's masks reproduce.

    python score.py <collection> [<collection> ...]     # e.g. out/tracks.zarr/hypilp_all
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from pymmm.tracks import read_trench_geff

ROOT = Path("/data/phage-signatures")
CUR = ROOT / "20261005_pbs_psc_yfp_ty013.curation.zarr/trackastra_S1"


def load(group):
    g = read_trench_geff(group)
    seg = np.asarray(zarr.open_array(str(group / "segmentation"), mode="r")[:])
    n = len(g["t"])
    parents, children = [[] for _ in range(n)], [[] for _ in range(n)]
    for p, c in g["edges"]:
        parents[c].append(int(p))
        children[p].append(int(c))
    key = {(int(t), int(s)): i for i, (t, s) in enumerate(zip(g["t"], g["seg_id"]))}
    return dict(g=g, seg=seg, parents=parents, children=children, key=key, n=n)


def match(gt, pr, thresh):
    out = {}
    for t in range(gt["seg"].shape[0]):
        a, b = gt["seg"][t].ravel().astype(np.int64), pr["seg"][t].ravel().astype(np.int64)
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        aa, bb = np.bincount(a), np.bincount(b)
        pairs, inter = np.unique(a[both] * 100000 + b[both], return_counts=True)
        for pair, i in zip(pairs, inter):
            la, lb = divmod(int(pair), 100000)
            if i / (aa[la] + bb[lb] - i) > thresh:
                ga, pb = gt["key"].get((t, la)), pr["key"].get((t, lb))
                if ga is not None and pb is not None:
                    out[ga] = pb
    return out


def score(group, tid):
    gt = load(CUR / f"trench_{tid:04d}")
    pr = load(group)
    m = match(gt, pr, 0.5)
    inv = {v: k for k, v in m.items()}
    c = Counter()
    c["curated masks"] = gt["n"]
    c["masks reproduced (IoU >= 0.8)"] = len(match(gt, pr, 0.8))
    for p, ch in enumerate(gt["children"]):
        for k in ch:
            c["curated links"] += 1
            if p in m and k in m:
                c["scored links"] += 1
                c["correct links"] += m[p] in pr["parents"][m[k]]
            else:
                c["links on a mask it got wrong"] += 1
    c["false links"] = sum(1 for P, ch in enumerate(pr["children"]) for K in ch
                           if P in inv and K in inv and inv[P] not in gt["parents"][inv[K]])
    gdiv = [p for p, ch in enumerate(gt["children"]) if len(ch) == 2]
    pdiv = {P for P, ch in enumerate(pr["children"]) if len(ch) == 2}
    c["curated divisions"] = len(gdiv)
    c["divisions found exactly"] = sum(1 for p in gdiv if p in m and m[p] in pdiv
                                       and set(pr["children"][m[p]]) == {m.get(k) for k in gt["children"][p]})
    c["extra divisions"] = sum(1 for P in pdiv if not (P in inv and len(gt["children"][inv[P]]) == 2))
    # timing-tolerant: a division by the mother at about the same place within 2 frames
    gev = [(int(gt["g"]["t"][p]), float(gt["g"]["y"][p])) for p in gdiv]
    pev = [(int(pr["g"]["t"][P]), float(pr["g"]["y"][P])) for P in pdiv]
    used, near = set(), 0
    for t, y in gev:
        for j, (tp, yp) in enumerate(pev):
            if j not in used and abs(tp - t) <= 2 and abs(yp - y) <= 20:
                used.add(j)
                near += 1
                break
    c["divisions within 2 frames"] = near
    c["predicted divisions with no curated one within 2 frames"] = len(pev) - len(used)
    return c


if __name__ == "__main__":
    groups = [Path(a) for a in sys.argv[1:]] or [ROOT / "20261005_pbs_psc_yfp_ty013.tracks.zarr/trackastra_S1"]
    for tid in (0, 1):
        for coll in groups:
            group = coll / f"trench_{tid:04d}"
            if not group.is_dir():
                continue
            c = score(group, tid)
            print(f"trench {tid}  {coll.name:16s} links {c['correct links']}/{c['scored links']} "
                  f"({c['correct links'] / c['scored links']:.1%}), false {c['false links']}, "
                  f"on wrong masks {c['links on a mask it got wrong']}; divisions {c['divisions found exactly']}/"
                  f"{c['curated divisions']} (within 2 frames {c['divisions within 2 frames']}), extra "
                  f"{c['extra divisions']} (none near {c['predicted divisions with no curated one within 2 frames']}); "
                  f"masks {c['masks reproduced (IoU >= 0.8)']}/{c['curated masks']}")
