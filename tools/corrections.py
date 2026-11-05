"""What a curator changed in a curated copy of tracks (read-only): links removed and added,
masks added or removed, each with its position (rows from the bottom of the crop) and the
lengths of the masks involved. The curated copy must come from the same run, so every
difference is a correction.

    pixi run -e seg python external/chaintrack/tools/corrections.py <tracks collection> <curated collection> [trenches]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S


def masks(group):
    g = S.load(group)
    t, sid = g["g"]["t"], g["g"]["seg_id"]
    ext = {}
    for i in range(g["n"]):
        m = g["seg"][t[i]] == sid[i]
        r, c = np.nonzero(m)
        ext[i] = (int(t[i]), int(r.min()), int(r.max()), int(c.max() - c.min() + 1), int(m.sum()))
    return g, ext


def describe(e, H):
    t, top, bot, w, area = e
    return f"t{t} rows {top}-{bot} (L{bot - top + 1}, w{w}, {H - 1 - bot} px above bottom)"


def main():
    run, cur = Path(sys.argv[1]), Path(sys.argv[2])
    trenches = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    for group in sorted(cur.glob("trench_*")):
        tid = int(group.name.split("_")[1])
        if trenches is not None and tid not in trenches:
            continue
        gr, er = masks(run / group.name)
        gc, ec = masks(group)
        H = gc["seg"].shape[1]
        key = lambda e: (e[0], e[1], e[2])
        node_r, node_c = {key(e) for e in er.values()}, {key(e) for e in ec.values()}
        link = lambda g, e: {(key(e[p]), key(e[c])) for p, c in g["g"]["edges"]}
        lr, lc = link(gr, er), link(gc, ec)
        by_key = {key(e): e for e in list(er.values()) + list(ec.values())}
        print(f"== {group.name}: masks removed {len(node_r - node_c)}, added {len(node_c - node_r)}; "
              f"links removed {len(lr - lc)}, added {len(lc - lr)}")
        for k in sorted(node_r - node_c):
            print("   mask removed:", describe(by_key[k], H))
        for k in sorted(node_c - node_r):
            print("   mask added:  ", describe(by_key[k], H))
        for a, b in sorted(lc - lr):
            print("   link added:  ", describe(by_key[a], H), "->", describe(by_key[b], H))
        for a, b in sorted(lr - lc):
            print("   link removed:", describe(by_key[a], H), "->", describe(by_key[b], H))


if __name__ == "__main__":
    main()
