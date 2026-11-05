"""Two runs against one curation (read-only): per trench, the links the curation disagrees with
in the old run that the new run fixes, the ones both still get wrong, and the new run's new
disagreements. Links are (frame, mask top) pairs, so runs on the same masks compare directly.

    pixi run -e seg python external/chaintrack/tools/compare_runs.py <old run> <new run> <curated> 0,1,3
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S


def links(group):
    g = S.load(group)
    t, sid = g["g"]["t"], g["g"]["seg_id"]
    rows = {i: np.flatnonzero((g["seg"][t[i]] == sid[i]).any(1)) for i in range(g["n"])}
    key = lambda i: (int(t[i]), int(rows[i].min()), int(rows[i].max()))
    return {(key(p), key(c)) for p, c in g["g"]["edges"]}


old, new, cur = (Path(a) for a in sys.argv[1:4])
tot = np.zeros(3, int)
for tid in (int(x) for x in sys.argv[4].split(",")):
    name = f"trench_{tid:04d}"
    c, o, n = links(cur / name), links(old / name), links(new / name)
    d_old, d_new = o ^ c, n ^ c          # links in one but not the other: each is a disagreement
    fixed, kept, broke = d_old - d_new, d_old & d_new, d_new - d_old
    tot += (len(fixed), len(kept), len(broke))
    frames = lambda s: sorted({a[0] for a, b in s})
    print(f"{name}: fixed {len(fixed)} (frames {frames(fixed)}), still wrong {len(kept)} (frames {frames(kept)}), "
          f"new {len(broke)} (frames {frames(broke)})")
print(f"all: fixed {tot[0]}, still wrong {tot[1]}, new {tot[2]}")
