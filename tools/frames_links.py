"""Print, for one trench and a frame range, every mask's rows and the rows of the mask it is
linked from, plus the masks that end that frame (read-only)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S

group, f0, f1 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
g = S.load(group)
t, sid = g["g"]["t"], g["g"]["seg_id"]
ext = {i: (lambda r: (int(r.min()), int(r.max())))(np.flatnonzero((g["seg"][t[i]] == sid[i]).any(1))) for i in range(g["n"]) if f0 - 4 <= t[i] <= f1}
parent = {c: p for p, c in g["g"]["edges"]}
for f in range(f0, f1 + 1):
    cells = sorted((i for i in ext if t[i] == f), key=lambda i: ext[i][0])
    src = lambda i: "new" if i not in parent else f"{ext[parent[i]][0]}-{ext[parent[i]][1]}" + (f"@t{t[parent[i]]}" if t[parent[i]] != f - 1 else "")
    links = " ".join(f"{ext[i][0]}-{ext[i][1]}<{src(i)}" for i in cells)
    ends = [f"{ext[i][0]}-{ext[i][1]}" for i in ext if t[i] == f - 1 and not g["children"][i]]
    print(f"f{f}: {links}   | ended at f{f - 1}: {' '.join(sorted(ends))}")
