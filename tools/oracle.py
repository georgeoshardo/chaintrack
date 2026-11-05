"""What alternative segmentations could fix, at best (read-only). For each curated cell, the best
IoU of a raw mask in the base hypothesis and in any hypothesis; how many curated cells only an
alternative matches well; and, for the masks the curator added or changed relative to a run
(their corrections), how many an alternative provides.

    pixi run -e seg python external/chaintrack/tools/oracle.py <run> <curated> 0,1,3 [ensemble group]
"""
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S

STORE = "/data/phage-signatures/20261005_pbs_psc_yfp_ty013.trenches.zarr"
def best_iou(cur_seg, raw):
    """For each label in cur_seg (one frame), the best IoU with any label in raw."""
    a, b = cur_seg.ravel().astype(np.int64), raw.ravel().astype(np.int64)
    out = {int(l): 0.0 for l in np.unique(a[a > 0])}
    both = (a > 0) & (b > 0)
    if not both.any():
        return out
    aa, bb = np.bincount(a), np.bincount(b)
    pairs, inter = np.unique(a[both] * 100000 + b[both], return_counts=True)
    for p, i in zip(pairs, inter):
        la, lb = divmod(int(p), 100000)
        out[la] = max(out[la], i / (aa[la] + bb[lb] - i))
    return out


def main():
    run, cur = Path(sys.argv[1]), Path(sys.argv[2])
    trenches = [int(x) for x in sys.argv[3].split(",")]
    group = sys.argv[4] if len(sys.argv) > 4 else "pc_affinity_ensemble"
    root = zarr.open_group(STORE, mode="r")
    labels = root[f"segmentation/{group}/labels"]
    hyps = [str(h) for h in root[f"segmentation/{group}/Hypothesis"][:]]
    done = dict(root[f"segmentation/{group}"].attrs["completed"])
    tot = np.zeros(4, int)
    for tid in trenches:
        use = [h for h, name in enumerate(hyps) if tid in done.get(name, [])]
        raw = np.asarray(labels.get_orthogonal_selection((use, tid)))   # (H, T, Y, X), base first
        gc, gr = S.load(cur / f"trench_{tid:04d}"), S.load(run / f"trench_{tid:04d}")
        H = gc["seg"].shape[1]
        n = base_ok = any_ok = only_alt = 0
        fixes, fixable = 0, 0
        for f in range(gc["seg"].shape[0]):
            cs = gc["seg"][f]
            if not cs.any():
                continue
            ious = [best_iou(cs, raw[h, f]) for h in range(len(use))]
            rs = best_iou(cs, gr["seg"][f])                 # how the run's own masks match each curated cell
            for l in ious[0]:
                n += 1
                b = ious[0][l]
                a = max(io[l] for io in ious)
                base_ok += b >= 0.8; any_ok += a >= 0.8; only_alt += (b < 0.8) and (a >= 0.8)
                if rs.get(l, 0) < 0.8:                      # the curator changed this cell's mask
                    fixes += 1; fixable += a >= 0.8
        tot += (n, only_alt, fixes, fixable)
        print(f"trench {tid}: {n} curated cells, {len(use)} hypotheses; matched (IoU >= 0.8) by the base {base_ok}, "
              f"by any {any_ok}, only by an alternative {only_alt}; cells whose mask you changed {fixes}, "
              f"of which an alternative provides {fixable}")
    print(f"all: {tot[0]} cells; only an alternative matches {tot[1]}; masks you changed {tot[2]}, "
          f"an alternative provides {tot[3]}")


if __name__ == "__main__":
    main()
