"""Score trackers against the curated trenches 0, 1 and 3 (read-only), on the mCherry masks and
on the PC Omnipose masks (pc_affinity_e250). The curation is drawn on mCherry masks; for the PC
masks its segmentation is moved onto PC with the store's channel shift (channel_shift/mcherry_to_pc,
the model behind segmentation/mcherry_on_pc), so cells match by overlap in the PC frame.
Same metrics as eval3, plus how many curated cells have a raw mask at all (IoU > 0.5), which
bounds what any tracker can score on those masks."""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import zarr
from traccuracy import TrackingGraph, run_metrics
from traccuracy.matchers import IOUMatcher
from traccuracy.metrics import CHOTAMetric, CTCMetrics, CompleteTracks, DivisionMetrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S
from pymmm.channel_shift import ChannelShift
from pymmm.tracks import read_trench_geff

ROOT = Path("/data/phage-signatures")
STORE = ROOT / "20261005_pbs_psc_yfp_ty013.trenches.zarr"
GT = ROOT / "20261005_pbs_psc_yfp_ty013.curation.zarr/trackastra_S1"
TRENCHES = (0, 1, 3)
RUNS = {  # name: (masks, tracks collection)
    "Trackastra / mCherry": ("mcherry", ROOT / "20261005_pbs_psc_yfp_ty013.tracks.zarr/trackastra_S1"),
    "chain / mCherry": ("mcherry", ROOT / "20261005_pbs_psc_yfp_ty013.tracks.zarr/chain_S1"),
    "Trackastra / PC": ("pc_affinity_e250", ROOT / "20261005_pbs_psc_yfp_ty013.pc_affinity_e250.tracks.zarr/trackastra_S1"),
    "chain / PC": ("pc_affinity_e250", ROOT / "20261005_pbs_psc_yfp_ty013.pc_affinity_e250.tracks.zarr/chain_S1"),
}
# extra runs to compare, as name=masks:path arguments
for arg in sys.argv[1:]:
    name, _, spec = arg.partition("=")
    masks, _, path = spec.partition(":")
    RUNS[name] = (masks, Path(path))
RUNS = {k: v for k, v in RUNS.items() if v[1].exists()}  # skip runs not made yet


def tracking_graph(group, name, seg=None):
    g = read_trench_geff(group)
    if seg is None:
        seg = np.asarray(zarr.open_array(str(group / "segmentation"), mode="r")[:])
    G = nx.DiGraph()
    for i in range(len(g["t"])):
        G.add_node(i, t=int(g["t"][i]), y=float(g["y"][i]), x=float(g["x"][i]), segmentation_id=int(g["seg_id"][i]))
    G.add_edges_from((int(a), int(b)) for a, b in g["edges"])
    return TrackingGraph(G, segmentation=seg.astype(np.int32), frame_key="t", label_key="segmentation_id",
                         location_keys=("y", "x"), name=name)


def lysis_ends(g):
    t, sid, seg = g["g"]["t"], g["g"]["seg_id"], g["seg"]
    H = seg.shape[1]
    out = set()
    for i in range(g["n"]):
        if g["children"][i] or t[i] >= t.max():
            continue
        rows = np.flatnonzero((seg[t[i]] == sid[i]).any(1))
        if len(rows) and rows.max() < H - 30:
            out.add(i)
    return out


def coverage(gt_seg, raw):
    """Share of curated cells that some raw mask overlaps with IoU > 0.5."""
    hit = tot = 0
    for t in range(gt_seg.shape[0]):
        a, b = gt_seg[t].ravel().astype(np.int64), raw[t].ravel().astype(np.int64)
        ids = np.unique(a[a > 0])
        tot += len(ids)
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        aa, bb = np.bincount(a), np.bincount(b)
        pairs, inter = np.unique(a[both] * 100000 + b[both], return_counts=True)
        good = {int(p) // 100000 for p, i in zip(pairs, inter) if i / (aa[int(p) // 100000] + bb[int(p) % 100000] - i) > 0.5}
        hit += len(good)
    return hit / tot


def main():
    shift = ChannelShift(STORE)
    store = zarr.open_group(str(STORE), mode="r")
    rows = {}
    for tid in TRENCHES:
        group = GT / f"trench_{tid:04d}"
        seg = np.asarray(zarr.open_array(str(group / "segmentation"), mode="r")[:])
        gt_seg = {"mcherry": seg, "pc_affinity_e250": shift.apply(seg, tid)}
        for name, (masks, coll) in RUNS.items():
            gs = gt_seg[masks]
            res, _ = run_metrics(gt_data=tracking_graph(group, "curated", gs), pred_data=tracking_graph(coll / f"trench_{tid:04d}", name),
                                 matcher=IOUMatcher(iou_threshold=0.5, one_to_one=True),
                                 metrics=[CHOTAMetric(), CTCMetrics(), CompleteTracks(), DivisionMetrics(max_frame_buffer=4)])
            r = {}
            for x in res:
                r.update(x["results"])
            gt, pr = S.load(group), S.load(coll / f"trench_{tid:04d}")
            gt["seg"] = gs
            inv = {v: k for k, v in S.match(gt, pr, 0.5).items()}
            gl, pl = lysis_ends(gt), lysis_ends(pr)
            right = sum(1 for p in pl if p in inv and inv[p] in gl)
            r["lysis"] = f"{right}/{len(gl)} +{len(pl) - right}"
            r["coverage"] = coverage(gs, np.asarray(store[f"segmentation/{masks}/labels"][0, tid]))
            rows[(tid, name)] = r
    names = list(RUNS)
    for tid in TRENCHES:
        print(f"-- trench {tid}" + " " * 20 + "".join(f"{n:>22s}" for n in names))
        print(f"{'curated cells with a mask':32s}" + "".join(f"{rows[(tid, n)]['coverage']:>22.3f}" for n in names))
        for k, label in (("CHOTA", "CHOTA"), ("TRA", "TRA"), ("complete_lineages", "complete lineages")):
            print(f"{label:32s}" + "".join(f"{rows[(tid, n)][k]:>22.4f}" for n in names))
        for fb in (0, 4):
            print(f"{'division F1 buffer ' + str(fb):32s}" + "".join(f"{rows[(tid, n)][f'Frame Buffer {fb}']['Division F1']:>22.4f}" for n in names))
        print(f"{'lysis: found/yours +extra':32s}" + "".join(f"{rows[(tid, n)]['lysis']:>22s}" for n in names))


if __name__ == "__main__":
    main()
