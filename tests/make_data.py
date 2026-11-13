"""Make the regression test's masks, tests/data/masks.npz, from an experiment group.

    python tests/make_data.py <experiment>.zarr <segmentation> <frames> <trench> [<trench> ...]

Takes the first <frames> frames of each trench's masks under every hypothesis of the
segmentation, with the store's pixel size and frame interval. The masks in the repository
came from

    python tests/make_data.py 20231220_SB8_TY009_sytox_phage.zarr pc_affinity_ensemble 190 0 110

(100x, 0.065 µm per pixel, a frame every 30 s; in trench 110 every cell lyses around frame 140,
trench 0 only grows in these frames). The images are the published run `exp1` of the lab's
single-cell phage data; the masks are our segmentation of them. After remaking them, remake the expected tracks:
CHAINTRACK_UPDATE=1 python -m pytest tests.
"""

import json
import sys
from pathlib import Path

import numpy as np

from pymmm.store.experiment_group import calibration, open_experiment, segmentation

store, seg, frames, trenches = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    [int(t) for t in sys.argv[4:]],
)
root = open_experiment(store)
px, dt = calibration(root)
arrays, names = {}, None
for t in trenches:
    names, labels = segmentation(root, t, seg)
    for name, lab in zip(names, labels):
        arrays[f"{t}/{name}"] = np.asarray(lab[:frames])
meta = {
    "source": Path(store).name,
    "segmentation": seg,
    "hypotheses": names,
    "trenches": trenches,
    "frames": frames,
    "pixel_size_um": px,
    "frame_interval_ms": dt * 60_000.0,
}
out = Path(__file__).parent / "data" / "masks.npz"
np.savez_compressed(out, meta=json.dumps(meta), **arrays)
print(out, f"{out.stat().st_size / 1e6:.1f} MB", meta)
