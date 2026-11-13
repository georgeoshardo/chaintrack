"""chaintrack on real masks gives the tracks it gave before.

tests/data/masks.npz holds two trenches' masks under 14 hypotheses (tests/make_data.py says
where they came from), tests/data/recipe.toml the recipe, and tests/data/expected.npz the fit
and the tracks chaintrack gave for them. The test builds an experiment group from the masks,
runs `chain.py fit` and `chain.py run` on it as a user would, and compares.

    python -m pytest tests                      # in an environment with PyMMM, numba and geff
    CHAINTRACK_UPDATE=1 python -m pytest tests  # after an intended change: rewrite expected.npz

Review what changed before committing a rewritten expected.npz: it is the only record of what
the tracks should be.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from pymmm.store import experiment_group as eg

HERE = Path(__file__).parent
CHAIN = HERE.parent / "chain.py"
DATA = HERE / "data"
NODE_PROPS = ("t", "seg_id", "track_id", "lineage_id", "y", "x")


def chain(*args):
    """Run chain.py; its output."""
    done = subprocess.run([sys.executable, str(CHAIN), *map(str, args)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def read_run(store, trench, run):
    """A trench's tracks as arrays: the nodes' properties, the edges, the masks, the lone daughters."""
    base = Path(store) / eg.trench_path(trench)
    graph = zarr.open_group(str(base / "tracks" / f"{run}.geff"), mode="r")
    out = {k: np.asarray(graph[f"nodes/props/{k}/values"][:]) for k in NODE_PROPS}
    out["ids"] = np.asarray(graph["nodes/ids"][:])
    out["edges"] = np.asarray(graph["edges/ids"][:]).reshape(-1, 2)
    out["masks"] = np.asarray(zarr.open_array(str(base / "labels" / f"tracks.{run}" / "0"), mode="r")[:])
    out["div_one"] = np.asarray(dict(graph.attrs)["geff"]["extra"]["div_one"], dtype=np.int64)
    return out


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    """An experiment group built from the test masks, fitted and tracked twice: several trenches
    at once (the run `chain`) and one at a time (`one`)."""
    data = np.load(DATA / "masks.npz")
    meta = json.loads(str(data["meta"]))
    store = tmp_path_factory.mktemp("chaintrack") / "test.zarr"
    root = zarr.open_group(str(store), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {"layout": "experiment", "pixel_size_um": meta["pixel_size_um"],
                           "frame_interval_ms": meta["frame_interval_ms"]}
    root.create_array("C", shape=(1,), dtype=str)[:] = np.asarray(["PC"], dtype=object)
    root.create_group("trenches").create_array("trench", data=np.asarray(meta["trenches"], dtype=np.int32))
    eg.update_experiment(root, "segmentations", meta["segmentation"], {"hypotheses": meta["hypotheses"]})
    for trench in meta["trenches"]:
        root.create_group(eg.trench_path(trench))
        eg.write_segmentation(root, trench, meta["segmentation"],
                              {h: data[f"{trench}/{h}"] for h in meta["hypotheses"]})
    on = ("--store", store, "--seg", meta["segmentation"])
    chain("fit", *meta["trenches"], *on, "--recipe", DATA / "recipe.toml")
    chain("run", *meta["trenches"], *on)
    chain("run", *meta["trenches"], *on, "--workers", 1, "--out", "one")
    fit = eg.records(eg.open_experiment(store), "fits")[meta["segmentation"]]
    runs = {t: read_run(store, t, "chain") for t in meta["trenches"]}
    if os.environ.get("CHAINTRACK_UPDATE"):
        np.savez_compressed(DATA / "expected.npz", fit=json.dumps(fit["values"]),
                            **{f"{t}/{k}": v for t, run in runs.items() for k, v in run.items()})
    return store, meta, fit, runs


def test_the_fit_is_the_expected_one(tracked):
    _, meta, fit, _ = tracked
    expected = json.loads(str(np.load(DATA / "expected.npz")["fit"]))
    assert fit["trenches"] == meta["trenches"] and fit["frames"] == "all"
    assert fit["values"] == pytest.approx(expected, rel=1e-9)


def test_the_tracks_are_the_expected_ones(tracked):
    _, meta, _, runs = tracked
    expected = np.load(DATA / "expected.npz")
    for trench, run in runs.items():
        for key, value in run.items():
            want = expected[f"{trench}/{key}"]
            assert value.shape == want.shape, f"trench {trench}: {key} has {value.shape}, expected {want.shape}"
            if key in ("y", "x"):
                np.testing.assert_allclose(value, want, rtol=0, atol=1e-3, err_msg=f"trench {trench}: {key}")
            else:
                np.testing.assert_array_equal(value, want, err_msg=f"trench {trench}: {key}")


def test_one_trench_at_a_time_gives_the_same_tracks(tracked):
    store, meta, _, runs = tracked
    for trench, run in runs.items():
        one = read_run(store, trench, "one")
        for key, value in run.items():
            np.testing.assert_array_equal(one[key], value, err_msg=f"trench {trench}: {key}")


def test_the_run_is_recorded_with_its_fit(tracked):
    store, meta, fit, _ = tracked
    record = eg.records(eg.open_experiment(store), "tracks")["chain"]
    assert record["segmentation"] == meta["segmentation"] and record["alternatives"] == "all"
    assert record["fit"]["name"] == meta["segmentation"] and record["fit"]["created"] == fit["created"]
    assert record["recipe"] == fit["recipe"]
