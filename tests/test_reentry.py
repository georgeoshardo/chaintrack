"""Cells that come back: a cell that leaves past the bottom and returns when the column moves up.

A made-up trench of seven cells of different lengths, so that each is known by its size; the
lowest is always cut off by the bottom of the crop. At frame 10 the column slides down past the
lowest cell's visible part and that cell leaves; at frame 20 it slides back up and the cell is in
view again. Without --reentry the tracker has to explain
the returning mask as a new cell or a daughter of the cell above; with it, the cell comes back
as itself, linked over the frames it was out.

    python -m pytest tests
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from pymmm.store import experiment_group as eg

HERE = Path(__file__).parent
CHAIN = HERE.parent / "chain.py"
RECIPE = HERE / "data" / "recipe.toml"
FRAMES = 30
LENGTHS, GAP, TOP, HEIGHT, WIDTH = (36, 42, 38, 46, 40, 44, 50), 4, 10, 320, 30
SLIDE, OUT, BACK = (
    30,
    10,
    20,
)  # how far the column slides down, and the frames it does so and comes back
TOPS = TOP + np.concatenate([[0], np.cumsum(np.asarray(LENGTHS[:-1]) + GAP)])


def chain(*args):
    """Run chain.py; its output."""
    done = subprocess.run(
        [sys.executable, str(CHAIN), *map(str, args)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    masks = np.zeros((FRAMES, HEIGHT, WIDTH), np.uint16)
    for t in range(FRAMES):
        shift = SLIDE if OUT <= t < BACK else 0
        for cell, (top, length) in enumerate(zip(TOPS + shift, LENGTHS)):
            if top < HEIGHT:
                masks[t, top : min(top + length, HEIGHT), 10:20] = cell + 1
    path = tmp_path_factory.mktemp("reentry") / "test.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {
        "layout": "experiment",
        "pixel_size_um": 0.1,
        "frame_interval_ms": 60_000.0,
    }
    root.create_array("C", shape=(1,), dtype=str)[:] = np.asarray(["PC"], dtype=object)
    root.create_group("trenches").create_array(
        "trench", data=np.asarray([0], dtype=np.int32)
    )
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["h0"]})
    root.create_group(eg.trench_path(0)).create_array(
        "0", data=np.zeros((FRAMES, 1, HEIGHT, WIDTH), np.float32)
    )
    eg.write_segmentation(root, 0, "seg", {"h0": masks})
    on = ("--store", path, "--seg", "seg")
    chain("fit", 0, *on, "--recipe", RECIPE)
    chain("run", 0, *on, "--out", "plain")
    chain("fit", 0, *on, "--recipe", RECIPE, "--fit", "back", "--reentry")
    chain("run", 0, *on, "--fit", "back", "--out", "back")
    return path


def graph(path, run):
    """The run's nodes as {id: (frame, y)} and its links as {child: parent}."""
    g = zarr.open_group(
        str(Path(path) / eg.trench_path(0) / "tracks" / f"{run}.geff"), mode="r"
    )
    ids = np.asarray(g["nodes/ids"][:])
    t = np.asarray(g["nodes/props/t/values"][:])
    y = np.asarray(g["nodes/props/y/values"][:])
    nodes = {int(i): (int(a), float(b)) for i, a, b in zip(ids, t, y)}
    edges = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    return nodes, {int(b): int(a) for a, b in edges}


def lowest(nodes, frame):
    return max(
        (n for n, (t, _) in nodes.items() if t == frame), key=lambda n: nodes[n][1]
    )


def test_without_reentry_the_returning_cell_is_not_the_one_that_left(store):
    nodes, parent = graph(store, "plain")
    back = lowest(nodes, BACK)
    assert parent.get(back) != lowest(nodes, OUT - 1)


def test_with_reentry_the_cell_comes_back_as_itself(store):
    nodes, parent = graph(store, "back")
    back = lowest(nodes, BACK)
    assert parent.get(back) == lowest(nodes, OUT - 1)
    # nothing new and no division: every node after the first frame has a parent, and no
    # parent has two children
    assert all(n in parent for n, (t, _) in nodes.items() if t > 0)
    children = list(parent.values())
    assert len(children) == len(set(children))


def test_the_fit_records_reentry_and_a_run_takes_no_reentry(store):
    fits = eg.records(eg.open_experiment(store), "fits")
    assert fits["back"]["reentry"]["recipe"]["p_return"] > 0
    assert "reentry" not in fits["seg"]
    assert fits["back"]["values"] == fits["seg"]["values"]
    done = subprocess.run(
        [
            sys.executable,
            str(CHAIN),
            "run",
            "0",
            "--store",
            str(store),
            "--seg",
            "seg",
            "--reentry",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0 and "takes no --reentry" in done.stderr
