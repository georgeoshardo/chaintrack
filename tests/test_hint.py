"""Intensity hints: a channel's mean inside each mask, carried along each cell.

A made-up trench of six cells of equal length and different brightness. At frame 30 the third
cell lyses and the three below it move up by exactly one place. By size and spacing alone that
looks like five cells staying put and the bottom one vanishing; the brightness of the masks says
which cells they are.

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
FRAMES, LYSIS, CELLS = 45, 30, 6
LENGTH, PITCH, TOP, HEIGHT, WIDTH = 40, 44, 10, 300, 30
LEVELS = 100.0 + 60.0 * np.arange(CELLS)  # each cell's brightness in the channel "F"


def chain(*args):
    """Run chain.py; its output."""
    done = subprocess.run(
        [sys.executable, str(CHAIN), *map(str, args)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def place(t, cell):
    """Where a cell's mask is at frame t (its place counted from the closed end), or None once lysed."""
    if t < LYSIS or cell < 2:
        return cell
    return None if cell == 2 else cell - 1


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """The made-up trench as an experiment group: masks, and an image with the channels PC (empty),
    F (each cell at its own level) and flat (the same level everywhere)."""
    rng = np.random.default_rng(0)
    masks = np.zeros((FRAMES, HEIGHT, WIDTH), np.uint16)
    pixels = np.zeros((FRAMES, 3, HEIGHT, WIDTH), np.float32)
    pixels[:, 2] = 50.0 + rng.normal(0, 1, (FRAMES, HEIGHT, WIDTH))
    for t in range(FRAMES):
        for cell in range(CELLS):
            at = place(t, cell)
            if at is None:
                continue
            rows = slice(TOP + at * PITCH, TOP + at * PITCH + LENGTH)
            masks[t, rows, 10:20] = at + 1
            pixels[t, 1, rows, 10:20] = LEVELS[cell] + rng.normal(0, 2)
    path = tmp_path_factory.mktemp("hint") / "test.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {
        "layout": "experiment",
        "pixel_size_um": 0.1,
        "frame_interval_ms": 60_000.0,
    }
    root.create_array("C", shape=(3,), dtype=str)[:] = np.asarray(
        ["PC", "F", "flat"], dtype=object
    )
    root.create_group("trenches").create_array(
        "trench", data=np.asarray([0], dtype=np.int32)
    )
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["h0"]})
    root.create_group(eg.trench_path(0)).create_array("0", data=pixels)
    eg.write_segmentation(root, 0, "seg", {"h0": masks})
    on = ("--store", path, "--seg", "seg")
    chain("fit", 0, *on, "--recipe", RECIPE)
    chain("run", 0, *on, "--out", "plain")
    for channel in ("F", "flat"):
        chain("fit", 0, *on, "--recipe", RECIPE, "--fit", channel, "--hint", channel)
        chain("run", 0, *on, "--fit", channel, "--out", channel)
    return path


def tracks(path, run):
    """The run's cells as {(frame, place): track id}, and its links as pairs of those keys."""
    graph = zarr.open_group(
        str(Path(path) / eg.trench_path(0) / "tracks" / f"{run}.geff"), mode="r"
    )
    t = np.asarray(graph["nodes/props/t/values"][:])
    y = np.asarray(graph["nodes/props/y/values"][:])
    ids = np.asarray(graph["nodes/ids"][:])
    key = {
        int(i): (int(tt), round((yy - TOP - LENGTH / 2) / PITCH))
        for i, tt, yy in zip(ids, t, y)
    }
    edges = np.asarray(graph["edges/ids"][:]).reshape(-1, 2)
    return {(key[int(a)], key[int(b)]) for a, b in edges}


def true_links():
    out = set()
    for cell in range(CELLS):
        for t in range(FRAMES - 1):
            a, b = place(t, cell), place(t + 1, cell)
            if a is not None and b is not None:
                out.add(((t, a), (t + 1, b)))
    return out


def test_the_masks_alone_follow_the_wrong_cells(store):
    """Without a hint the tracker keeps the top five cells in place and loses the bottom one."""
    links = tracks(store, "plain")
    assert ((LYSIS - 1, 2), (LYSIS, 2)) in links
    assert links != true_links()


def test_a_hint_follows_the_right_cells(store):
    assert tracks(store, "F") == true_links()


def test_a_flat_channel_changes_nothing(store):
    """A channel in which every cell looks alike costs every choice the same."""
    assert tracks(store, "flat") == tracks(store, "plain")


def test_the_fit_records_the_hint_and_a_run_takes_no_hint(store):
    fit = eg.records(eg.open_experiment(store), "fits")["F"]
    assert fit["hint"]["channels"] == ["F"]
    assert set(fit["hint"]["values"]["F"]) == {"dark", "sigma0", "kappa", "jump_scale"}
    assert "hint" not in eg.records(eg.open_experiment(store), "fits")["seg"]
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
            "--hint",
            "F",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0 and "takes no --hint" in done.stderr
