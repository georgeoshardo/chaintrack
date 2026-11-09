# chaintrack

A mother-machine trench tracked as an ordered chain of cells, judged by size and spacing, with a
beam of lineage histories. Masks only (no image). Rates are fitted per segmentation without labels.
The model is described in the docstring at the top of `chain.py`.

Run from the phage-signatures repo, whose pixi `seg` environment it uses:

    pixi run -e seg python external/chaintrack/chain.py fit 2-9 --seg pc_affinity_ensemble
    pixi run -e seg python external/chaintrack/chain.py run 0-436 --seg pc_affinity_ensemble
    pixi run curate 0

- `params/<segmentation>.json`: the fitted rates (written by `fit`, read by `run`).
- `run` writes each trench's tracks into the experiment group, beside its pixels:
  `trench_NNNN/tracks/chain_S1.geff` and the masks it chose, `labels/tracks.chain_S1` (`--out` names
  another run). It never writes `curation/`: mm-track copies a trench there at its first edit, so
  re-running replaces only tracker output.
- `--crop-bottom N` removes masks with more than half their area in the bottom N rows, for crops
  that reach the halo at the trench opening (the old 655 px crops took 30). PyMMM's crops stop
  above it, so nothing is removed by default. A run records the value it used.
- `--clear-bottom N` removes every mask reaching the bottom N rows: cells cut off by the crop's edge
  (Omnipose stops such a mask 3-4 px short of the edge, so 6 for PyMMM's crops). Off by default.
- `--beam N`: how many lineage histories are kept each frame (default 40). Wider lets later frames
  overturn more ambiguous calls; 120 takes about 2.3x as long to track.
- At the open end and in lysis waves, these are on by default since 2026-11-09, when they agreed
  best with the Reviewed curations. Runs made before, such as `chain_S1`, had them off. Each flag
  below turns one off:
  - `--no-trunc-bound`: by default a cell cut off at the bottom is no shorter than predicted when
    it shows whole again, so its short upper daughter, or a short mask shared with its sister, is
    not the same cell.
  - `--no-div-one`: by default a cell may divide with only its upper daughter in view, the lower
    one past the bottom. The graph shows the mother with one child; the GEFF's `extra.div_one`
    lists those daughters' first nodes.
  - `--revival`: by default a cell that went unseen comes back near where it was predicted, never
    by a jump, so a lysed cell cannot take a survivor's mask. With `--revival` it may jump.
  - `--side`: side cells (off the centre line) are kept out of the chain and by default not
    written; `--side` writes their tracks.
- `tools/curation_score.py` scores runs against the Reviewed curations in the experiment group,
  tolerating what a curation can't pin down (cells cut off at the bottom, divisions a frame or two
  apart). `tools/evaluate.py` scores runs against a curated collection, and `tools/frames_links.py`
  prints links frame by frame.
