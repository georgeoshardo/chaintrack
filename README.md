# chaintrack

A mother-machine trench tracked as an ordered chain of cells, judged by size and spacing, with a
beam of lineage histories. Masks only (no image). Rates are fitted per store without labels.
The model is described in the docstring at the top of `chain.py`.

Run from the phage-signatures repo, whose pixi `seg` environment it uses:

    S="--store 20261005_pbs_psc_yfp_ty013.zarr --recipe recipes/tracking/chain.toml --seg pc_affinity_ensemble"
    pixi run -e seg python external/chaintrack/chain.py fit 2-9 $S
    pixi run -e seg python external/chaintrack/chain.py run 0-436 $S
    pixi run curate 0

- `--store` is the experiment group to track (one OME-Zarr image per trench), `--seg` the
  segmentation in it. chaintrack reads the store's pixel size and frame interval.
- `--recipe` is a toml of every constant set by hand, in µm and minutes
  (phage-signatures keeps `recipes/tracking/chain.toml`). chaintrack converts it to the store's
  pixels and frames, so one recipe serves every magnification and frame rate. Every key is
  required, and unknown keys are refused. What used to be flags is in the recipe:
  - `beam`: how many lineage histories are kept each frame (40). Wider lets later frames overturn
    more ambiguous calls; 120 takes about 2.3x as long to track.
  - `crop_bottom_um` removes masks with more than half their area in those bottom rows, for crops
    that reach the halo at the trench opening (the old 655 px crops took 30 px). PyMMM's crops
    stop above it, so it is 0.
  - `clear_bottom_um` removes every mask reaching those bottom rows: cells cut off by the crop's
    edge (at 40x Omnipose stops such a mask 3-4 px short of the edge, so 6 px, 0.65 µm, for
    PyMMM's crops). 0 in the recipe.
  - At the open end and in lysis waves, these are on since 2026-11-09, when they agreed best with
    the Reviewed curations. Runs made before, such as `chain_S1`, had them off:
    - `trunc_bound`: a cell cut off at the bottom is no shorter than predicted when it shows whole
      again, so its short upper daughter, or a short mask shared with its sister, is not the same
      cell.
    - `div_one`: a cell may divide with only its upper daughter in view, the lower one past the
      bottom. The graph shows the mother with one child; the GEFF's `extra.div_one` lists those
      daughters' first nodes.
    - `no_revival`: a cell that went unseen comes back near where it was predicted, never by a
      jump, so a lysed cell cannot take a survivor's mask.
  - `write_side`: side cells (off the centre line) are kept out of the chain and not written;
    with it their tracks are written.
- `fit` measures growth, the two noise widths, the length at division and the jump size on the
  store's own trenches and records them in the store, `attrs.pymmm.fits[<fit>]`, in µm and minutes,
  with the trenches used, the recipe and chaintrack's commit (`--fit NAME`; default the
  segmentation's name). `run` reads the fit from the store and stops if there is none.
- `run` writes each trench's tracks into the experiment group, beside its pixels:
  `trench_NNNN/tracks/chain_S1.geff` and the masks it chose, `labels/tracks.chain_S1` (`--out` names
  another run). It never writes `curation/`: mm-track copies a trench there at its first edit, so
  re-running replaces only tracker output. The run's record, `attrs.pymmm.tracks[<out>]`, holds the
  recipe as given, the fit, the parameters as used (pixels and frames), the store's calibration
  and chaintrack's commit.
- `tools/curation_score.py` scores runs against the Reviewed curations in the experiment group,
  tolerating what a curation can't pin down (cells cut off at the bottom, divisions a frame or two
  apart). `tools/evaluate.py` scores runs against a curated collection, and `tools/frames_links.py`
  prints links frame by frame.
