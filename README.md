# chaintrack

chaintrack tracks the cells of a mother-machine trench as an ordered chain. It judges each cell
by its size and spacing and keeps a beam of lineage histories.[^jug] It works from segmentation
masks alone, with no image, and fits its rates to each experiment without labels. The docstring
at the top of `chain.py` describes the model.

## What it needs

- An experiment group written by [PyMMM](https://github.com/georgeoshardo/PyMMM): one OME-Zarr
  image per trench, already segmented under one or more hypotheses.
- Python with PyMMM installed from its repository with the `tracks` extra (for GEFF),
  plus numba and scipy.

chaintrack is a single script. Clone the repository and run `chain.py`:

    S="--store experiment.zarr --seg <segmentation>"
    python chain.py fit 0-7 $S --recipe tests/data/recipe.toml     # once per experiment, on a few trenches
    python chain.py run 0-99 $S                                     # the trenches to track

PyMMM's `Tracker` runs the same two steps from a notebook; give it the path to `chain.py`.

## The two steps

`fit` measures the experiment's own growth rate, length at division and noise on the trenches
you give it, and records them in the store. It also takes the recipe, a toml of the constants
set by hand, in µm and minutes. `tests/data/recipe.toml` explains each one. chaintrack converts
them with the store's pixel size and frame interval, so one recipe serves any magnification and
frame rate.

`run` tracks trenches with that fit and writes each one's tracks beside its pixels:
`trench_NNNN/tracks/chain.geff` and the masks it chose, `labels/tracks.chain`. `follow` does the
same while a segmentation is still running, taking each trench as it is finished.

`fit` copies the recipe into the store and `run` uses that copy, so editing the file later
cannot change an experiment that is already fitted. To track with other
constants, fit again under another name and track with it:

    python chain.py fit 0-7 $S --recipe other.toml --fit other
    python chain.py run 0-99 $S --fit other --out chain_other

## Options

| | |
|---|---|
| `--store` | the experiment group |
| `--seg` | the segmentation to track: its first hypothesis, with the others as alternatives |
| `--recipe` | the recipe file (`fit` only) |
| `--fit` | the fit's name in the store (default: the segmentation's name) |
| `--out` | the run's name (default: `chain`) |
| `--alts a,b` | use only these hypotheses as alternatives; `--base-only` uses none |
| `--workers` | trenches tracked at once (default: as many as fit in the free memory) |

Give trenches as numbers or ranges: `0 2-5 7`.

The store keeps a record of each fit and each run, with the recipe, the parameters as used and
chaintrack's commit. `fit` will not replace a fit that a run has used, and `run` adds trenches to
an existing run only with the fit and segmentation it was started with.

`tools/curation_score.py` scores runs against curated trenches.

## Licence

GPL v2; see `LICENSE`.

[^jug]: Two ideas here are from Jug et al. (2014), *Optimal Joint Segmentation and
    Tracking of Escherichia coli in the Mother Machine*: cells in a trench keep their
    order, and the tracker chooses among several segmentation hypotheses.
