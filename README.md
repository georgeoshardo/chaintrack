# chaintrack

A mother-machine trench tracked as an ordered chain of cells, judged by size and spacing, with a
beam of lineage histories. Masks only (no image). Rates are fitted per segmentation without labels.
The model is described in the docstring at the top of `chain.py`.

Run from the phage-signatures repo, whose pixi `seg` environment it uses:

    pixi run -e seg python external/chaintrack/chain.py fit 2-9 --seg pc_affinity_e250
    pixi run -e seg python external/chaintrack/chain.py run 0-9 --seg pc_affinity_e250
    pixi run curate 0 "--experiment external/chaintrack/experiments/pc_affinity_e250.toml"

- `params/<segmentation>.json`: the fitted rates (written by `fit`, read by `run`).
- `run` writes the collection `chain_S1` into the experiment's tracks store and copies it into the
  curation store. A trench already edited there is never replaced.
- `tools/evaluate.py` scores runs against a curated collection, and `tools/frames_links.py` prints
  links frame by frame.
