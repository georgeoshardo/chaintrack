"""Alternative segmentations for chaintrack: the PC channel segmented by saved checkpoints of the
fine-tuned Omnipose model, with segment_pc.py's settings (affinity method, mask threshold 0,
flow threshold 0.4, min area 1.75 um^2). Each trench of the experiment group PyMMM extracts
(<experiment>.zarr, one OME-Zarr image per trench) gets one (t, y, x) label image per
checkpoint, labels/<name>.<hypothesis> (see pymmm.trench_images); a trench's segmentation
is finished once all of them are listed in its labels/.
The settings and hypotheses are recorded once, in the experiment group's
attrs.pymmm.segmentations[<name>]. Only the given trenches are segmented.

The checkpoints segmented, and so the hypotheses, are every --every-th in epoch order from
--min-epoch on (e050, e150, ..., e950, e992, ..., e998), with the base: chaintrack uses every
hypothesis. Earlier checkpoints and the untuned model add noise fragments; every other
checkpoint tracks the curated trenches exactly as all 28 do. A segmentation keeps the
hypotheses it was started with; asking it for others is an error.

Building masks is the slow part, so trenches go to worker processes, each with the checkpoints
loaded once: --local N here, and --remote HOST:N on another machine that has the seg
environment, this file and the checkpoints under ~/phage-seg (the same layout as this repo).
Frames and labels pass through a folder both machines see (--shared here, --remote-shared
there). Trenches go out in order, so they finish roughly in order; this process alone writes
the store.

    pixi run -e seg python external/chaintrack/tools/candidates.py --trenches 10-492 --local 6 \
        --remote georgeos@192.168.1.106:4 --shared /data/cluster/phage-seg/work --remote-shared /cluster/phage-seg/work
"""
import argparse
import datetime
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]   # phage-signatures, or ~/phage-seg on another machine
sys.path.insert(0, str(REPO / "cellpose_finetune"))
from cpseg.omni_masks import build  # noqa: E402
from cpseg.segment_pc import MIN_AREA_UM2, trench_list  # noqa: E402

BASE = "omni_epoch_250"
MODELS = REPO / "cellpose_finetune" / "runs" / "omni" / "ft1" / "models"
REMOTE_REPO = "~/phage-seg"


def checkpoints(models, min_epoch, every):
    """(hypothesis, checkpoint) pairs to segment, the base first."""
    names = [p.name for p in models.iterdir() if p.name.startswith("omni_epoch_")]
    epochs = sorted(int(n.rsplit("_", 1)[1]) for n in names)
    chosen = [e for e in epochs if e >= min_epoch][::every]
    base = int(BASE.rsplit("_", 1)[1])
    return [(f"e{e:03d}", f"omni_epoch_{e}") for e in [base] + [e for e in chosen if e != base]]


def serve(a):
    """Worker: segment the trenches named on stdin ("<trench> <frames.npy> <labels.npz> <lower> <upper>")
    under every checkpoint given, and answer "done <trench>" (or "fail <trench> <error>")."""
    import torch
    from cpseg.omni_watch import load_model
    from pymmm.segmenter import normalise, order_from_closed_end

    torch.set_num_threads(a.threads)
    pairs = [c.split("=") for c in a.checkpoints.split(",")]
    models = [load_model(str(MODELS / ck)) for _, ck in pairs]
    print("ready", flush=True)
    for line in sys.stdin:
        ti, src, dst, lo, hi = line.split()
        try:
            x = normalise(np.load(src), float(lo), float(hi)).astype(np.float32)
            out = np.zeros((len(models),) + x.shape, np.uint16)
            for h, model in enumerate(models):
                lab = np.zeros(x.shape, np.uint16)
                for s in range(0, len(x), a.batch):
                    r = model.eval(x[s:s + a.batch], normalize=False, compute_masks=False, show_progress=False,
                                   batch_size=a.batch)
                    for k, f in enumerate(r.flows):
                        lab[s + k] = build(np.asarray(f.dP), np.asarray(f.dist), "affinity", model.device,
                                           min_size=a.min_size)
                out[h] = order_from_closed_end(lab).astype(np.uint16)
            np.savez_compressed(dst + ".part.npz", labels=out)
            Path(dst + ".part.npz").rename(dst)
            print("done", ti, flush=True)
        except Exception as e:  # noqa: BLE001 - reported to the dispatcher, which retries elsewhere
            print("fail", ti, repr(e).replace("\n", " "), flush=True)


class Slot(threading.Thread):
    """One worker process (here or over ssh), fed trenches until none are left."""

    def __init__(self, name, cmd, shared, seen_as, d):
        super().__init__(daemon=True)
        self.name, self.cmd, self.shared, self.seen_as, self.d = name, cmd, Path(shared), seen_as, d

    def run(self):
        d = self.d
        proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if not self.wait_for("ready", proc):
            print(f"{self.name}: worker did not start", flush=True)
            return
        while (ti := d.next_job()) is not None:
            frames = d.frames(ti)
            lo, hi = d.bounds(ti, frames)
            src, dst = self.shared / f"trench_{ti:04d}.npy", self.shared / f"trench_{ti:04d}.labels.npz"
            np.save(src, frames)
            there = lambda p: f"{self.seen_as}/{p.name}"
            proc.stdin.write(f"{ti} {there(src)} {there(dst)} {lo!r} {hi!r}\n")
            reply = self.wait_for(("done", "fail"), proc)
            src.unlink(missing_ok=True)
            if reply and reply.startswith("done"):
                d.finish(ti, np.load(dst)["labels"], self.name)
                dst.unlink()
            else:
                print(f"{self.name}: trench {ti} failed ({reply or 'worker exited'}); back in the queue", flush=True)
                d.retry(ti)
                if not reply:
                    return
        proc.stdin.close()
        proc.wait()

    @staticmethod
    def wait_for(words, proc):
        for line in proc.stdout:
            if line.startswith(words):
                return line.strip()
        return None


class Dispatcher:
    """Hands out trenches and writes their labels (the only writer of the experiment group)."""

    def __init__(self, root, name, c, hyps, settings, jobs):
        self.root, self.name, self.c, self.hyps, self.settings = root, name, c, hyps, settings
        self.jobs, self.left, self.lock = list(jobs), len(jobs), threading.Lock()
        self.t0, self.n, self.failures, self.given_up = time.time(), 0, {}, []
        self._bounds = {}

    def frames(self, ti):
        from pymmm.trench_images import image

        return np.asarray(image(self.root, ti)[:, self.c])

    def bounds(self, ti, frames):
        """The trench's (lower, upper) normalisation bounds, from its own frames."""
        from pymmm.segmenter import trench_bounds

        with self.lock:
            if ti not in self._bounds:
                self._bounds[ti] = tuple(float(v) for v in trench_bounds(frames, self.settings["floor"]))
            return self._bounds[ti]

    def next_job(self):
        with self.lock:
            return self.jobs.pop(0) if self.jobs else None

    def retry(self, ti):
        with self.lock:
            self.failures[ti] = self.failures.get(ti, 0) + 1
            if self.failures[ti] < 3:
                self.jobs.insert(0, ti)
            else:
                self.given_up.append(ti)

    def finish(self, ti, labels, who):
        from pymmm.trench_images import write_segmentation

        with self.lock:
            lo, hi = self._bounds.pop(ti)
            # the worker's (hypothesis, t, y, x), one label image per hypothesis
            write_segmentation(self.root, ti, self.name, dict(zip(self.hyps, labels)), attrs={
                                  "channel": self.settings["channel"], "lower": lo, "upper": hi,
                                  "long_name": "cell labels; 0 is background, 1..n are ordered from "
                                               "the trench's closed end (y = 0)"})
            self.n += 1
            el = time.time() - self.t0
            print(f"trench {ti} done by {who} ({self.n}/{self.left}), {el / 60:.0f} min, "
                  f"about {el / self.n * (self.left - self.n) / 60:.0f} min left", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="dispatch", choices=["dispatch", "serve"])
    ap.add_argument("--name", default="pc_affinity_ensemble")
    ap.add_argument("--store", default="/data/phage-signatures/20261005_pbs_psc_yfp_ty013.zarr",
                    help="the experiment group (one OME-Zarr image per trench)")
    ap.add_argument("--trenches")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-epoch", type=int, default=50)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--local", type=int, default=1)
    ap.add_argument("--remote", action="append", default=[], help="HOST:N, N workers on HOST")
    ap.add_argument("--shared", default="/data/cluster/phage-seg/work")
    ap.add_argument("--remote-shared", default="/cluster/phage-seg/work")
    ap.add_argument("--threads", type=int, default=2)
    # serve only
    ap.add_argument("--checkpoints")
    ap.add_argument("--min-size", type=int)
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a)

    from pymmm.provenance import provenance
    from pymmm.trench_images import calibration, channel_index, has_segmentation, open_experiment
    from pymmm.trench_images import trench_ids as all_trenches
    from pymmm.trench_images import update_experiment

    root = open_experiment(a.store, mode="r+")
    use = checkpoints(MODELS, a.min_epoch, a.every)
    px, _ = calibration(root)
    min_size = max(1, int(round(MIN_AREA_UM2 / px ** 2)))
    settings = {"checkpoints": dict(use), "method": "affinity", "channel": "PC", "floor": 0.0,
                "mask_threshold": 0.0, "flow_threshold": 0.4, "affinity_angle_deg": 60,
                "min_area_um2": MIN_AREA_UM2, "min_size_px": min_size}
    known = (dict(root.attrs)["pymmm"].get("segmentations") or {}).get(a.name)
    if known is None:
        update_experiment(root, "segmentations", a.name, {
            "description": ("Alternative PC segmentations: one hypothesis per fine-tuned Omnipose checkpoint "
                            f"({BASE} first), affinity method; each trench's labels/{a.name}.<hypothesis>."),
            "channel": "PC", "hypotheses": [hyp for hyp, _ in use], "params": settings,
            "provenance": provenance(), "made_by": "phage-signatures/external/chaintrack/tools/candidates.py",
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds")})
        hyps = [hyp for hyp, _ in use]
    else:
        hyps = list(known["hypotheses"])
        if [hyp for hyp, _ in use] != hyps:
            raise SystemExit(f"{a.name} has the hypotheses {hyps}, not {[h for h, _ in use]}; "
                             "segment others under a new --name")
    ids = set(all_trenches(root))
    wanted = trench_list(a.trenches) if a.trenches else sorted(ids)
    unknown = [t for t in wanted if t not in ids]
    if unknown:
        raise SystemExit(f"no trenches {unknown} in {a.store}")
    jobs = [t for t in wanted if not has_segmentation(root, t, a.name)]
    Path(a.shared).mkdir(parents=True, exist_ok=True)
    serve_args = ["serve", "--checkpoints", ",".join(f"{h}={c}" for h, c in use), "--min-size", str(min_size),
                  "--batch", str(a.batch), "--threads", str(a.threads)]
    d = Dispatcher(root, a.name, channel_index(root, "PC"), hyps, settings, jobs)
    slots = [Slot(f"local{k}", [sys.executable, __file__] + serve_args, a.shared, a.shared, d) for k in range(a.local)]
    for spec in a.remote:
        host, _, n = spec.rpartition(":")
        cmd = (f"cd {REMOTE_REPO} && ~/.pixi/bin/pixi run --frozen -e seg python external/chaintrack/tools/candidates.py "
               + " ".join(serve_args))
        slots += [Slot(f"{host.split('@')[-1]}#{k}", ["ssh", "-o", "BatchMode=yes", host, cmd], a.shared,
                       a.remote_shared, d) for k in range(int(n))]
    print(f"{len(jobs)} trenches to segment under {len(use)} checkpoints ({', '.join(h for h, _ in use)}), "
          f"{len(slots)} workers", flush=True)
    for s in slots:
        s.start()
    for s in slots:
        s.join()
    if d.jobs or d.given_up:
        print(f"unfinished: {sorted(d.jobs + d.given_up)}", flush=True)


if __name__ == "__main__":
    main()
