"""Train the 13 models in the stages grid, big and extend; the README gives the commands."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POMO_BATCH = {14: 128, 20: 128, 30: 128, 50: 64, 100: 24}
POMO_EPOCH_INST = 100_000


def train(tag, N, hardness, a, seed=2026):
    """Train one class with train_pomo.py, halving the batch on out-of-memory."""
    out = Path(a.out) / tag
    if (out / "config.json").exists():
        print(f"skip {tag}, already done", flush=True)
        return
    b = a.batch or POMO_BATCH[N]
    k = min(a.K, N)
    while b >= 1:
        steps = max(1, round(a.epoch_inst / b))
        inst = b * steps
        print(f"=== {tag}: N={N} {hardness}, batch {b}, {inst} instances per epoch, "
              f"{a.epochs} epochs, K={k}", flush=True)
        log = Path(a.out) / f"{tag}.log"
        cmd = [sys.executable, "-u", str(ROOT / "training" / "train_pomo.py"),
               "--N", str(N), "--hardness", hardness, "--epochs", str(a.epochs), "--inst", str(inst),
               "--K", str(k), "--batch", str(b), "--device", a.device, "--lr", str(a.lr), "--wd", str(a.wd),
               "--seed", str(seed), "--val-seed", str(900000 + N), "--val-inst", str(a.val_inst),
               "--val-every", str(a.val_every), "--out", str(out),
               "--patience", str(a.patience), "--min-epochs", str(a.min_epochs)]
        with open(log, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return
        if "out of memory" in log.read_text()[-4000:].lower() and b > 1:
            b //= 2
            print(f"{tag} ran out of memory, retrying at batch {b}", flush=True)
            continue
        sys.exit(f"{tag} failed, see {log}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["grid", "big", "extend", "all"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epoch-inst", type=int, default=POMO_EPOCH_INST)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=0, help="0 uses POMO_BATCH")
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-inst", type=int, default=128)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "runs" / "train"))
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for st in (["big", "extend"] if a.stage == "all" else [a.stage]):
        if st == "grid":
            for h in ("hard", "medium", "easy"):
                for N in (14, 20, 30, 50):
                    train(f"g_{N}_{h}", N, h, a)
        elif st == "big":
            train("g_100_hard", 100, "hard", a)
        elif st == "extend":
            for h in ("medium", "easy"):
                for N in (14, 20, 30, 50):
                    tag = f"g_{N}_{h}"
                    hist_f = Path(a.out) / tag / "history.json"
                    if hist_f.exists():
                        hist = json.loads(hist_f.read_text())
                        if len(hist) >= a.epochs or hist[-1].get("stopped_early"):
                            print(f"skip {tag}, extension already complete", flush=True)
                            continue
                    (Path(a.out) / tag / "config.json").unlink(missing_ok=True)
                    train(tag, N, h, a)


if __name__ == "__main__":
    main()
