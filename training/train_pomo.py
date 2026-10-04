"""ConnPOMO training (Algorithm 1): graded violation reward, shared baseline, truncated log-probability."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from connpomo.data import generate_batch
from connpomo.model import ConnPOMO, arc_violations, tour_length

LAMBDA = 2.0


def reward_fn(tour, xy, conn, feasible):
    """Graded violation reward, route length and rideable flag."""
    length = tour_length(tour, xy)
    viol = arc_violations(tour, conn)
    return -length - LAMBDA * viol, length, feasible & (viol == 0)


def save_atomic(model, path):
    tmp = str(path) + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def save_resume(path, model, opt, ep, best, hist):
    tmp = str(path) + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, best=best, hist=hist,
                    torch_rng=torch.get_rng_state(), np_rng=np.random.get_state()), tmp)
    os.replace(tmp, path)


def load_resume(path, model, opt, device):
    if not Path(path).exists():
        return 0, (-1.0, -1e18), []
    st = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["torch_rng"].cpu()); np.random.set_state(st["np_rng"])
    print(f"  resuming after epoch {st['epoch']}", flush=True)
    return st["epoch"], tuple(st["best"]), st["hist"]


def epoch(model, opt, N, hardness, n_inst, K, device, batch):
    """One epoch of n_inst // batch policy-gradient steps."""
    model.train()
    dev = next(model.parameters()).device
    tot_r = torch.zeros((), device=dev)
    tot_f = torch.zeros((), device=dev)
    tot_l = torch.zeros((), device=dev)
    nb = 0
    for _ in range(max(1, n_inst // batch)):
        xy_np, conn_np, _ = generate_batch(batch, N, hardness)
        xy = torch.from_numpy(xy_np).to(device).repeat_interleave(K, 0).contiguous()
        conn = torch.from_numpy(conn_np).to(device).repeat_interleave(K, 0).contiguous()
        starts = torch.stack([torch.randperm(N)[:K] for _ in range(batch)]).reshape(-1).to(device)
        tour, logp, feas = model(xy, conn, starts, decode_type="sample", use_pi=True)
        reward, length, full = reward_fn(tour, xy, conn, feas)
        rb = reward.view(batch, K)
        adv = (rb - rb.mean(dim=1, keepdim=True)).reshape(-1).detach()     # shared baseline
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot_r += reward.mean().detach(); tot_f += full.float().mean().detach()
        tot_l += length.mean().detach(); nb += 1
    return float(tot_r) / nb, float(tot_f) / nb, float(tot_l) / nb


def build_valset(N, hardness, n_inst, seed):
    """Fixed validation instances seeded with seed + i."""
    return [generate_batch(1, N, hardness, seed=seed + i) for i in range(n_inst)]


def validate(model, n_inst, K, device, valset):
    """Share of solved validation instances and mean shortest rideable length."""
    model.eval(); solved = 0; lens = []
    for xy_np, conn_np, _ in valset:
        N = xy_np.shape[1]
        xy = torch.from_numpy(xy_np).to(device).expand(K, N, 2).contiguous()
        conn = torch.from_numpy(conn_np).to(device).expand(K, N, N).contiguous()
        starts = torch.arange(min(K, N)) % N
        with torch.no_grad():
            tour, _, feas = model(xy, conn, starts.to(device), decode_type="sample", use_pi=True)
            _, length, full = reward_fn(tour, xy, conn, feas)
        if bool(full.any()):
            solved += 1; lens.append(float(length[full].min()))
    return solved / n_inst, (float(np.mean(lens)) if lens else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--hardness", default="hard", help="easy, medium or hard (dense, medium, sparse)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--inst", type=int, default=99968, help="instances per epoch")
    p.add_argument("--K", type=int, default=24, help="starts per instance")
    p.add_argument("--batch", type=int, default=128, help="instances per gradient step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--val-seed", type=int, required=True)
    p.add_argument("--val-inst", type=int, default=128)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--patience", type=int, default=0, help="epochs without a new best, 0 = off")
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.K = min(a.K, a.N)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    model = ConnPOMO(use_dyn=True).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    valset = build_valset(a.N, a.hardness, a.val_inst, a.val_seed)
    res = out / "resume.pt"
    start_ep, best, hist = load_resume(res, model, opt, a.device)
    best_ep = max((e["epoch"] for e in hist if e.get("new_best")), default=start_ep)
    t0 = time.time()
    vs, vl = (hist[-1]["val_solved"], hist[-1]["val_len"]) if hist else (0.0, float("nan"))
    for ep in range(start_ep + 1, a.epochs + 1):
        te = time.time()
        r, f, l = epoch(model, opt, a.N, a.hardness, a.inst, a.K, a.device, a.batch)
        train_s = time.time() - te
        if ep % a.val_every == 0 or ep == a.epochs:
            vs, vl = validate(model, a.val_inst, min(a.K * 2, a.N), a.device, valset)
        hist.append(dict(epoch=ep, reward=r, rollout_feas=f, val_solved=vs, val_len=vl, train_s=train_s,
                         elapsed_s=time.time() - t0))
        score = (vs, -(vl if vl == vl else 1e9))          # solved, then length
        new_best = score > best
        if new_best:
            best = score; best_ep = ep; save_atomic(model, out / "best.pt")
        hist[-1]["new_best"] = bool(new_best)
        if ep == 1 or ep % 5 == 0 or ep == a.epochs:
            print(f"  epoch {ep:3d}  rollout feas {f:.3f}  instances solved {vs:.3f}  "
                  f"len {vl:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        save_resume(res, model, opt, ep, best, hist)
        if a.patience and ep >= a.min_epochs and ep - best_ep >= a.patience:
            print(f"  stopping at epoch {ep}: no new best since epoch {best_ep}, "
                  f"patience {a.patience}", flush=True)
            hist[-1]["stopped_early"] = True
            json.dump(hist, open(out / "history.json", "w"), indent=1)
            break
    save_atomic(model, out / "final.pt")
    json.dump(dict(N=a.N, hardness=a.hardness, epochs=a.epochs, inst=a.inst, K=a.K, batch=a.batch, lr=a.lr,
                   wd=a.wd, seed=a.seed, val_seed=a.val_seed, val_inst=a.val_inst, device=a.device,
                   total_s=time.time() - t0, instances_seen=a.epochs * a.inst),
              open(out / "config.json", "w"), indent=1)
    print(f"saved {out}/best.pt and final.pt, best solved {best[0]:.3f} len {-best[1]:.3f}, "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
