"""ConnPOMO query: checkpoint rule, min(N, 64) starts x 8 copies x 2 draws, shortest rideable rollout."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from connpomo.checkpoint_rule import pick, rho_of
from connpomo.data import dihedral, euclid, generate_batch, load_instances
from connpomo.model import load_model

CKPT = Path(__file__).resolve().parents[1] / "checkpoints"


def solve(model, xy, conn, dist, use_pi=True, decode="sample", return_tour=False):
    """Length of the shortest rideable rollout (None if none), and its route if return_tour."""
    N = xy.shape[0]
    K = min(N, 64)
    CN = torch.tensor(conn).unsqueeze(0).expand(K, N, N).contiguous()
    best, best_tour = None, None
    with torch.no_grad():
        for aug in dihedral(xy):
            XY = torch.tensor(np.ascontiguousarray(aug)).unsqueeze(0).expand(K, N, 2).contiguous()
            for rep in range(2):
                st = torch.arange(K) % N if rep == 0 else torch.randint(0, N, (K,))
                tours, _, _ = model(XY, CN, st, decode_type=decode, use_pi=use_pi)
                for t in tours.numpy():
                    if len(set(t.tolist())) == N and all(conn[t[i], t[i + 1]] > 0 for i in range(N - 1)):
                        L = float(sum(dist[t[i], t[i + 1]] for i in range(N - 1)))
                        if best is None or L < best:
                            best, best_tour = L, t.tolist()
    return (best, best_tour) if return_tour else best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default=None, help="npz with xy, conn and dist")
    p.add_argument("--N", type=int, default=None, help="generate one instance with N stops instead")
    p.add_argument("--density", default="hard", help="easy, medium or hard (dense, medium, sparse)")
    p.add_argument("--ckpt", default=None, help="default: checkpoint rule")
    p.add_argument("--seed", type=int, default=None, help="torch seed, default 1007 + N")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-mask", action="store_true", help="switch the preventative mask off")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if not a.instances and not a.N:
        p.error("give --instances or --N")
    torch.set_num_threads(a.threads)
    if a.instances:
        insts = load_instances(a.instances)
    else:
        xy, conn, _ = generate_batch(1, a.N, a.density, seed=a.seed)
        insts = [(xy[0], conn[0], euclid(xy[0]))]
    N = insts[0][0].shape[0]
    ckpts = [a.ckpt or str(CKPT / "N{}_{}.pt".format(*pick(c.shape[0], rho_of(c)))) for _, c, _ in insts]
    models = {c: load_model(c) for c in sorted(set(ckpts))}
    seed = a.seed if a.seed is not None else 1007 + N
    torch.manual_seed(seed)
    per, routes = [], []
    t0 = time.perf_counter()
    for (xy, conn, dist), ck in zip(insts, ckpts):
        L, tour = solve(models[ck], xy, conn, dist, not a.no_mask, "greedy" if a.greedy else "sample", True)
        per.append(L)
        routes.append(tour)
    sec = (time.perf_counter() - t0) / len(insts)
    for i, (L, tour, ck) in enumerate(zip(per, routes, ckpts)):
        res = "no rideable route" if L is None else f"length {L:.4f}  route {tour}"
        print(f"instance {i} ({Path(ck).stem}): {res}")
    lens = [x for x in per if x is not None]
    mean = float(np.mean(lens)) if lens else None
    print(f"solved {len(lens)}/{len(insts)}  mean length {mean if mean is None else round(mean, 4)}  "
          f"{sec:.3f} s/instance")
    if a.out:
        json.dump(dict(ckpt=ckpts, instances=a.instances, seed=seed, mask=not a.no_mask,
                       decode="greedy" if a.greedy else "sample", solved=len(lens),
                       n=len(insts), length=mean, sec=sec, per_instance=per, routes=routes),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
