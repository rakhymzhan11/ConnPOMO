"""Instance generator and benchmark; easy, medium and hard are the dense, medium and sparse classes."""
import argparse
import math
from pathlib import Path

import numpy as np

DENSITY = {"easy": 0.50, "medium": 0.25, "hard": 0.10}
GRID = [(N, h) for N in (14, 20, 30, 50) for h in ("easy", "medium", "hard")] + [(100, "hard")]


def generate_instance(N, hardness, rng):
    """Random stops, a planted Hamiltonian path, other pairs linked with DENSITY[hardness]."""
    xy = rng.random((N, 2), dtype=np.float64).astype(np.float32)
    perm = rng.permutation(N)
    conn = np.zeros((N, N), dtype=np.float32)
    for k in range(N - 1):
        u, v = int(perm[k]), int(perm[k + 1])
        conn[u, v] = 1.0
        conn[v, u] = 1.0
    d = DENSITY[hardness]
    for i in range(N):
        for j in range(i + 1, N):
            if conn[i, j] == 0 and rng.random() < d:
                conn[i, j] = 1.0
                conn[j, i] = 1.0
    return xy, conn, int(perm[0])


def generate_batch(B, N, hardness, seed=None):
    """B instances from one generator; seed None draws fresh entropy."""
    rng = np.random.default_rng(seed)
    xy = np.empty((B, N, 2), dtype=np.float32)
    conn = np.empty((B, N, N), dtype=np.float32)
    start = np.empty((B,), dtype=np.int64)
    for b in range(B):
        x, c, s = generate_instance(N, hardness, rng)
        xy[b], conn[b], start[b] = x, c, s
    return xy, conn, start


def euclid(xy):
    xy = np.asarray(xy, np.float32)
    return np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1)).astype(np.float64)


def unit_square(latlon):
    """Equirectangular projection to km, scaled by the larger extent and centered in the unit square."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    y = (lat - lat.mean()) * 110.574
    x = (lon - lon.mean()) * 111.320 * math.cos(math.radians(lat.mean()))
    s = max(x.max() - x.min(), y.max() - y.min())
    x = (x - x.min()) / s
    y = (y - y.min()) / s
    x += (1 - x.max()) / 2
    y += (1 - y.max()) / 2
    return np.stack([x, y], -1).astype(np.float32)


def dihedral(v):
    """The 8 rotations and reflections of the unit square."""
    x, y = v[..., 0:1], v[..., 1:2]
    return [np.concatenate(t, -1) for t in
            ((x, y), (y, x), (1 - x, y), (y, 1 - x),
             (x, 1 - y), (1 - y, x), (1 - x, 1 - y), (1 - y, 1 - x))]


def load_instances(path):
    """List of (xy, conn, dist) from an npz file."""
    d = np.load(path)
    xy = d["xy"].astype(np.float32)
    conn = (d["conn"] > 0).astype(np.float32)
    dist = d["dist"].astype(np.float64)
    return [(xy[i], conn[i], dist[i]) for i in range(len(xy))]


def write_benchmark(folder, n=20):
    """The 260 benchmark instances, one generator per size seeded with 3100 + N."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for N, h in GRID:
        xy, conn, _ = generate_batch(n, N, h, seed=3100 + N)
        dist = np.stack([euclid(v) for v in xy])
        np.savez_compressed(folder / f"N{N}_{h}.npz", xy=xy, conn=conn.astype(np.uint8), dist=dist)
        print(f"wrote N{N}_{h}.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?", default=Path(__file__).resolve().parents[1] / "data" / "benchmark")
    write_benchmark(p.parse_args().folder)
