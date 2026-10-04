"""Density rho and checkpoint rule: nearest trained size (larger on a tie), then nearest class in rho."""
import math

import numpy as np

from connpomo.data import load_instances

SIZES = [14, 20, 30, 50, 100]
CLASSES = {14: ["easy", "medium", "hard"], 20: ["easy", "medium", "hard"], 30: ["easy", "medium", "hard"],
           50: ["easy", "medium", "hard"], 100: ["hard"]}
RHO = {(14, "easy"): 1.996726632118225, (14, "medium"): 1.2130757570266724, (14, "hard"): 0.8252080678939819,
       (20, "easy"): 2.514096736907959, (20, "medium"): 1.5135890245437622, (20, "hard"): 0.8978917002677917,
       (30, "easy"): 3.310758113861084, (30, "medium"): 1.8895971775054932, (30, "hard"): 1.0103795528411865,
       (50, "easy"): 4.8179731369018555, (50, "medium"): 2.5769143104553223, (50, "hard"): 1.2573734521865845,
       (100, "hard"): 1.9247106313705444}


def rho_of(conn):
    """Average degree divided by log N + log log N."""
    N = conn.shape[0]
    d = conn.sum() / N
    return d / (math.log(N) + math.log(math.log(N)))


def class_rho(benchmark_dir):
    """Mean rho of each benchmark class (the values in RHO)."""
    return {(N, h): float(np.mean([rho_of(c) for _, c, _ in load_instances(f"{benchmark_dir}/N{N}_{h}.npz")]))
            for N, hs in CLASSES.items() for h in hs}


def pick(N, rho, rho_table=RHO):
    """(size, class) of the checkpoint for N stops at density rho."""
    size = min(SIZES, key=lambda s: (abs(s - N), -s))
    h = min(CLASSES[size], key=lambda k: abs(rho_table[(size, k)] - rho))
    return size, h
