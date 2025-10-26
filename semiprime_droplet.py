#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semiprime “Droplet” — Structure, Bands, and Envelopes
------------------------------------------------------

This script visualizes semiprimes c = p*q (p ≤ q, both prime) and the transformed
quantity

    Y = p - b2 / t,

where b2 = 2 * floor( sqrt(2c) ) and t is a constant (default t=3.5).

What you’ll see:
1) A tilted “droplet” envelope in the (c, Y) plane.
2) Internal diagonal/oblique banding.
   - Bands spaced ~ 2/t come from the floor quantization of sqrt(2c).
   - Oblique families come from arithmetic “tracks” with p or q nearly fixed.

This script reproduces the original plot and adds:
- “No-floor” version (b2 = 2*sqrt(2c)): same hull, bands largely vanish.
- Coloring by k = floor(sqrt(2c)) to reveal quantization blocks.
- Coloring by q to reveal oblique families.
- Analytic envelope overlays:
      Upper: Y_max(c) ≈ (1 - 2√2/t) * √c
      Lower tracks for small fixed p:  Y_p(c) ≈ p - (2√2/t) * √c
- Simple CLI flags to adjust limits and sampling.

Usage (basic):
    python semiprime_droplet.py

Optional:
    python semiprime_droplet.py --limit 5000000 --t 3.5 --max-points 120000

Author: (your name or handle)
License: MIT
"""

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---------------------------
# Prime + semiprime generators
# ---------------------------

def sieve_of_eratosthenes(n: int) -> List[int]:
    """Return all primes ≤ n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return [int(x) for x in np.nonzero(sieve)[0]]

def generate_semiprimes(limit: int) -> List[Tuple[int, int, int]]:
    """
    Return list of (c, p, q) with c = p*q < limit, p ≤ q, p,q prime.
    Primes only up to sqrt(limit) are required to enumerate.
    """
    primes = sieve_of_eratosthenes(int(limit**0.5) + 10)
    semiprimes = []
    for i, p in enumerate(primes):
        if p * p >= limit:
            break
        for q in primes[i:]:
            c = p * q
            if c >= limit:
                break
            semiprimes.append((c, p, q))
    semiprimes.sort(key=lambda x: x[0])
    return semiprimes

# ---------------------------
# Core computations
# ---------------------------

def build_dataset(limit: int, t: float) -> np.ndarray:
    """
    Build an array with columns:
        0: c
        1: b2_floor = 2 * floor( sqrt(2c) )
        2: p
        3: q
        4: b = p + q
        5: disc_over_4 = (b2_floor^2 - 4c) // 4  (kept for compatibility)
        6: k = floor( sqrt(2c) )
        7: b2_cont = 2 * sqrt(2c)  (no-floor)
        8: Y_floor = p - b2_floor / t
        9: Y_cont  = p - b2_cont  / t
    """
    semiprimes = generate_semiprimes(limit)
    rows = []
    for c, p, q in semiprimes:
        k = int(math.sqrt(2*c))                       # floor
        b2_floor = 2 * k
        disc_over_4 = (b2_floor*b2_floor - 4*c) // 4  # optional, not essential
        if 0 <= disc_over_4 < c:
            b = p + q
            b2_cont = 2.0 * math.sqrt(2.0*c)
            Y_floor = p - (b2_floor / t)
            Y_cont  = p - (b2_cont  / t)
            rows.append((c, b2_floor, p, q, b, disc_over_4, k, b2_cont, Y_floor, Y_cont))
    data = np.array(rows, dtype=float)
    return data

def maybe_downsample(data: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Randomly downsample large datasets for plotting clarity."""
    if len(data) <= max_points:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=max_points, replace=False)
    return data[idx]

# ---------------------------
# Plot helpers
# ---------------------------

def scatter_basic(x, y, *, c=None, title="", xlabel="", ylabel="", cmap="viridis", s=6, alpha=0.6):
    plt.figure(figsize=(11, 6.5))
    if c is None:
        plt.scatter(x, y, s=s, alpha=alpha, edgecolor="none")
    else:
        plt.scatter(x, y, c=c, s=s, alpha=alpha, edgecolor="none", cmap=cmap)
        cb = plt.colorbar()
        cb.set_label("color scale")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()

def overlay_envelopes(limit: int, t: float, *, p_tracks=(2, 3, 5, 7)):
    """
    Overlay analytic envelopes on the current axes:
        Y_top(c) = (1 - 2√2/t) * √c
        Y_p(c)   = p - (2√2/t) * √c   for each p in p_tracks
    """
    c_vals = np.linspace(1.0, limit*0.999, 1000)
    alpha = 2.0 * math.sqrt(2.0) / t
    y_top = (1.0 - alpha) * np.sqrt(c_vals)

    plt.plot(c_vals, y_top, lw=2.0, label=r"Upper hull $\;Y=(1-\frac{2\sqrt{2}}{t})\sqrt{c}$")

    for p in p_tracks:
        y_p = p - alpha * np.sqrt(c_vals)
        plt.plot(c_vals, y_p, lw=1.3, ls="--", label=fr"Track $p={p}$: $Y=p-\frac{{2\sqrt2}}t\sqrt c$")

    plt.legend(loc="best", frameon=True)

# ---------------------------
# Main visualization recipes
# ---------------------------

def make_plots(data: np.ndarray, limit: int, t: float, max_points: int):
    # Columns for readability
    C   = data[:, 0]
    b2f = data[:, 1]
    P   = data[:, 2]
    Q   = data[:, 3]
    # b = data[:, 4]
    # disc = data[:, 5]
    K   = data[:, 6]
    b2c = data[:, 7]
    Yf  = data[:, 8]
    Yc  = data[:, 9]

    # Optional downsample for clarity
    data_small = maybe_downsample(data, max_points=max_points, seed=1)
    mask = np.isin(C, data_small[:, 0])  # quick way to reuse split arrays
    C_s, P_s, Q_s, K_s, Yf_s, Yc_s = C[mask], P[mask], Q[mask], K[mask], Yf[mask], Yc[mask]

    # 1) Original banded droplet: Y_floor vs c
    scatter_basic(C_s, Yf_s,
                  title=fr"Semiprimes: $c=pq$ vs $Y=p-\frac{{b2}}t$  (t={t}, with floor)",
                  xlabel="c = p*q",
                  ylabel=fr"Y = p - b2/t   (b2 = 2⌊√(2c)⌋)")
    overlay_envelopes(limit, t)
    plt.show()

    # 2) No-floor version: bands should largely disappear, same outer hull
    scatter_basic(C_s, Yc_s,
                  title=fr"No-floor: $c$ vs $Y=p-\frac{{2\sqrt{{2c}}}}t$  (t={t})",
                  xlabel="c = p*q",
                  ylabel=fr"Y = p - (2√(2c))/t")
    overlay_envelopes(limit, t)
    plt.show()

    # 3) Color by k = floor(sqrt(2c)) → reveals quantization bands
    scatter_basic(C_s, Yf_s, c=K_s,
                  title=r"Bands from $k=\lfloor\sqrt{2c}\rfloor$ (color by k)",
                  xlabel="c = p*q",
                  ylabel=fr"Y = p - b2/t   (b2 = 2k)")
    # envelope optional here
    plt.show()

    # 4) Color by q (reveals oblique arithmetic families)
    scatter_basic(C_s, Yf_s, c=Q_s,
                  title=r"Oblique families (color by larger prime $q$)",
                  xlabel="c = p*q",
                  ylabel=fr"Y = p - b2/t   (b2 = 2⌊√(2c)⌋)",
                  cmap="plasma")
    plt.show()

    # 5) For completeness: your original “c vs b2” hover plot (static version)
    scatter_basic(C_s, b2f[mask],
                  title=r"Original baseline: $c$ vs $b2$ (with floor)",
                  xlabel="c = p*q",
                  ylabel=r"b2 = 2⌊√(2c)⌋",
                  s=4)
    plt.show()

# ---------------------------
# CLI + entry point
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Semiprime droplet exploration")
    ap.add_argument("--limit", type=int, default=5_000_000,
                    help="Upper bound on c (plot semiprimes c=p*q < limit)")
    ap.add_argument("--t", type=float, default=3.5,
                    help="Divisor t in Y = p - b2/t")
    ap.add_argument("--max-points", type=int, default=120_000,
                    help="Max points to display per scatter (downsamples if needed)")
    return ap.parse_args()

def main():
    args = parse_args()
    limit = int(args.limit)
    t = float(args.t)
    max_points = int(args.max_points)

    print(f"[info] Generating semiprimes with c < {limit} ...")
    data = build_dataset(limit, t)
    if data.size == 0:
        print("[warn] No data points survived the Δ/4 filter; try increasing --limit.")
        return

    print(f"[info] Points after filter: {len(data):,}")
    print("[info] Making plots ...")
    make_plots(data, limit, t, max_points)
    print("[done]")

if __name__ == "__main__":
    main()
