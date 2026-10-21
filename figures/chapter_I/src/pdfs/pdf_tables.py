"""x f(x, Q) tables with 68% replica bands from an LHAPDF lhagrid1 set (pure Python, no numpy).

Usage: python pdf_tables.py <set_dir> <out_dir> Q1 [Q2 ...]
Writes, for each Q, one file per combination: x  central  lo  hi  (lo/hi = mean -/+ standard deviation
over the Monte Carlo replicas), plus a closed polygon file for the band.
"""
import glob
import math
import os
import sys

set_dir, out_dir = sys.argv[1], sys.argv[2]
qs = [float(q) for q in sys.argv[3:]]
os.makedirs(out_dir, exist_ok=True)

COMBOS = {  # name: {pid: coefficient}, scale factor applied afterwards
    "uv": ({2: 1, -2: -1}, 1.0),
    "dv": ({1: 1, -1: -1}, 1.0),
    "ubar": ({-2: 1}, 1.0),
    "dbar": ({-1: 1}, 1.0),
    "s": ({3: 0.5, -3: 0.5}, 1.0),
    "c": ({4: 0.5, -4: 0.5}, 1.0),
    "g": ({21: 1}, 0.1),
}
XMIN, XMAX = 1e-3, 0.95


def read_member(path):
    blocks, cur = [], []
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = lines.index("---") + 1
    while i < len(lines):
        if lines[i].strip() in ("---", ""):
            i += 1
            continue
        xs = [float(v) for v in lines[i].split()]
        qq = [float(v) for v in lines[i + 1].split()]
        pids = [int(v) for v in lines[i + 2].split()]
        n = len(xs) * len(qq)
        rows = [[float(v) for v in lines[i + 3 + k].split()] for k in range(n)]
        blocks.append((xs, qq, pids, rows))
        i += 3 + n
    return blocks


def at_q(blocks, q):
    """dict pid -> list of (x, xf) at scale q, log-linear interpolation in Q^2 between nodes."""
    for xs, qq, pids, rows in blocks:
        if qq[0] <= q <= qq[-1]:
            j = max(k for k in range(len(qq) - 1) if qq[k] <= q)
            t = (math.log(q * q) - math.log(qq[j] ** 2)) / (math.log(qq[j + 1] ** 2) - math.log(qq[j] ** 2))
            out = {p: [] for p in pids}
            for ix, x in enumerate(xs):
                r0, r1 = rows[ix * len(qq) + j], rows[ix * len(qq) + j + 1]
                for ip, p in enumerate(pids):
                    out[p].append((x, (1 - t) * r0[ip] + t * r1[ip]))
            return out
    raise ValueError(f"Q={q} outside grid")


members = sorted(glob.glob(os.path.join(set_dir, "*_0[0-9][0-9][0-9].dat")))
replicas = members[1:]
print(f"{len(replicas)} replicas")
data = {q: {name: [] for name in COMBOS} for q in qs}
xgrid = {}
for path in replicas:
    blocks = read_member(path)
    for q in qs:
        tab = at_q(blocks, q)
        some = next(iter(tab.values()))
        xgrid[q] = [x for x, _ in some]
        for name, (coefs, scale) in COMBOS.items():
            vals = [scale * sum(c * tab[p][ix][1] for p, c in coefs.items()) for ix in range(len(some))]
            data[q][name].append(vals)

for q in qs:
    tag = f"Q{q:g}".replace(".", "p")
    for name in COMBOS:
        reps = data[q][name]
        rows = []
        for ix, x in enumerate(xgrid[q]):
            if not (XMIN <= x <= XMAX):
                continue
            v = [r[ix] for r in reps]
            m = sum(v) / len(v)
            sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))
            rows.append((x, m, m - sd, m + sd))
        with open(os.path.join(out_dir, f"{tag}_{name}.dat"), "w") as fh:
            fh.write("x central lo hi\n")
            for r in rows:
                fh.write("%.6e %.6e %.6e %.6e\n" % r)
        with open(os.path.join(out_dir, f"{tag}_{name}_band.dat"), "w") as fh:
            fh.write("x y\n")
            for x, _, lo, _ in rows:
                fh.write("%.6e %.6e\n" % (x, lo))
            for x, _, _, hi in reversed(rows):
                fh.write("%.6e %.6e\n" % (x, hi))
        peak = max(r[3] for r in rows)
        print(f"Q={q:g} {name:5s} points={len(rows)} max={peak:.3f} at x=1e-3: {rows[0][1]:.3f}")
