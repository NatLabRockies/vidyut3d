#!/usr/bin/env python3
"""L2 errors and observed order for the MMS2 annulus case, numpy only.

Reads single-level 2D AMReX plotfiles directly (no yt), masks to the fluid
with cellmask > 1 - 1e-10 and normalises the L2 norm by the whole domain,
the same convention as all_errors.py.

    python3 check_errors.py 32/plt00001 64/plt00001 --een-scale 2.4032649e-20
    python3 check_errors.py 32/plt00001 64/plt00001 --deck inputs2d_joule \
        --min-order 1.8 --fields rel_Electron_energy rel_Te

--een-scale must match prob.een_scale in the run; --deck reads it from the
input file instead (and refuses a conflicting --een-scale). With --min-order the
script exits non-zero if any checked field converges slower than that
between the last two plotfiles, so it can gate a CI step.
"""
import argparse
import os
import re
import sys

import numpy as np

ECHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
ALPHA = ECHARGE / EPS0
N0 = 1.0e6


def read_plotfile(path):
    """Return (names, prob_lo, dx, ncell, data[ny, nx, ncomp]) for level 0."""
    with open(os.path.join(path, "Header")) as f:
        lines = f.read().split("\n")
    nvar = int(lines[1])
    names = [lines[2 + n].strip() for n in range(nvar)]
    p = 2 + nvar
    dim = int(lines[p])
    if dim != 2:
        raise SystemExit("check_errors.py reads 2D plotfiles only")
    # dim, time, finest_level, prob_lo, prob_hi, ref_ratio, domain, step, dx
    finest = int(lines[p + 2])
    if finest != 0:
        raise SystemExit("check_errors.py reads single-level plotfiles only")
    prob_lo = np.array(lines[p + 3].split(), dtype=float)
    box = re.findall(r"-?\d+", lines[p + 6])
    lo = np.array(box[0:2], dtype=int)
    hi = np.array(box[2:4], dtype=int)
    ncell = hi - lo + 1
    dx = np.array(lines[p + 8].split(), dtype=float)

    data = np.full((ncell[1], ncell[0], nvar), np.nan)
    lev = os.path.join(path, "Level_0")
    with open(os.path.join(lev, "Cell_H")) as f:
        cellh = f.read()
    for fname, offset in re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", cellh):
        with open(os.path.join(lev, fname), "rb") as fb:
            fb.seek(int(offset))
            head = fb.readline().decode()
            m = re.search(
                r"\(\((\-?\d+),(\-?\d+)\) \((\-?\d+),(\-?\d+)\) \(\d+,\d+\)\) (\d+)",
                head,
            )
            prec = re.search(r"\(8, \(([\d ]+)\)\)\)", head)
            if m is None or prec is None:
                raise SystemExit(
                    f"{path}: not a 2D double-precision FAB header: {head!r}")
            blo = np.array([int(m.group(1)), int(m.group(2))])
            bhi = np.array([int(m.group(3)), int(m.group(4))])
            nc = int(m.group(5))
            n = bhi - blo + 1
            order = prec.group(1).split()
            # AMReX writes "8 7 6 5 4 3 2 1" for little-endian doubles
            dtype = "<f8" if order[0] == "8" else ">f8"
            arr = np.frombuffer(fb.read(8 * n[0] * n[1] * nc), dtype=dtype)
            arr = arr.reshape(nc, n[1], n[0])
            ys = slice(blo[1] - lo[1], bhi[1] - lo[1] + 1)
            xs = slice(blo[0] - lo[0], bhi[0] - lo[0] + 1)
            data[ys, xs, :] = np.moveaxis(arr, 0, -1)
    return names, prob_lo, dx, ncell, data


def errors(path, een_scale):
    names, prob_lo, dx, ncell, data = read_plotfile(path)
    get = lambda nm: data[:, :, names.index(nm)]
    xc = prob_lo[0] + (np.arange(ncell[0]) + 0.5) * dx[0]
    yc = prob_lo[1] + (np.arange(ncell[1]) + 0.5) * dx[1]
    x, y = np.meshgrid(xc, yc)
    r2 = x * x + y * y
    m = (get("cellmask") > 1 - 1e-10).astype(float)

    # np.where, not f * m: covered cells may hold inf/NaN, and inf * 0 = NaN
    def l2(f, ex):
        return np.sqrt(np.mean(np.where(m > 0, f - ex, 0.0) ** 2))

    ne = r2 / ALPHA + N0
    out = {
        "N": int(ncell[0]),
        "Potential": l2(get("Potential"), r2 * r2 / 32.0),
        "E": l2(get("E"), ne),
        "HEp": l2(get("HEp"), 0.5 * r2 / ALPHA + N0),
        "Electron_energy": l2(get("Electron_energy"), een_scale * ne),
    }
    # relative errors, so the energy reads on the same scale whatever een_scale
    out["rel_E"] = out["E"] / l2(0.0, ne)
    out["rel_Electron_energy"] = out["Electron_energy"] / l2(0.0, een_scale * ne)
    # Te backed out of E_e / n_e (only meaningful with vidyut.freeze_etemp=0)
    te_exact = (2.0 / 3.0) * een_scale / K_B
    out["rel_Te"] = l2(get("Electron_Temp"), te_exact) / (
        np.sqrt(np.mean(m)) * te_exact
    )
    return out


def een_scale_from_deck(path):
    """Last prob.een_scale in an AMReX input file (ParmParse keeps the last)."""
    val = 1.0
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0]
            m = re.match(r"\s*prob\.een_scale\s*=\s*(\S+)", line)
            if m:
                val = float(m.group(1))
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("plotfiles", nargs="+")
    ap.add_argument("--een-scale", type=float, default=None,
                    help="prob.een_scale used in the run (default 1)")
    ap.add_argument("--deck", default=None,
                    help="input file of the run; prob.een_scale is read from it")
    ap.add_argument("--fields", nargs="+", default=["Electron_energy"],
                    help="fields gated by --min-order")
    ap.add_argument("--min-order", type=float, default=None)
    a = ap.parse_args()
    if a.deck is not None:
        deck_scale = een_scale_from_deck(a.deck)
        if a.een_scale is not None and a.een_scale != deck_scale:
            raise SystemExit(f"--een-scale {a.een_scale} conflicts with "
                             f"prob.een_scale = {deck_scale} in {a.deck}")
        a.een_scale = deck_scale
    if a.een_scale is None:
        a.een_scale = 1.0
    if not a.een_scale > 0.0:
        raise SystemExit("--een-scale must be > 0")

    rows = [errors(p, a.een_scale) for p in a.plotfiles]
    cols = ["Potential", "E", "HEp", "Electron_energy",
            "rel_E", "rel_Electron_energy", "rel_Te"]
    print("| N | " + " | ".join(f"{c} | p" for c in cols) + " |")
    print("|---" * (1 + 2 * len(cols)) + "|")
    for k, r in enumerate(rows):
        cells = [str(r["N"])]
        for c in cols:
            cells.append(f"{r[c]:.3e}")
            if k == 0:
                cells.append("--")
            else:
                q = rows[k - 1]
                cells.append(f"{np.log(q[c] / r[c]) / np.log(r['N'] / q['N']):.2f}")
        print("| " + " | ".join(cells) + " |")

    if a.min_order is not None:
        if len(rows) < 2:
            raise SystemExit("--min-order needs at least two plotfiles")
        q, r = rows[-2], rows[-1]
        fail = False
        for c in a.fields:
            if c not in r:
                raise SystemExit(f"unknown field {c!r}; choose from {sorted(r)}")
            p = np.log(q[c] / r[c]) / np.log(r["N"] / q["N"])
            ok = np.isfinite(p) and p >= a.min_order
            print(f"{c}: observed order {p:.2f} (need >= {a.min_order}) "
                  f"{'PASS' if ok else 'FAIL'}")
            fail |= not ok
        sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
