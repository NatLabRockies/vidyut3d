#!/usr/bin/env python3
"""Accuracy and cost, uniform against refined, for Laplace_2D_Posts.

Errors come from amr_error.analyse; cost is cell updates summed over levels and
substeps ("Advanced N cells" in the log), because a refined run subcycles and
n^2 is not what it pays. Read in PAIRS at equal finest spacing.

Usage: python3 cost_table.py [--near HALF]
"""
import glob
import os
import re
import sys

import amr_error

BASE = 64
PAIRS = [("uni128", "amrL1", 128), ("uni256", "amrL2", 256), ("uni512", "amrL3", 512)]


def cost(d):
    cells, secs = 0, float("nan")
    with open(os.path.join(d, "log")) as fh:
        for line in fh:
            m = re.search(r"Advanced (\d+) cells", line)
            if m:
                cells += int(m.group(1))
            m = re.match(r"Evolve_Time:\s+\S+\s+(\S+)", line)
            if m:
                secs = float(m.group(1))
    return cells, secs


def main():
    half = None
    if "--near" in sys.argv:
        half = float(sys.argv[sys.argv.index("--near") + 1])
        print(f"errors restricted to |x|,|y| <= {half} (the refined patch)\n")
    hdr = (f"{'finest':>7s}  {'run':8s} {'cellupd':>10s} {'sec':>6s} "
           f"{'L2':>12s} {'Linf':>12s}  {'saving':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for u, a, eff in PAIRS:
        row = []
        for d in (u, a):
            pf = sorted(glob.glob(os.path.join(d, "plt*")))
            if not pf:
                return
            _, l2, linf, _ = amr_error.analyse(pf[-1], half)
            c, s = cost(d)
            row.append((d, c, s, l2, linf))
        (du, cu, su, l2u, liu), (da, ca, sa, l2a, lia) = row
        print(f"{eff:7d}  {du:8s} {cu:10d} {su:6.1f} {l2u:12.4e} {liu:12.4e}")
        print(f"{'':7s}  {da:8s} {ca:10d} {sa:6.1f} {l2a:12.4e} {lia:12.4e}"
              f"  {cu / ca:6.1f}x")


if __name__ == "__main__":
    main()
