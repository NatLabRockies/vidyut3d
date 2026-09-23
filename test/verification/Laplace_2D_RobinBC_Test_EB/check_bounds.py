#!/usr/bin/env python3
"""Smoke check for the EB (immersed boundary) build path.

The case is an annulus held at 10 V on the inner radius and 20 V on the outer,
so the exact potential is a logarithmic profile bounded by those two values.
This checks the maximum of the solved potential against that bound.

Only the maximum is checked: the covered cells inside the immersed body hold
zero, so the global minimum is 0 by construction and says nothing. The upper
bound catches an unstable or overshooting solve, and the lower bound on the
maximum catches a solve that collapsed to zero or a mask that swallowed the
fluid region. It is deliberately loose enough to be compiler independent.

This is not an accuracy check. Convergence order is measured by the case's
get_error.py, which needs yt and is not run in CI.

Usage: check_bounds.py <fextrema output file>
"""
import sys

LO, HI = 19.0, 20.001  # exact max is 20 V at the outer radius; discrete max is just under

def main(path):
    for line in open(path):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "Potential":
            vmax = float(parts[2])
            print(f"max Potential = {vmax:.9g} (expected {LO} to {HI})")
            if not (LO <= vmax <= HI):
                sys.exit(
                    f"FAIL: maximum potential {vmax:.9g} is outside the "
                    f"10-20 V Dirichlet range of this problem"
                )
            print("PASS")
            return
    sys.exit(f"FAIL: no Potential row found in {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
