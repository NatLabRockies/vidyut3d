"""Compare two plotfiles of the pin-to-pin case (e.g. two wall_extrap settings).
usage: python compare_runs.py pltA pltB [var ...]   (default: Potential E ARp cellmask)"""
import sys, numpy as np
from pfread import Plotfile
a, b = Plotfile(sys.argv[1]), Plotfile(sys.argv[2])
vars_ = sys.argv[3:] or ["Potential", "E", "ARp", "cellmask"]
da, db = a.read(vars_), b.read(vars_)
print(f"{'var':10s} {'max|A|':>10s} {'max|A-B|':>10s} {'L2 rel':>10s} {'cells>1%':>8s}")
for v in vars_:
    x, y = da[v], db[v]
    fluid = (da["cellmask"] > 0.5) & (db["cellmask"] > 0.5) if "cellmask" in da else np.ones_like(x, bool)
    if v == "cellmask":
        print(f"{v:10s} mismatching cells: {int((x != y).sum())}")
        continue
    diff = np.abs(x - y)[fluid]; scale = np.abs(x[fluid]).max()
    l2 = np.sqrt((diff**2).mean()) / max(np.sqrt((x[fluid]**2).mean()), 1e-300)
    print(f"{v:10s} {scale:10.3e} {diff.max():10.3e} {l2:10.3e} {int((diff > 0.01*scale).sum()):8d}")
