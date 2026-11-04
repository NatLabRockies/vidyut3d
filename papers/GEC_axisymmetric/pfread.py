"""Minimal reader for single-level AMReX plotfiles (native format), 2D or 3D."""
import numpy as np, re, os
class Plotfile:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "Header")) as f: L = f.read().split("\n")
        self.nvar = int(L[1]); self.vars = L[2:2 + self.nvar]; p = 2 + self.nvar
        self.dim = int(L[p]); self.time = float(L[p + 1]); p += 3
        self.prob_lo = np.array(L[p].split(), float); self.prob_hi = np.array(L[p + 1].split(), float); p += 3
        m = re.match(r"\(\((.*?)\) \((.*?)\) \((.*?)\)\)", L[p].strip())
        lo = np.array(m.group(1).split(","), int); hi = np.array(m.group(2).split(","), int)
        self.dims = hi - lo + 1; self.dx = (self.prob_hi - self.prob_lo) / self.dims
        with open(os.path.join(path, "Level_0", "Cell_H")) as f: H = f.read().split("\n")
        nf = int(H[4].split()[0].strip("(")); self.boxes = []
        for i in range(nf):
            m = re.match(r"\(\((.*?)\) \((.*?)\) \((.*?)\)\)", H[5 + i].strip())
            self.boxes.append((np.array(m.group(1).split(","), int), np.array(m.group(2).split(","), int)))
        q = 5 + nf + 1; nf2 = int(H[q]); q += 1
        self.fabs = [(H[q + i].split()[1], int(H[q + i].split()[2])) for i in range(nf2)]
    def read(self, comps):
        idx = [self.vars.index(c) for c in comps]
        out = {c: np.zeros(tuple(self.dims), np.float64) for c in comps}
        for (lo, hi), (fn, off) in zip(self.boxes, self.fabs):
            n = hi - lo + 1; ncell = int(np.prod(n))
            with open(os.path.join(self.path, "Level_0", fn), "rb") as f:
                f.seek(off); f.readline(); base = f.tell()
                for c, k in zip(comps, idx):
                    f.seek(base + k * ncell * 8)
                    a = np.fromfile(f, np.float64, ncell).reshape(n[::-1]).T
                    out[c][tuple(slice(lo[d], hi[d] + 1) for d in range(self.dim))] = a
        return out
