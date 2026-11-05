"""Paper figure for Case 5: L2 errors of phi and n_i (style of mms1_error_plot.py).
Numbers are the all-cell L2 errors printed by sphere_errors.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=16)
plt.rc("axes", titlesize=18, labelsize=20)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
cmap = ["#EE2E2F", "#008C48", "#185AA9", "#F47D23", "#662C91", "#A21D21", "#B43894", "#010202"]
marker = ["s", "d"]

N = np.array([32, 64, 128, 256, 512])
data = {
    "error": (r"$L_2(\phi)$",
              {"Ray value": [4.3620e-2, 1.3283e-2, 3.6857e-3, 9.2305e-4, 2.2935e-4],
               "Linearized value": [1.6652e-2, 1.4172e-3, 2.9343e-4, 8.5783e-5, 2.2361e-5]}),
    "error_NI": (r"$L_2(n_i)$",
                 {"Ray value": [9.0347e-1, 2.7247e-1, 7.6090e-2, 2.0647e-2, 5.4297e-3],
                  "Linearized value": [9.7145e-1, 2.8791e-1, 8.0938e-2, 2.1963e-2, 5.8025e-3]}),
}
for name, (ylabel, series) in data.items():
    plt.figure(figsize=(8, 6))
    for k, (lab, e) in enumerate(series.items()):
        plt.loglog(N, e, lw=2, marker=marker[k], ms=5, color=cmap[k], label=lab)
    ref = np.array(series["Ray value"])
    plt.loglog(N, ref[1] * (N[1] / N) ** 1, lw=2, color=cmap[-1], label="1st order", zorder=0)
    plt.loglog(N, ref[1] * (N[1] / N) ** 2, lw=2, ls="--", color=cmap[-1], label="2nd order", zorder=0)
    plt.xlabel(r"$N_x$"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{name}.png", dpi=300); plt.close()
print("wrote error.png error_NI.png")
