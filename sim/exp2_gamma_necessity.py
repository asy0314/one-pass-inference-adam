"""Experiment 2 -- Necessity of the sub-linear momentum exponent gamma < 1.

SELF-CONTAINED: running this single file reproduces the experiment's figure and
table values; it has no dependencies beyond numpy / matplotlib / numba.

The sub-linear exponent gamma < 1 is essential: at the canonical value gamma = 1
the endpoint momentum buffer leaves a persistent Theta(n^{-1/2}) residual that
inflates the iterate-marginal covariance above the Polyak--Ruppert sandwich.
We show this by an EXACT, deterministic evaluation -- in the scalar model
H = sigma = 1 -- of the limiting variance

    V(gamma) := lim_n n * Var[ xbar_n ],

obtained by propagating the closed 3x3 covariance recursion of the augmented
state z_t = (Delta_t, m_{t-1}, S_t),  S_t = sum_{s<=t} Delta_s:

    m_t         = (1 - rho_t) m_{t-1} + rho_t Delta_t + rho_t xi_t
    Delta_{t+1} = Delta_t - eta_t m_t
    S_t         = S_{t-1} + Delta_t,   xi_t ~ N(0, sigma^2),

with step eta_t = eta0 t^{-alpha} and momentum rate rho_t = c1 t^{-gamma}.
There is no Monte-Carlo error: V(gamma, n) = Var[S_n]/n = n Var[xbar_n] is exact.

Prediction:  V(gamma) -> 1 for gamma in (alpha, 1);
             V(gamma) -> 1 + 1/(2 c1 - 1 - alpha) at gamma = 1; diverges for gamma > 1.

Run:  python exp2_gamma_necessity.py      (~5-6 min; writes the figure + CSV)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

# Enlarged fonts for a 1x2 figure.
PLOT_STYLE = {"font.size": 26, "axes.titlesize": 32, "axes.labelsize": 28,
              "xtick.labelsize": 22, "ytick.labelsize": 22,
              "legend.fontsize": 24, "figure.titlesize": 34}
matplotlib.rcParams.update(PLOT_STYLE)

ALPHA, C1, ETA0, SIGMA = 0.7, 1.0, 0.2, 1.0          # scalar-model schedule
PRED_BD = 1.0 + 1.0 / (2 * C1 - 1 - ALPHA)            # boundary limit at gamma=1 (=4.333)
PRED_MN = C1 ** 2 / (2 * C1 - 1 - ALPHA)              # n*E[m_n^2] at gamma=1 (=3.333)


@njit(cache=True)
def V_checkpoints(n_max, alpha, gamma, c1, eta0, sigma, cps):
    """Exact V(gamma,n) = n Var[xbar_n] at each n in cps, via one pass to n_max."""
    C = np.zeros((3, 3))
    A = np.zeros((3, 3)); b = np.zeros(3)
    A[2, 0] = 1.0; A[2, 2] = 1.0                      # constant rows of the transition
    out = np.zeros(len(cps)); ci = 0
    for t in range(1, n_max + 1):
        eta = eta0 * t ** (-alpha)
        rho = c1 * t ** (-gamma)
        if rho > 0.999:
            rho = 0.999
        A[0, 0] = 1.0 - eta * rho; A[0, 1] = -eta * (1.0 - rho)
        A[1, 0] = rho;             A[1, 1] = 1.0 - rho
        b[0] = -eta * rho;         b[1] = rho
        C2 = A @ C @ A.T
        for i in range(3):
            for j in range(3):
                C[i, j] = C2[i, j] + sigma * sigma * b[i] * b[j]
        if ci < len(cps) and t == cps[ci]:
            out[ci] = C[2, 2] / t
            ci += 1
    return out


def V_at(n, gamma):
    return float(V_checkpoints(int(n), ALPHA, gamma, C1, ETA0, SIGMA,
                               np.array([int(n)], dtype=np.int64))[0])


def main():
    gammas = [0.75, 0.85, 0.95]          # in (alpha, 1): sandwich holds
    gamma_bd, gamma_super = 1.0, 1.1
    print("=" * 78)
    print("Experiment 2: necessity of sublinear momentum (gamma < 1)")
    print(f"scalar model H=sigma=1; alpha={ALPHA}, c1={C1}, eta0={ETA0}")
    print(f"predicted V(gamma=1) = 1 + 1/(2c1-1-alpha) = {PRED_BD:.4f};  "
          f"n*E[m_n^2] -> {PRED_MN:.4f}")
    print("=" * 78)
    t0 = time.time()

    # ---- left-panel data: V(gamma,n) vs n, recorded in ONE pass per gamma ----
    n_grid = np.unique(np.concatenate([
        [1e4, 1e6], np.logspace(3, 8, 16)]).round().astype(np.int64))
    n_max = int(n_grid[-1])              # 10^8
    curves = {}
    for g in gammas + [gamma_bd]:
        curves[g] = V_checkpoints(n_max, ALPHA, g, C1, ETA0, SIGMA, n_grid)

    # ---- TABLE VALUES (printed + CSV) ----
    tab_n = [10**4, 10**6, 10**8]
    idx = {nn: int(np.where(n_grid == nn)[0][0]) for nn in tab_n}
    print(f"\nTABLE VALUES  V(gamma, n) = n*Var[xbar_n]  (sandwich limit = 1):")
    print(f"  {'gamma':>7}{'V(1e4)':>10}{'V(1e6)':>10}{'V(1e8)':>10}   verdict")
    for g in gammas:
        v = [curves[g][idx[nn]] for nn in tab_n]
        print(f"  {g:>7}{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>10.3f}   -> 1 (sandwich)")
    vb = [curves[gamma_bd][idx[nn]] for nn in tab_n]
    print(f"  {gamma_bd:>7}{vb[0]:>10.3f}{vb[1]:>10.3f}{vb[2]:>10.3f}   "
          f"-> {PRED_BD:.3f} (inflated)")
    vsup = V_at(10**6, gamma_super)
    print(f"  {gamma_super:>7}{'--':>10}{vsup:>10.3f}{'--':>10}   diverges (gamma>1)")

    figs_dir = os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)        # so the script runs from a bare folder
    out_csv = os.path.join(figs_dir, "exp2_gamma_necessity_values.csv")
    with open(out_csv, "w") as f:
        f.write("gamma," + ",".join(f"V_n={int(n)}" for n in n_grid) + "\n")
        for g in gammas + [gamma_bd]:
            f.write(f"{g}," + ",".join(f"{v:.5f}" for v in curves[g]) + "\n")
    print(f"\ntable-values CSV -> {os.path.normpath(out_csv)}")

    # ---- right-panel data: V(gamma) across a fine gamma grid at n = 1e7 ----
    g_fine = np.linspace(0.55, 1.10, 21)
    n_right = 10 ** 7
    Vfine = np.array([V_checkpoints(n_right, ALPHA, g, C1, ETA0, SIGMA,
                                    np.array([n_right], dtype=np.int64))[0]
                      for g in g_fine])

    # ---- Figure (1x2) ----
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    cmap = {0.75: "tab:green", 0.85: "tab:blue", 0.95: "tab:cyan", 1.0: "k"}
    for g in gammas + [gamma_bd]:
        ax[0].plot(n_grid, curves[g], "-o", ms=6, color=cmap[g],
                   lw=(4.5 if g == 1.0 else 3.2), label=rf"$\gamma={g}$")
    ax[0].axhline(1.0, ls=":", color="gray", lw=2.5, label="sandwich $V=1$")
    ax[0].axhline(PRED_BD, ls="--", color="k", lw=2.5,
                  label=rf"pred. $V(1)={PRED_BD:.2f}$")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Sample size $n$")
    ax[0].set_ylabel(r"$V(\gamma,n)=n\,\mathrm{Var}[\bar x_n]$")
    ax[0].set_title("Limiting variance vs $n$ (exact)"); ax[0].legend(fontsize=18)

    ax[1].plot(g_fine, Vfine, "-o", ms=6, lw=3.2, color="tab:blue")
    ax[1].axhline(1.0, ls=":", color="gray", lw=2.5, label="sandwich $V=1$")
    ax[1].axvline(1.0, ls="--", color="k", lw=2.5, label=r"boundary $\gamma=1$")
    ax[1].axvline(ALPHA, ls="-.", color="tab:red", lw=2.5, label=r"$\gamma=\alpha$")
    ax[1].set_xlabel(r"momentum exponent $\gamma$")
    ax[1].set_ylabel(rf"$V(\gamma,\, n=10^7)$")
    ax[1].set_title(r"Sharp transition at $\gamma=1$"); ax[1].legend(fontsize=18)
    fig.tight_layout()
    out_pdf = os.path.join(figs_dir, "exp2_gamma_necessity.pdf")
    fig.savefig(out_pdf)
    print(f"figure -> {os.path.normpath(out_pdf)}")
    print(f"(elapsed {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
