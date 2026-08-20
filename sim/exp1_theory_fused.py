"""Merged THEORY figure -- invisibility (Exp 1) + necessity of gamma<1 (Exp 2)
fused into a single three-panel figure for the consolidated paper.

This script does NO new simulation: it loads the cached d=20 raw iterates from
Experiment 1 (figs/exp1_projection_clt_raw.npz) and the exact scalar limiting
variance V(gamma,n) from Experiment 2's recursion, and renders

  (a) Invisibility -- Mahalanobis Q-Q at the largest n: plain PR-SGD, the RMSProp
      preconditioner, and SA-Adam(gamma=0.75) all collapse onto the chi^2_d line,
      so the adaptive machinery is distributionally invisible.
  (b) Invisibility over n -- median T_n -> chi^2_d median for the same three arms,
      together at every checkpoint.
  (c) Necessity (the FUSED panel) -- the EXACT scalar variance curve V(gamma)
      (evaluated at the same n=10^8) with the d=20 SIMULATION points
      mean(T_n)/d overlaid on the SAME axes.  The simulated near-boundary
      inflation (gamma=0.95) sits on the curve -> the variance formula is exact,
      not asymptotic hand-waving; the curve then blows up to V(1)=1+1/(2c1-1-a)
      ~ 4.33 at the boundary and diverges beyond -> gamma<1 is necessary.

The d=20 mean(T_n)/d = (1/d) tr( Sigma_sw^{-1} . n Cov(xbar_n) ) is the multivariate
analogue of the scalar V(gamma): it equals 1 exactly when the projection identity
holds (n Cov -> sandwich) and inflates as gamma -> 1.

Run:  python exp1_theory_fused.py        (instant; reads the cached .npz)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2
from numba import njit

# ---- shared schedule (identical in Exp 1 and Exp 2) ------------------------
ALPHA, C1, ETA0, SIGMA = 0.7, 1.0, 0.2, 1.0
PRED_BD = 1.0 + 1.0 / (2 * C1 - 1 - ALPHA)        # V(gamma=1) limit = 4.333

PLOT_STYLE = {"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 19,
              "xtick.labelsize": 17, "ytick.labelsize": 17,
              "legend.fontsize": 15, "figure.titlesize": 22}
matplotlib.rcParams.update(PLOT_STYLE)


@njit(cache=True)
def V_at_n(n_max, alpha, gamma, c1, eta0, sigma):
    """Exact V(gamma,n_max) = n Var[xbar_n] in the scalar model (H=sigma=1), via
    the closed 3x3 covariance recursion of the augmented state."""
    C = np.zeros((3, 3)); A = np.zeros((3, 3)); b = np.zeros(3)
    A[2, 0] = 1.0; A[2, 2] = 1.0
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
    return C[2, 2] / n_max


def main():
    here = os.path.dirname(__file__)
    figs_dir = os.path.join(here, "..", "figs")
    z = np.load(os.path.join(figs_dir, "exp1_projection_clt_raw.npz"),
                allow_pickle=True)
    out = z["out"]; cps = z["cps"]; arms = [str(a) for a in z["arms"]]
    swinv = z["sandwich_inv"]; x_star = z["x_star"]
    M = out.shape[1]; d = out.shape[3]
    big = int(cps[-1]); bi = len(cps) - 1
    n_big_exp = int(round(np.log10(big)))

    # arm indices
    i_id = arms.index("identity")
    i_rms = arms.index("rmsprop")
    sa_arms = [(a, float(a.split("=")[1].rstrip(")")))
               for a in arms if a.startswith("sa_adam")]
    i_sa075 = arms.index("sa_adam(g=0.75)")

    def mahal(ci, ai):
        E = out[ci, :, ai, :] - x_star
        return int(cps[ci]) * np.einsum("ij,jk,ik->i", E, swinv, E)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.4))

    # ===== Panel (a): invisibility -- Mahalanobis Q-Q at the largest n =====
    qs = (np.arange(1, M + 1) - 0.5) / M
    theo = chi2.ppf(qs, df=d)
    qq_arms = [(i_id, "PR-SGD", "tab:blue"),
               (i_rms, "RMSProp precond.", "tab:green"),
               (i_sa075, r"SA-Adam ($\gamma{=}0.75$)", "tab:orange")]
    for ai, lab, col in qq_arms:
        T = np.sort(mahal(bi, ai))
        ax[0].plot(theo, T, "o", ms=7, alpha=0.55, color=col, label=lab)
    lim = [chi2.ppf(0.005, d), chi2.ppf(0.995, d)]
    ax[0].plot(lim, lim, "k--", lw=2.2, label=r"$\chi^2_d$ (ideal)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    xt = [10, 20, 30, 40]
    ax[0].xaxis.set_major_locator(mticker.FixedLocator(xt))
    ax[0].xaxis.set_minor_locator(mticker.NullLocator())
    ax[0].xaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in xt]))
    ax[0].yaxis.set_major_locator(mticker.FixedLocator(xt))
    ax[0].yaxis.set_minor_locator(mticker.NullLocator())
    ax[0].yaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in xt]))
    ax[0].set_xlabel(r"$\chi^2_d$ quantile")
    ax[0].set_ylabel(r"sample $T_n$ quantile")
    ax[0].set_title(rf"(a) Adaptivity is invisible ($n{{=}}10^{{{n_big_exp}}}$, $d{{=}}{d}$)")
    ax[0].legend(fontsize=17, loc="upper left")

    # ===== Panel (b): invisibility over n -- median T_n -> chi^2_d median =====
    for ai, lab, col in qq_arms:
        med = [float(np.median(mahal(ci, ai))) for ci in range(len(cps))]
        ax[1].plot([int(c) for c in cps], med, "-o", ms=9, lw=3.0,
                   color=col, label=lab)
    ax[1].axhline(chi2.median(d), ls="--", color="k", lw=2.2,
                  label=r"$\chi^2_d$ median")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"sample size $n$")
    ax[1].set_ylabel(r"median $T_n$")
    ax[1].set_title(r"(b) median $T_n\to\chi^2_d$ median (adaptivity washes out)")
    ax[1].legend(fontsize=17, loc="upper right")

    # ===== Panel (c): necessity -- exact V(gamma) curve + d=20 sim points =====
    g_fine = np.linspace(0.66, 1.03, 38)
    cache = os.path.join(figs_dir, "exp1_theory_fused_vcurve.npz")
    Vcurve = None
    if os.path.exists(cache):
        cz = np.load(cache)
        if (cz["g_fine"].shape == g_fine.shape and np.allclose(cz["g_fine"], g_fine)
                and int(cz["n"]) == big):
            Vcurve = cz["Vcurve"]
    if Vcurve is None:
        Vcurve = np.array([V_at_n(big, ALPHA, g, C1, ETA0, SIGMA) for g in g_fine])
        np.savez(cache, g_fine=g_fine, Vcurve=Vcurve, n=big)
    # d=20 simulation: mean(T_n)/d at the largest n, for each swept gamma
    sim_g = np.array([g for _, g in sa_arms])
    sim_V = np.array([mahal(bi, arms.index(a)).mean() / d for a, _ in sa_arms])
    print("d=20 simulation  mean(T_n)/d  vs  exact scalar V(gamma, n=%g):" % big)
    for (a, g), v in zip(sa_arms, sim_V):
        print(f"  gamma={g}:  sim={v:.4f}   V={V_at_n(big, ALPHA, g, C1, ETA0, SIGMA):.4f}")

    ax[2].axvspan(1.0, 1.03, color="0.85", alpha=0.6, zorder=0)
    ax[2].plot(g_fine, Vcurve, "-", lw=3.2, color="tab:purple",
               label=r"exact $V(\gamma)$ (theory)")
    ax[2].plot(sim_g, sim_V, "s", ms=14, color="tab:orange",
               markeredgecolor="k", markeredgewidth=1.2, zorder=5,
               label=r"$d{=}20$ simulation: $\overline{T}_n/d$")
    ax[2].axhline(1.0, ls=":", color="gray", lw=2.2, label=r"sandwich $V{=}1$")
    ax[2].axvline(1.0, ls="--", color="k", lw=2.0)
    ax[2].axvline(ALPHA, ls="-.", color="tab:red", lw=2.0)
    # annotate the boundary blow-up and the alpha threshold
    ax[2].annotate(rf"$V(1)\approx{PRED_BD:.2f}$",
                   xy=(1.0, PRED_BD), xytext=(0.80, 4.42), fontsize=18,
                   ha="center", fontweight="bold",
                   arrowprops=dict(arrowstyle="->", lw=1.9))
    ax[2].text(0.706, 0.66, r"$\gamma{=}\alpha$", color="tab:red",
               fontsize=21, va="center", ha="left")
    ax[2].text(1.014, 2.7, "unstable", color="0.35", fontsize=21,
               rotation=90, va="center")
    ax[2].annotate(r"near-boundary lag", xy=(0.945, sim_V[-1] + 0.04),
                   xytext=(0.682, 1.80), fontsize=21, color="tab:orange",
                   arrowprops=dict(arrowstyle="->", lw=1.7, color="tab:orange"))
    ax[2].set_xlim(0.66, 1.03); ax[2].set_ylim(0.5, 5.0)
    ax[2].set_xlabel(r"momentum exponent $\gamma$")
    ax[2].set_ylabel(r"variance ratio $V(\gamma)=n\,\mathrm{Var}/\,$sandwich")
    ax[2].set_title(r"(c) Necessity of $\gamma<1$: formula is exact")
    ax[2].legend(fontsize=18, loc="center")

    fig.tight_layout()
    out_pdf = os.path.join(figs_dir, "exp1_theory_fused.pdf")
    fig.savefig(out_pdf)
    print(f"\nfused theory figure -> {os.path.normpath(out_pdf)}")


if __name__ == "__main__":
    main()
