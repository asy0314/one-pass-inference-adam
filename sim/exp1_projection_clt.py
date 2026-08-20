"""Experiment 1 -- Projection identity / momentum-invisibility (Polyak--Ruppert CLT).

SELF-CONTAINED: running this single file reproduces the experiment's figure, table
(CSV) values, and the in-text numbers; it has no local-module dependencies (only
numpy / scipy / matplotlib / numba).

Streaming linear-regression model: Gaussian covariates a_t ~ N(0, H) on the
Toeplitz Hessian H_{jk} = rho_feat^{|j-k|} (rho_feat=0.4, d=20); responses
y_t = a_t^T x* + eps_t with heteroskedastic noise
eps_t | a_t ~ N(0, sigma0^2 + sigma1^2 (a_t^T u)^2) (sigma0=0.35, sigma1=0.8),
placing the problem in the genuine sandwich regime S != H. Per-sample half-squared
loss; gradient g_t = (a_t^T(x-x*) - eps_t) a_t, clipped at norm 500.

Arms (all on a SHARED gradient stream so the invisibility statistic is meaningful):
  - identity : plain Polyak--Ruppert SGD,  x_{t+1} = x_t - eta_t g_t
  - rmsprop  : diagonal second-moment preconditioner P_t = diag(D_{t-1})^{-1/2},
               D_t = (1-rho_v) D_{t-1} + rho_v (g_t^2 + ridge), rho_v=1/(t+1), ridge=0.5
  - sa_adam  : adds the bias-corrected momentum buffer m_t=(1-rho_m)m_{t-1}+rho_m g_t,
               rho_m = c1 (t+t0)^{-gamma}, gamma in (alpha,1), on top of that preconditioner.
Step size eta_t = 0.2 t^{-0.7}. These match the companion paper An & Huo (2026).

Diagnostics vs the ORACLE sandwich H^{-1} S H^{-1}: Mahalanobis statistic
T_n = n (xbar_n-x*)^T (H^{-1}SH^{-1})^{-1} (xbar_n-x*) -> chi^2_d, its median and KS
distance D_M vs 1.36/sqrt(M); paired-stream invisibility
n||xbar_adam - xbar_sgd||^2_{(H^{-1}SH^{-1})^{-1}}.

This is a large-n model (the methods reach the chi^2_d regime only near n=1e7-1e8),
so the per-step recursion is compiled with numba and parallelized over the M
independent streams; n=1e8, M=200 runs in ~40 min on 7 threads.

Run:  python exp1_projection_clt.py --n 100000000 --M 200 --threads 7
      python exp1_projection_clt.py --n 1000000 --M 200 --check     # quick check
"""
import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.linalg import toeplitz, eigh
from scipy.stats import chi2, kstest
import numba
from numba import njit, prange

PLOT_STYLE = {"font.size": 26, "axes.titlesize": 32, "axes.labelsize": 28,
              "xtick.labelsize": 22, "ytick.labelsize": 22,
              "legend.fontsize": 24, "figure.titlesize": 34}
matplotlib.rcParams.update(PLOT_STYLE)

# ---- Model constants -------------------------------------------------------
DIM = 20
FEATURE_RHO = 0.4
ALPHA = 0.7                 # step exponent: eta_t = ETA0 t^{-ALPHA}
C1 = 1.0                    # momentum gain; second-moment gain rho_v = C1/(t+1)
RIDGE = 0.5                 # stabilization ridge inside the second-moment input
ETA0 = 0.2
SIGNAL_NORM = 0.5
BASE_NOISE = 0.35          # sigma0
HETERO_NOISE = 0.8         # sigma1
EIG_FLOOR = 1e-10
GRADIENT_CLIP = 500.0
BASE_SEED = 20260518


def _spd_inverse(Mx, floor=EIG_FLOOR):
    w, V = eigh(Mx, check_finite=False)
    return V @ np.diag(1.0 / np.clip(w, floor, None)) @ V.T


def _spd_sqrt(Mx, floor=EIG_FLOOR):
    w, V = eigh(Mx, check_finite=False)
    return V @ np.diag(np.sqrt(np.clip(w, floor, None))) @ V.T


def build_truth(d=DIM):
    """Ground truth (x*, H, S, sandwich) for the streaming regression model."""
    H = toeplitz(FEATURE_RHO ** np.arange(d, dtype=float))
    H_inv, H_sqrt = _spd_inverse(H), _spd_sqrt(H)
    eig = np.linalg.eigvalsh(H)
    kappa_H = float(eig.max() / max(eig.min(), EIG_FLOOR))
    cs = SIGNAL_NORM / np.sqrt(d)
    x_star = np.array([cs if j % 2 == 0 else -cs for j in range(d)])
    hetero = np.linspace(0.6, 1.6, d); hetero /= np.linalg.norm(hetero)
    Hu = H @ hetero
    S = BASE_NOISE**2 * H + HETERO_NOISE**2 * (float(hetero @ Hu) * H + 2.0 * np.outer(Hu, Hu))
    sandwich = H_inv @ S @ H_inv
    return dict(H=H, H_sqrt=H_sqrt, S=S, kappa_H=kappa_H, sandwich=sandwich,
                sandwich_inv=_spd_inverse(sandwich), x_star=x_star, hetero=hetero)


def _rand_spd(d, rng):
    """Random symmetric positive-definite matrix with random eigenvectors."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * rng.uniform(0.5, 3.0, size=d)) @ Q.T


def projection_identity_check(d=DIM, n_trials=1000, seed=0):
    """Exact (algebraic) check of Theorem 2: the iterate-marginal block of the
    frozen-chain Polyak--Ruppert covariance Sigma_z = L^{-1} Sigma_w L^{-T} equals
    H^{-1} S H^{-1} REGARDLESS of the preconditioner P and the rates rho, tau.
    Tests random NON-COMMUTING (P, H, S) and random rho, eta (tau = rho/eta).
    Returns the maximum relative error over n_trials draws (-> machine precision).
    """
    rng = np.random.default_rng(seed)
    I = np.eye(d)
    max_rel = 0.0
    for _ in range(n_trials):
        H, S, P = _rand_spd(d, rng), _rand_spd(d, rng), _rand_spd(d, rng)
        rho = rng.uniform(0.1, 0.9)
        eta = rng.uniform(0.01, 0.3)
        tau = rho / eta
        L = np.block([[rho * P @ H, (1 - rho) * P], [-tau * H, tau * I]])
        B = np.block([[-eta * rho * P], [rho * I]])
        Sigma_w = B @ S @ B.T / eta ** 2
        Linv = np.linalg.inv(L)
        Sigma_z = Linv @ Sigma_w @ Linv.T
        sandwich = np.linalg.solve(H, S) @ np.linalg.inv(H)
        rel = np.linalg.norm(Sigma_z[:d, :d] - sandwich) / np.linalg.norm(sandwich)
        max_rel = max(max_rel, rel)
    return max_rel


@njit(parallel=True, fastmath=False, cache=True)
def _kernel(H_sqrt, x_star, hetero, n, M, d, cps, gammas, t0s,
            eta0, alpha, c1, ridge, base_noise, hetero_noise, clip, base_seed,
            out, H_out, S0_out, S2_out):
    n_arms = 2 + gammas.shape[0]          # 0 identity, 1 rmsprop, 2.. sa_adam(gamma)
    n_cps = cps.shape[0]
    for i in prange(M):
        np.random.seed(base_seed + i)
        x = np.zeros((n_arms, d)); xs = np.zeros((n_arms, d))
        D = np.ones((n_arms, d)); m = np.zeros((n_arms, d))
        logkm = np.zeros(n_arms); cidx = 0
        # Online plug-in accumulators (oracle-free, Corollary): Hsum = sum a a^T
        # (shared; the LS Hessian is a a^T), Ssum_* = sum g g^T with the REALIZED
        # gradient g = r a, for arm 0 (SGD) and arm 2 (SA-Adam, gammas[0]).  Only
        # the upper triangle (j<=k) is filled; symmetrized in post-processing.
        Hsum = np.zeros((d, d)); Ssum0 = np.zeros((d, d)); Ssum2 = np.zeros((d, d))
        for t in range(1, n + 1):
            raw = np.random.standard_normal(d + 1)
            a = np.zeros(d)
            for j in range(d):
                s = 0.0
                for k in range(d):
                    s += raw[k] * H_sqrt[k, j]
                a[j] = s
            proj = 0.0; axs = 0.0; a_norm2 = 0.0
            for j in range(d):
                proj += a[j] * hetero[j]; axs += a[j] * x_star[j]; a_norm2 += a[j] * a[j]
            eps = np.sqrt(base_noise * base_noise + (hetero_noise * proj) ** 2) * raw[d]
            a_norm = np.sqrt(a_norm2)
            eta = eta0 * t ** (-alpha)
            rho_v = c1 / (t + 1.0)
            rs0 = 0.0; rs2 = 0.0          # realized-gradient residuals for the plug-in arms
            for arm in range(n_arms):
                resid = -axs - eps
                for j in range(d):
                    xs[arm, j] += x[arm, j]            # PR average (before update)
                    resid += a[j] * x[arm, j]
                gnorm = abs(resid) * a_norm
                rs = resid * (clip / gnorm if gnorm > clip else 1.0)   # g_j = rs a[j]
                if arm == 0:
                    rs0 = rs
                elif arm == 2:
                    rs2 = rs
                if arm == 0:                           # plain PR-SGD
                    for j in range(d):
                        x[arm, j] -= eta * rs * a[j]
                elif arm == 1:                         # RMSProp preconditioner
                    for j in range(d):
                        gj = rs * a[j]
                        x[arm, j] -= eta * gj / np.sqrt(D[arm, j])
                        D[arm, j] = (1 - rho_v) * D[arm, j] + rho_v * (gj * gj + ridge)
                else:                                  # SA-Adam(gamma)
                    gidx = arm - 2
                    rho_m = c1 * (t + t0s[gidx]) ** (-gammas[gidx])
                    if rho_m > 0.999:
                        rho_m = 0.999
                    logkm[arm] += np.log(1.0 - rho_m)
                    km = 1.0 - np.exp(logkm[arm])
                    if km < 1e-15:
                        km = 1e-15
                    for j in range(d):
                        gj = rs * a[j]
                        m[arm, j] = (1 - rho_m) * m[arm, j] + rho_m * gj
                        x[arm, j] -= eta * (m[arm, j] / km) / np.sqrt(D[arm, j])
                        D[arm, j] = (1 - rho_v) * D[arm, j] + rho_v * (gj * gj + ridge)
            # accumulate plug-in statistics on the realized gradients (upper tri)
            rs0sq = rs0 * rs0; rs2sq = rs2 * rs2
            for j in range(d):
                aj = a[j]
                for k in range(j, d):
                    o = aj * a[k]
                    Hsum[j, k] += o
                    Ssum0[j, k] += rs0sq * o
                    Ssum2[j, k] += rs2sq * o
            if cidx < n_cps and t == cps[cidx]:
                inv_t = 1.0 / t
                for arm in range(n_arms):
                    for j in range(d):
                        out[cidx, i, arm, j] = xs[arm, j] / t
                for j in range(d):
                    for k in range(j, d):
                        H_out[cidx, i, j, k] = Hsum[j, k] * inv_t
                        S0_out[cidx, i, j, k] = Ssum0[j, k] * inv_t
                        S2_out[cidx, i, j, k] = Ssum2[j, k] * inv_t
                cidx += 1


def _make_figure(out, cps, arms, swinv, x_star, d, figs_dir):
    """Two-panel figure (Mahalanobis Q-Q at largest n; median_T vs n) from the
    raw iterates `out`.  Single source for both the normal run and `--replot`."""
    M = out.shape[1]
    big = int(cps[-1]); bi = len(cps) - 1
    qs = (np.arange(1, M + 1) - 0.5) / M
    theo = chi2.ppf(qs, df=d)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    # Panel A: Mahalanobis Q-Q against chi^2_d at the largest n
    for ai, a in enumerate(arms):
        E = out[bi, :, ai, :] - x_star
        T = np.sort(big * np.einsum("ij,jk,ik->i", E, swinv, E))
        ax[0].plot(theo, T, "o", ms=7, alpha=0.6, label=str(a))
    lim = [chi2.ppf(0.005, d), chi2.ppf(0.995, d)]
    ax[0].plot(lim, lim, "k--", lw=2.5)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    # plain integer x-ticks {10, 20, 30, 40} instead of 10^1, 2x10^1, 3x10^1, 4x10^1
    xt = [10, 20, 30, 40]
    ax[0].xaxis.set_major_locator(mticker.FixedLocator(xt))
    ax[0].xaxis.set_minor_locator(mticker.NullLocator())
    ax[0].xaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in xt]))
    ax[0].set_xlabel(r"$\chi^2_d$ quantile"); ax[0].set_ylabel(r"sample $T_n$ quantile")
    ax[0].set_title(rf"Mahalanobis Q-Q at $n=10^{{{int(round(np.log10(big)))}}}$ "
                    rf"($d={d}$, $M={M}$)", fontsize=23)
    ax[0].legend(fontsize=18)
    # Panel B: convergence of median_T to the chi^2_d median
    for ai, a in enumerate(arms):
        med = [float(np.median(int(cps[ci]) * np.einsum(
                   "ij,jk,ik->i", out[ci, :, ai, :] - x_star, swinv,
                   out[ci, :, ai, :] - x_star)))
               for ci in range(len(cps))]
        ax[1].plot([int(c) for c in cps], med, "-o", ms=9, lw=3.2, label=str(a))
    ax[1].axhline(chi2.median(d), ls="--", color="k", lw=2.5, label=r"$\chi^2_d$ median")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Sample size $n$"); ax[1].set_ylabel(r"median $T_n$")
    ax[1].set_title(r"Convergence of median $T_n$ to the $\chi^2_d$ median", fontsize=23)
    ax[1].legend(fontsize=18)
    fig.tight_layout()
    out_pdf = os.path.join(figs_dir, "exp1_projection_clt.pdf")
    fig.savefig(out_pdf); print(f"figure -> {os.path.normpath(out_pdf)}")


def _symm_upper(U):
    """U (..., d, d) with only the upper triangle (j<=k) filled -> symmetric."""
    Us = U + np.swapaxes(U, -1, -2)
    di = np.arange(U.shape[-1])
    Us[..., di, di] -= U[..., di, di]
    return Us


def _plugin_T(out_ci, H_ci, S_ci, x_star, n_ci):
    """Oracle-free Mahalanobis T_n = n E^T Sigma_hat^{-1} E per stream, with the
    plug-in Sigma_hat = H_hat^{-1} S_hat H_hat^{-1} formed from the realized
    stream.  out_ci (M,d); H_ci,S_ci (M,d,d) upper-tri -> T (M,)."""
    Hinv = np.linalg.inv(_symm_upper(H_ci))
    Sigma = Hinv @ _symm_upper(S_ci) @ Hinv
    E = out_ci - x_star
    return n_ci * np.einsum("md,mde,me->m", E, np.linalg.inv(Sigma), E)


def _make_plugin_figure(out, H_out, S0_out, S2_out, cps, swinv, x_star, d, figs_dir):
    """Oracle-free check at d=20: the plug-in Mahalanobis statistic (each stream
    forms its OWN Sigma_hat, no oracle) tracks chi^2_d, for SGD (arm 0) and
    SA-Adam gammas[0] (arm 2).  Separate figure, so the shared 2-panel
    projection figure and the DMC version are untouched."""
    M = out.shape[1]
    big = int(cps[-1]); bi = len(cps) - 1
    qs = (np.arange(1, M + 1) - 0.5) / M
    theo = chi2.ppf(qs, df=d)
    arms = [(0, "SGD"), (2, r"SA-Adam ($\gamma{=}0.75$)")]
    Souts = {0: S0_out, 2: S2_out}
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    # Panel A: oracle-free Mahalanobis Q-Q at the largest n
    for ai, lab in arms:
        Tp = np.sort(_plugin_T(out[bi, :, ai, :], H_out[bi], Souts[ai][bi], x_star, big))
        ax[0].plot(theo, Tp, "o", ms=7, alpha=0.6, label=lab)
    lim = [chi2.ppf(0.005, d), chi2.ppf(0.995, d)]
    ax[0].plot(lim, lim, "k--", lw=2.5)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    xt = [10, 20, 30, 40]
    ax[0].xaxis.set_major_locator(mticker.FixedLocator(xt))
    ax[0].xaxis.set_minor_locator(mticker.NullLocator())
    ax[0].xaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in xt]))
    ax[0].set_xlabel(r"$\chi^2_d$ quantile"); ax[0].set_ylabel(r"sample $T_n$ quantile")
    ax[0].set_title(rf"Plug-in (oracle-free) Q-Q at $n=10^{{{int(round(np.log10(big)))}}}$ "
                    rf"($d={d}$, $M={M}$)", fontsize=23)
    ax[0].legend(fontsize=18)
    # Panel B: median T_n vs n -- oracle (solid) vs plug-in (dashed)
    for ai, lab in arms:
        med_o = [float(np.median(int(cps[ci]) * np.einsum(
                     "ij,jk,ik->i", out[ci, :, ai, :] - x_star, swinv,
                     out[ci, :, ai, :] - x_star))) for ci in range(len(cps))]
        med_p = [float(np.median(_plugin_T(out[ci, :, ai, :], H_out[ci], Souts[ai][ci],
                                           x_star, int(cps[ci])))) for ci in range(len(cps))]
        line, = ax[1].plot([int(c) for c in cps], med_o, "-o", ms=9, lw=3.2, label=lab + " (oracle)")
        ax[1].plot([int(c) for c in cps], med_p, "--s", ms=9, lw=3.2,
                   color=line.get_color(), alpha=0.85, label=lab + " (plug-in)")
    ax[1].axhline(chi2.median(d), ls=":", color="k", lw=2.5, label=r"$\chi^2_d$ median")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Sample size $n$"); ax[1].set_ylabel(r"median $T_n$")
    ax[1].set_title(r"Oracle vs. plug-in median $T_n$", fontsize=23)
    ax[1].legend(fontsize=16)
    fig.tight_layout()
    out_pdf = os.path.join(figs_dir, "exp1_plugin.pdf")
    fig.savefig(out_pdf); print(f"plug-in figure -> {os.path.normpath(out_pdf)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000_000)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.75, 0.85, 0.95])
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pcheck", action="store_true",
                    help="exact projection-identity check over random P (instant); then exit")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from the saved raw .npz (no simulation); then exit")
    args = ap.parse_args()
    numba.set_num_threads(args.threads)

    if args.replot:
        figs_dir = os.path.join(os.path.dirname(__file__), "..", "figs")
        z = np.load(os.path.join(figs_dir, "exp1_projection_clt_raw.npz"),
                    allow_pickle=True)
        out = z["out"]
        _make_figure(out, z["cps"], list(z["arms"]), z["sandwich_inv"],
                     z["x_star"], int(out.shape[3]), figs_dir)
        if "H_out" in z:
            _make_plugin_figure(out, z["H_out"], z["S0_out"], z["S2_out"], z["cps"],
                                z["sandwich_inv"], z["x_star"], int(out.shape[3]), figs_dir)
        return

    if args.pcheck:
        mr = projection_identity_check(d=DIM, n_trials=1000, seed=0)
        print(f"projection identity Sigma_z^(1,1) = H^-1 S H^-1 over 1000 random "
              f"non-commuting (P, rho, tau): max relative error = {mr:.2e}")
        return

    d = DIM
    truth = build_truth(d)
    figs_dir = os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)        # so the script runs from a bare folder
    swinv, sw = truth["sandwich_inv"], truth["sandwich"]
    tr_sw = float(np.trace(sw))
    cps = np.array(sorted({c for c in (1e4, 1e5, 1e6, 1e7, 1e8) if c <= args.n}
                          | {args.n}), dtype=np.int64)
    ref = 1.36 / np.sqrt(args.M)
    gammas = np.array(args.gammas, dtype=np.float64)
    t0s = np.array([int(np.ceil(C1 ** (1.0 / g))) for g in args.gammas], dtype=np.int64)
    arms = ["identity", "rmsprop"] + [f"sa_adam(g={g})" for g in args.gammas]
    out = np.zeros((len(cps), args.M, len(arms), d))
    # oracle-free plug-in accumulators: H_hat (shared) and S_hat for SGD (arm 0)
    # and SA-Adam gammas[0] (arm 2); upper triangle filled, symmetrized in post.
    H_out = np.zeros((len(cps), args.M, d, d))
    S0_out = np.zeros((len(cps), args.M, d, d))
    S2_out = np.zeros((len(cps), args.M, d, d))

    print(f"E1 (projection-identity CLT): d={d} kappa(H)={truth['kappa_H']:.2f} "
          f"n={args.n} M={args.M} threads={args.threads} gammas={args.gammas}")
    print(f"chi2_{d}: mean={d} median={chi2.median(d):.3f}; KS ref={ref:.4f}; "
          f"tr(sandwich)={tr_sw:.4f}; checkpoints={cps.tolist()}")
    t0 = time.time()
    _kernel(truth["H_sqrt"], truth["x_star"], truth["hetero"], args.n, args.M, d,
            cps, gammas, t0s, ETA0, ALPHA, C1, RIDGE, BASE_NOISE, HETERO_NOISE,
            GRADIENT_CLIP, BASE_SEED, out, H_out, S0_out, S2_out)
    print(f"run done in {time.time()-t0:.1f}s\n")

    rows = []
    print(f"{'arm':<16}{'n':>11}{'median_T':>10}{'mean_T':>9}{'D_M':>8}"
          f"{'D_M/ref':>9}{'NMSE':>8}")
    for ai, a in enumerate(arms):
        for ci, cp in enumerate(cps):
            E = out[ci, :, ai, :] - truth["x_star"]
            T = cp * np.einsum("ij,jk,ik->i", E, swinv, E)
            nmse = cp * (E ** 2).sum(axis=1).mean() / tr_sw
            D_M = kstest(T, lambda x: chi2.cdf(x, df=d)).statistic
            rows.append((a, int(cp), float(np.median(T)), float(T.mean()),
                         float(D_M), float(D_M / ref), float(nmse)))
            print(f"{a:<16}{int(cp):>11}{np.median(T):>10.3f}{T.mean():>9.3f}"
                  f"{D_M:>8.3f}{D_M/ref:>9.2f}{nmse:>8.3f}")

    big = int(cps[-1]); bi = len(cps) - 1
    base = out[bi, :, 0, :]
    print(f"\nmomentum-invisibility (paired) at n={big}:")
    for ai, a in enumerate(arms):
        if a == "identity":
            continue
        diff = out[bi, :, ai, :] - base
        tr = (big * np.einsum("ij,jk,ik->i", diff, swinv, diff)).mean()
        print(f"  {a:<16}{tr:>12.4f}   (tr sandwich={tr_sw:.3f})")

    out_csv = os.path.join(os.path.dirname(__file__), "..", "figs",
                           "exp1_projection_clt_summary.csv")
    with open(out_csv, "w") as f:
        f.write("arm,n,median_T,mean_T,D_M,ratio,NMSE\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"\nsummary CSV -> {os.path.normpath(out_csv)}")

    # ---- oracle-free plug-in diagnostics (SGD arm 0, SA-Adam gammas[0] arm 2) ----
    print(f"\noracle-free plug-in (each stream forms its own Sigma_hat) at d={d}:")
    print(f"{'arm':<22}{'n':>11}{'med_T(orc)':>12}{'med_T(plg)':>12}{'D_M(plg)':>10}{'ratio':>8}")
    prows = []
    for ai, lab in ((0, "SGD"), (2, f"SA-Adam(g={args.gammas[0]})")):
        Sout = S0_out if ai == 0 else S2_out
        for ci, cp in enumerate(cps):
            E = out[ci, :, ai, :] - truth["x_star"]
            T_o = int(cp) * np.einsum("ij,jk,ik->i", E, swinv, E)
            T_p = _plugin_T(out[ci, :, ai, :], H_out[ci], Sout[ci], truth["x_star"], int(cp))
            D_M = kstest(T_p, lambda x: chi2.cdf(x, df=d)).statistic
            prows.append((lab, int(cp), float(np.median(T_o)), float(np.median(T_p)),
                          float(D_M), float(D_M / ref)))
            print(f"{lab:<22}{int(cp):>11}{np.median(T_o):>12.3f}{np.median(T_p):>12.3f}"
                  f"{D_M:>10.3f}{D_M/ref:>8.2f}")
    out_csv_p = os.path.join(os.path.dirname(__file__), "..", "figs",
                             "exp1_plugin_summary.csv")
    with open(out_csv_p, "w") as f:
        f.write("arm,n,median_T_oracle,median_T_plugin,D_M_plugin,ratio\n")
        for r in prows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")
    print(f"plug-in summary CSV -> {os.path.normpath(out_csv_p)}")

    if args.check:
        print("\n[check] identity median_T ~31-40 at n~3e4 (and ~20 at n>=1e7) "
              "confirms the model/recursion.")
        return

    raw_npz = os.path.join(os.path.dirname(__file__), "..", "figs",
                           "exp1_projection_clt_raw.npz")
    np.savez(raw_npz, out=out, cps=np.array(cps), arms=np.array(arms),
             sandwich_inv=swinv, x_star=truth["x_star"],
             H_out=H_out, S0_out=S0_out, S2_out=S2_out)
    print(f"raw iterates -> {os.path.normpath(raw_npz)}")

    # ---- Figures: shared projection figure + oracle-free plug-in figure ----
    _make_figure(out, cps, arms, swinv, truth["x_star"], d, figs_dir)
    _make_plugin_figure(out, H_out, S0_out, S2_out, cps, swinv, truth["x_star"], d, figs_dir)


if __name__ == "__main__":
    main()
