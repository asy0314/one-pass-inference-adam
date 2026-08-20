#!/usr/bin/env python
"""Experiment 3 -- Semi-synthetic Polyak--Ruppert coverage for averaged SA-Adam
(Section 6.3).  Self-contained: this single file reproduces its figure and
summary CSV with no shared module and no reliance on any prior output.

Design (real features, known truth -- so coverage can be checked exactly):
  * Real covariates: the bundled (no-network) scikit-learn ``diabetes`` feature
    matrix, standardized; the two near-collinear serum features (s1, s4) are
    dropped (full set has cond(H) ~ 470; the kept 8 give a realistic, well-
    conditioned cond(H) ~ 7.5), plus an intercept -> pool A in R^{N x d}, d = 9.
  * Known model: y = a^T beta* + eps with HETEROSCEDASTIC noise
        Var(eps | a) = sigma^2(a),   sigma(a) = 0.5 + 0.8 |a^T w|,
    so the gradient covariance S = E[sigma^2(a) a a^T] differs from the Hessian
    H = E[a a^T]; the sandwich Sigma = H^{-1} S H^{-1} is non-trivial.
  * Stream: each step draws a pool row uniformly with replacement (population =
    empirical distribution of the rows) and a fresh noise draw.  Because the
    population is the pool, beta*, H, S, Sigma are known EXACTLY, so Wald-CI
    coverage is checkable against ground truth -- the point of a semi-synthetic
    (rather than pure real-data) design.

We compare averaged SA-Adam (decaying-momentum, bias-corrected Adam with the
stochastic-approximation schedules beta_{1,t}=1-t^{-gamma}, beta_{2,t}=1-t^{-1},
eta_t = eta0 t^{-alpha}) against plain averaged SGD.  The theory predicts BOTH
averaged iterates are sqrt(n)-asymptotically N(0, Sigma) with the SAME sandwich
Sigma -- preconditioning and momentum are invisible to the averaged-iterate
covariance (the projection identity).

Outputs (written to ../figs/):
  * exp3_coverage.pdf          -- chi^2_d Q-Q, coverage calibration, coverage vs n
  * exp3_coverage_summary.csv  -- flat per-n summary table

Two covariance diagnostics are reported side by side: (i) the ORACLE sandwich
H^{-1}SH^{-1} (isolates the distributional claim from covariance estimation), and
(ii) the single-pass ONLINE PLUG-IN of Chen et al. (2020) -- each replication
forms its own Sigma_hat = H_hat^{-1} S_hat H_hat^{-1} from the realized stream
(H_hat = mean a a^T, S_hat = mean g g^T over the realized gradients g = r a), with
NO oracle and NO second pass.  The plug-in is the end-to-end Algorithm 1: it
produces calibrated Wald intervals from data alone.  Parallel over the M
replications; each worker is single-threaded (BLAS pinned to 1) so --cores
processes use ~--cores cores total.

Run:  python exp3_coverage.py            # n=1e8, M=1000, 7 cores
      python exp3_coverage.py --n 1000000 --M 1000 --cores 7         # faster
      python exp3_coverage.py --check                                # quick smoke (safe outputs)
"""

import os
# Pin BLAS / OpenMP to a single thread BEFORE importing numpy, so that the
# worker processes use ~--cores cores total (not cores x #BLAS-threads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import math
import argparse
import numpy as np
from multiprocessing import Pool
from numba import njit

# ----------------------------------------------------------------------------
# Fixed schedule (canonical operating point of Section 6.3); alpha in (1/2,1),
# gamma in (alpha,1).  These are constants (not CLI knobs): they are evaluated
# identically in the parent and in every spawned worker, so the run is
# reproducible regardless of the start method.
# ----------------------------------------------------------------------------
ALPHA      = 0.6      # step:     eta_t = eta0 * t^{-alpha}
GAMMA      = 0.75     # momentum: rho_t = c1 * t^{-gamma}
C1         = 1.0      # rho_t = c1/t^gamma   (c1=1 => no first-moment bias offset)
C2         = 1.0      # beta2_t = 1 - c2/t   (c2=1 => no second-moment bias offset)
EPS        = 1e-8     # Adam epsilon
ETA_SAFETY = 1.0      # eta0 = ETA_SAFETY / lambda_max(effective Hessian)
SEED0      = 20260601


def checkpoints_for(n):
    """Powers of ten 1e3 .. n (n is assumed a power of ten)."""
    hi = int(round(math.log10(n)))
    return np.array([10 ** k for k in range(3, hi + 1)], dtype=np.int64)


# ----------------------------------------------------------------------------
# Population (real features + known heteroscedastic model)
# ----------------------------------------------------------------------------
def build_population():
    from sklearn.datasets import load_diabetes
    X = load_diabetes().data                      # 442 x 10, real covariates
    X = (X - X.mean(0)) / X.std(0)                # standardize columns
    keep = [0, 1, 2, 3, 5, 6, 8, 9]               # drop near-collinear s1, s4
    X = X[:, keep]
    N = X.shape[0]
    A = np.hstack([np.ones((N, 1)), X])           # intercept + 8 features -> d = 9
    d = A.shape[1]

    rng = np.random.default_rng(0)
    beta = rng.normal(size=d)                     # known true parameter
    w = rng.normal(size=d); w /= np.linalg.norm(w)
    s = A @ w
    sigma2 = (0.5 + 0.8 * np.abs(s)) ** 2         # heteroscedastic noise variance

    H = A.T @ A / N
    S = (A * sigma2[:, None]).T @ A / N
    Hinv = np.linalg.inv(H)
    Sigma = Hinv @ S @ Hinv                       # known-exactly sandwich
    return A, beta, sigma2, H, S, Sigma, d, N


def step_sizes(H, S):
    """Step sizes from the (known) geometry; Polyak averaging makes the
    asymptotic covariance independent of eta0, so any stable choice is fine."""
    lam_H = np.linalg.eigvalsh(H).max()
    Dm = np.diag(np.diag(S) ** -0.25)             # P^{1/2} with P = diag(S)^{-1/2}
    lam_eff = np.linalg.eigvalsh(Dm @ H @ Dm).max()
    return ETA_SAFETY / lam_H, ETA_SAFETY / lam_eff


# ----------------------------------------------------------------------------
# JIT-compiled inner loop.  The data stream is generated *inside* the kernel
# (one bootstrap index + one heteroscedastic-noise draw per step, seeded per
# replication), so we never materialize length-n arrays -- essential at n = 1e8.
# Per-step schedules are computed inline.  cache=True => compiled once and
# reused by the spawned workers.
# ----------------------------------------------------------------------------
@njit(cache=True)
def _run_kernel(A, Abeta, sigma, d, N, n, c1, gamma, c2, alpha,
                eta0_a, eta0_s, eps, ck, seed):
    np.random.seed(seed)
    nck = ck.shape[0]
    th_a = np.zeros(d); m = np.zeros(d); v = np.zeros(d); sum_a = np.zeros(d)
    th_s = np.zeros(d); sum_s = np.zeros(d)
    # ---- Online single-pass plug-in (Chen et al. 2020), accumulated on the
    # realized stream -- no oracle, no second pass.  Hsum = sum_t a_t a_t^T is
    # shared (the LS Hessian is a a^T, independent of the iterate); Ssum_* =
    # sum_t g_t g_t^T with the REALIZED gradients g_t = r_t a_t already formed
    # below, one accumulator per arm (Adam / SGD residuals differ).
    Hsum   = np.zeros((d, d))
    Ssum_a = np.zeros((d, d))
    Ssum_s = np.zeros((d, d))
    rec_a = np.empty((nck, d)); rec_s = np.empty((nck, d))
    rec_H  = np.empty((nck, d, d))                 # plug-in Hessian  H_hat_n
    rec_Sa = np.empty((nck, d, d))                 # plug-in S_hat_n (SA-Adam)
    rec_Ss = np.empty((nck, d, d))                 # plug-in S_hat_n (SGD)
    cptr = 0
    for i in range(n):
        ii = np.random.randint(0, N)                              # bootstrap a real-feature row
        y = Abeta[ii] + sigma[ii] * np.random.standard_normal()   # known model + heterosced. noise
        t = float(i + 1)
        rh = c1 * t ** (-gamma); omrh = 1.0 - rh
        b2 = 1.0 - c2 / t;       omb2 = 1.0 - b2
        ea = eta0_a * t ** (-alpha); es = eta0_s * t ** (-alpha)
        dot_a = 0.0; dot_s = 0.0
        for j in range(d):
            aij = A[ii, j]
            dot_a += aij * th_a[j]
            dot_s += aij * th_s[j]
        ra = dot_a - y; rs = dot_s - y
        ra2 = ra * ra; rs2 = rs * rs
        for j in range(d):
            aij = A[ii, j]
            g = ra * aij
            v[j] = b2 * v[j] + omb2 * g * g
            m[j] = omrh * m[j] + rh * g
            th_a[j] -= ea * (m[j] / (math.sqrt(v[j]) + eps))
            sum_a[j] += th_a[j]
            th_s[j] -= es * (rs * aij)
            sum_s[j] += th_s[j]
        # plug-in covariance statistics on the realized gradients (g_a=ra*a,
        # g_s=rs*a); a a^T accumulated once and reused with the two residuals.
        for j in range(d):
            aij = A[ii, j]
            for k in range(d):
                o = aij * A[ii, k]
                Hsum[j, k]   += o
                Ssum_a[j, k] += ra2 * o
                Ssum_s[j, k] += rs2 * o
        if cptr < nck and (i + 1) == ck[cptr]:
            inv = 1.0 / (i + 1)
            for j in range(d):
                rec_a[cptr, j] = sum_a[j] * inv
                rec_s[cptr, j] = sum_s[j] * inv
                for k in range(d):
                    rec_H[cptr, j, k]  = Hsum[j, k]   * inv
                    rec_Sa[cptr, j, k] = Ssum_a[j, k] * inv
                    rec_Ss[cptr, j, k] = Ssum_s[j, k] * inv
            cptr += 1
    return rec_a, rec_s, rec_H, rec_Sa, rec_Ss


# ----------------------------------------------------------------------------
# Per-worker population (built once per process under 'spawn').  Only the fixed
# population lives in a global; the per-run config (n, checkpoints) is passed
# through pool.map, so it reaches the workers correctly under any start method.
# ----------------------------------------------------------------------------
_G = {}
def init_worker():
    A, beta, sigma2, H, S, Sigma, d, N = build_population()
    eta0_sgd, eta0_adam = step_sizes(H, S)
    _G.update(dict(
        A=np.ascontiguousarray(A), Abeta=A @ beta, sigma=np.sqrt(sigma2),
        d=d, N=N, eta0_sgd=float(eta0_sgd), eta0_adam=float(eta0_adam),
    ))


def run_rep(arg):
    """One replication on a fresh stream (generated inside the kernel from this
    seed): returns the running Polyak average at each checkpoint, per method."""
    seed, n, ck = arg
    g = _G
    return _run_kernel(g["A"], g["Abeta"], g["sigma"], g["d"], g["N"], n,
                       C1, GAMMA, C2, ALPHA, g["eta0_adam"], g["eta0_sgd"],
                       EPS, ck, int(seed))


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def summarize(bars, beta, Sigma, n):
    """bars: (M, d) averaged iterates at sample size n -> dict of metrics."""
    from scipy.stats import chi2
    M, d = bars.shape
    diff = bars - beta
    Sinv = np.linalg.inv(Sigma)
    T = n * np.einsum("md,de,me->m", diff, Sinv, diff)          # Mahalanobis ~ chi^2_d
    emp_cov_scaled = n * np.cov(bars, rowvar=False)
    sand_relerr = np.linalg.norm(emp_cov_scaled - Sigma) / np.linalg.norm(Sigma)
    se = np.sqrt(np.diag(Sigma) / n)
    marg_cov = np.mean(np.abs(diff) <= 1.96 * se[None, :], axis=0)
    return dict(
        n=int(n),
        bias_norm=float(np.linalg.norm(diff.mean(0))),
        sandwich_relerr=float(sand_relerr),
        maha_mean=float(T.mean()),
        maha_median=float(np.median(T)),
        chi2_median=float(chi2.ppf(0.5, d)),
        joint_cov95=float(np.mean(T <= chi2.ppf(0.95, d))),
        marg_cov95_mean=float(marg_cov.mean()),
        marg_cov95_min=float(marg_cov.min()),
    )


def plugin_sigma(H_hat, S_hat):
    """Per-replication plug-in sandwich Sigma_hat = H^{-1} S H^{-1}.
    H_hat, S_hat: (M, d, d) stacks -> (M, d, d)."""
    Hinv = np.linalg.inv(H_hat)
    return Hinv @ S_hat @ Hinv


def summarize_plugin(bars, beta, Sig_hat, n):
    """Coverage of DATA-DRIVEN Wald intervals: each replication uses its own
    plug-in Sig_hat (no oracle).  bars,(M,d); Sig_hat,(M,d,d) -> metrics."""
    from scipy.stats import chi2
    M, d = bars.shape
    diff = bars - beta
    Sinv = np.linalg.inv(Sig_hat)                                  # (M,d,d)
    T = n * np.einsum("md,mde,me->m", diff, Sinv, diff)            # per-rep Mahalanobis
    diag = np.diagonal(Sig_hat, axis1=1, axis2=2)                  # (M,d) plug-in variances
    se = np.sqrt(diag / n)
    marg_cov = np.mean(np.abs(diff) <= 1.96 * se, axis=0)          # per-coordinate
    relerr = np.linalg.norm(Sig_hat.mean(0) - np.cov(bars, rowvar=False) * n) \
        / np.linalg.norm(np.cov(bars, rowvar=False) * n)
    return dict(
        n=int(n),
        maha_median=float(np.median(T)),
        chi2_median=float(chi2.ppf(0.5, d)),
        joint_cov95=float(np.mean(T <= chi2.ppf(0.95, d))),
        marg_cov95_mean=float(marg_cov.mean()),
        marg_cov95_min=float(marg_cov.min()),
        sigmahat_relerr=float(relerr),                            # mean plug-in vs empirical cov
    )


def _maha_plugin(diff, Sig):
    """Per-replication Mahalanobis n*diff' Sig_hat^{-1} diff (each rep its own Sig)."""
    return np.einsum("md,mde,me->m", diff, np.linalg.inv(Sig), diff)


def make_figure(bars_a, bars_s, Sig_a, Sig_s, beta, Sigma, d, ck, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    plt.rcParams.update({
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 12, "lines.markersize": 5,
    })

    Sinv = np.linalg.inv(Sigma)
    n_final = int(ck[-1])
    da_f = bars_a[:, -1, :] - beta; ds_f = bars_s[:, -1, :] - beta
    # plug-in Mahalanobis at final n (each replication uses its OWN Sigma_hat)
    Ta = n_final * _maha_plugin(da_f, Sig_a[:, -1])
    Ts = n_final * _maha_plugin(ds_f, Sig_s[:, -1])
    M = bars_a.shape[0]
    nlabel = r"$n=10^{%d}$" % int(round(math.log10(n_final)))

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # (a) chi^2_d Q-Q at final n -- DATA-DRIVEN plug-in CIs (no oracle)
    pp = (np.arange(1, M + 1) - 0.5) / M
    q = chi2.ppf(pp, d)
    ax[0].plot(q, np.sort(Ts), ".", ms=3, alpha=.6, label="SGD (plug-in)")
    ax[0].plot(q, np.sort(Ta), ".", ms=3, alpha=.6, label="SA-Adam (plug-in)")
    lim = [0, max(q.max(), Ta.max(), Ts.max()) * 1.02]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel=r"$\chi^2_d$ quantile", ylabel=r"sample $T_n$ quantile",
              title=fr"(a) Plug-in $T_n$ vs $\chi^2_{{{d}}}$  ({nlabel})", xlim=lim, ylim=lim)
    ax[0].legend(frameon=True)

    # (b) coverage calibration at final n: plug-in (markers) vs oracle (thin)
    Ta_o = n_final * np.einsum("md,de,me->m", da_f, Sinv, da_f)
    Ts_o = n_final * np.einsum("md,de,me->m", ds_f, Sinv, ds_f)
    lv = np.linspace(0.50, 0.99, 30)
    ax[1].plot(lv, [np.mean(Ts <= chi2.ppf(p, d)) for p in lv], "-o", ms=3, label="SGD (plug-in)")
    ax[1].plot(lv, [np.mean(Ta <= chi2.ppf(p, d)) for p in lv], "-o", ms=3, label="SA-Adam (plug-in)")
    ax[1].plot(lv, [np.mean(Ta_o <= chi2.ppf(p, d)) for p in lv], "-", lw=1, color="gray",
               alpha=.7, label="SA-Adam (oracle)")
    ax[1].plot([0.5, 1], [0.5, 1], "k--", lw=1)
    ax[1].set(xlabel="nominal level", ylabel="empirical joint coverage",
              title=fr"(b) Wald-CI calibration  ({nlabel})")
    ax[1].legend(frameon=True)

    # (c) joint 95% coverage vs n: plug-in (solid) vs oracle (dashed)
    cov_a_p, cov_s_p, cov_a_o, cov_s_o = [], [], [], []
    for k, n in enumerate(ck):
        da = bars_a[:, k, :] - beta; ds = bars_s[:, k, :] - beta
        c = chi2.ppf(0.95, d)
        cov_a_p.append(np.mean(n * _maha_plugin(da, Sig_a[:, k]) <= c))
        cov_s_p.append(np.mean(n * _maha_plugin(ds, Sig_s[:, k]) <= c))
        cov_a_o.append(np.mean(n * np.einsum("md,de,me->m", da, Sinv, da) <= c))
        cov_s_o.append(np.mean(n * np.einsum("md,de,me->m", ds, Sinv, ds) <= c))
    cov_a_p = np.array(cov_a_p); cov_s_p = np.array(cov_s_p)
    err_a = 1.96 * np.sqrt(cov_a_p * (1 - cov_a_p) / M)
    err_s = 1.96 * np.sqrt(cov_s_p * (1 - cov_s_p) / M)
    ax[2].errorbar(ck, cov_s_p, yerr=err_s, fmt="-o", ms=4, capsize=3, label="SGD (plug-in)")
    ax[2].errorbar(ck, cov_a_p, yerr=err_a, fmt="-o", ms=4, capsize=3, label="SA-Adam (plug-in)")
    ax[2].plot(ck, cov_s_o, ":", color="C0", alpha=.7, label="SGD (oracle)")
    ax[2].plot(ck, cov_a_o, ":", color="C1", alpha=.7, label="SA-Adam (oracle)")
    ax[2].set_xscale("log")
    ax[2].axhline(0.95, color="k", ls="--", lw=1)
    ax[2].set(xlabel="$n$", ylabel="empirical joint 95% coverage",
              title="(c) Coverage vs $n$ (95% MC bands)", ylim=(0.80, 1.0))
    ax[2].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_pdf)


def main():
    ap = argparse.ArgumentParser(description="Experiment 3: semi-synthetic coverage")
    ap.add_argument("--n", type=int, default=100_000_000, help="stream length (power of ten)")
    ap.add_argument("--M", type=int, default=1000, help="number of replications / seeds")
    ap.add_argument("--cores", type=int, default=7, help="worker processes")
    ap.add_argument("--check", action="store_true",
                    help="quick smoke run (n=1e4, M=200); writes *_check outputs so the "
                         "publication figure is never overwritten")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output filenames (e.g. --tag _n1e7); keeps "
                         "verification runs from overwriting the publication figure")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from the cached arrays (no simulation)")
    args = ap.parse_args()

    n, M, cores = args.n, args.M, args.cores
    base = "exp3_coverage" + args.tag
    if args.check:
        n, M, base = 10_000, 200, "exp3_coverage_check"

    ck = checkpoints_for(n)
    figs_dir = os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)
    out_pdf = os.path.join(figs_dir, base + ".pdf")
    out_csv = os.path.join(figs_dir, base + "_summary.csv")
    replot_npz = os.path.join(figs_dir, base + "_replot.npz")

    if args.replot:
        z = np.load(replot_npz)
        make_figure(z["bars_a"], z["bars_s"], z["Sig_a"], z["Sig_s"],
                    z["beta"], z["Sigma"], int(z["d"]), z["ck"], out_pdf)
        print(f"replotted from cache -> {out_pdf}")
        return

    A, beta, sigma2, H, S, Sigma, d, N = build_population()
    eta0_sgd, eta0_adam = step_sizes(H, S)
    from scipy.stats import chi2
    print(f"d={d}, N(pool)={N}, n={n}, M={M}, alpha={ALPHA}, gamma={GAMMA}, cores={cores}")
    print(f"eta0_sgd={eta0_sgd:.4f}, eta0_adam={eta0_adam:.4f}, "
          f"cond(H)={np.linalg.cond(H):.2f}")
    print(f"||S - tr(S)/tr(H) H||/||S|| = "
          f"{np.linalg.norm(S - (np.trace(S)/np.trace(H))*H)/np.linalg.norm(S):.3f} "
          f"(>0 confirms S not proportional to H)")

    # compile the JIT kernel once (writes the numba disk cache the workers reuse)
    _run_kernel(np.ascontiguousarray(A), A @ beta, np.sqrt(sigma2), d, N, 5,
                C1, GAMMA, C2, ALPHA, eta0_adam, eta0_sgd, EPS,
                np.array([1, 5], dtype=np.int64), 0)

    seeds = (SEED0 + np.arange(M)).tolist()
    tasks = [(int(s), n, ck) for s in seeds]
    with Pool(processes=cores, initializer=init_worker) as pool:
        out = pool.map(run_rep, tasks, chunksize=max(1, M // (cores * 4)))

    bars_a = np.stack([o[0] for o in out])     # (M, n_ck, d)
    bars_s = np.stack([o[1] for o in out])
    H_hat  = np.stack([o[2] for o in out])     # (M, n_ck, d, d) plug-in Hessian
    Sa_hat = np.stack([o[3] for o in out])     # (M, n_ck, d, d) plug-in S (SA-Adam)
    Ss_hat = np.stack([o[4] for o in out])     # (M, n_ck, d, d) plug-in S (SGD)
    # per-replication plug-in sandwich Sigma_hat at each checkpoint, per arm
    Sig_a = np.stack([plugin_sigma(H_hat[:, k], Sa_hat[:, k]) for k in range(len(ck))], axis=1)
    Sig_s = np.stack([plugin_sigma(H_hat[:, k], Ss_hat[:, k]) for k in range(len(ck))], axis=1)
    methods = {"SA-Adam": (bars_a, Sig_a), "SGD": (bars_s, Sig_s)}

    # cache the (small) figure arrays so the figure can be restyled without re-running
    np.savez(replot_npz, bars_a=bars_a, bars_s=bars_s, Sig_a=Sig_a, Sig_s=Sig_s,
             beta=beta, Sigma=Sigma, d=d, ck=np.array(ck))

    # ---- summary CSV (oracle vs single-pass plug-in, side by side) ----
    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["method", "n", "bias_norm",
                       "orc_maha_median", "orc_joint95", "orc_marg95",
                       "plg_maha_median", "plg_joint95", "plg_marg95",
                       "sigmahat_relerr"])
        for name, (bars, Sig) in methods.items():
            for k, ni in enumerate(ck):
                o = summarize(bars[:, k, :], beta, Sigma, ni)              # oracle Sigma
                p = summarize_plugin(bars[:, k, :], beta, Sig[:, k, :], ni)  # data-driven Sigma_hat
                wcsv.writerow([name, o["n"], f'{o["bias_norm"]:.4g}',
                               f'{o["maha_median"]:.3f}', f'{o["joint_cov95"]:.3f}',
                               f'{o["marg_cov95_mean"]:.3f}',
                               f'{p["maha_median"]:.3f}', f'{p["joint_cov95"]:.3f}',
                               f'{p["marg_cov95_mean"]:.3f}', f'{p["sigmahat_relerr"]:.4g}'])

    # ---- figure ----
    make_figure(bars_a, bars_s, Sig_a, Sig_s, beta, Sigma, d, ck, out_pdf)

    # ---- console summary (final n) ----
    chi2_med = chi2.ppf(0.5, d)
    print(f"\n=== final (n = {int(ck[-1])}) :  oracle  vs  single-pass plug-in ===")
    for name, (bars, Sig) in methods.items():
        o = summarize(bars[:, -1, :], beta, Sigma, int(ck[-1]))
        p = summarize_plugin(bars[:, -1, :], beta, Sig[:, -1, :], int(ck[-1]))
        print(f"{name:8s}: bias={o['bias_norm']:.2e}  "
              f"medT_n[orc/plg]={o['maha_median']:.2f}/{p['maha_median']:.2f} "
              f"(chi2_{d} med {chi2_med:.2f})  "
              f"joint95[orc/plg]={o['joint_cov95']:.3f}/{p['joint_cov95']:.3f}  "
              f"marg95[orc/plg]={o['marg_cov95_mean']:.3f}/{p['marg_cov95_mean']:.3f}  "
              f"Sigma_hat relerr={p['sigmahat_relerr']:.3f}")
    print(f"\nfigure -> {os.path.normpath(out_pdf)}")
    print(f"summary -> {os.path.normpath(out_csv)}")


if __name__ == "__main__":
    main()
