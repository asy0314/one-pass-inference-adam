#!/usr/bin/env python
"""Experiment 4 -- One real large-stream logistic-regression
experiment: one-pass Wald inference vs. the stored-batch sandwich.

A single pass over a LARGE real stream (SUSY: n=5,000,000, 18 standardized
physics features + intercept; real labels) recovers the full-data ridge-logistic
coefficients AND calibrated Wald CIs, maintaining only O(d^2) running state -- the
SAME intervals the stored-batch sandwich produces from O(nd) storage and a
multi-pass solver, and additionally resolving each coefficient's significance
ONLINE as the stream arrives (which offline batch cannot).

Model: F_lambda(x) = (1/n) sum_t logloss(a_t^T x, y_t) + (lambda/2)||x||^2
(strongly convex; Corollary `cor:ridge`).  Methods:
  * SA-Adam (ours): one pass; decaying gains; online plug-in
    Sigma_hat = Hhat^{-1} Shat Hhat^{-1} -> Wald CIs.  State O(d^2).
  * Stored-batch sandwich (reference): one full-data fit + the plug-in sandwich.
    Storage O(nd); multi-pass Newton.
  * Adam (deployed EMA): same as ours with constant beta=0.9/0.999 (a probe).

Run:  python exp4_streaming_logistic.py                  # SUSY (needs /tmp/susy.npz)
      python exp4_streaming_logistic.py --dataset covtype
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import math
import csv
import time
import numpy as np
from numba import njit

ALPHA = 0.6
GAMMA = 0.75
C1 = 1.0
C2 = 1.0
EPS = 1e-8
B1_DEP = 0.9
B2_DEP = 0.999
SEED = 20260622
Z = 1.959964
SUSY_NPZ = "/tmp/susy.npz"


def load_dataset(name, nfeat):
    if name == "susy":
        z = np.load(SUSY_NPZ)
        X = z["X"].astype(np.float64)
        y = z["y"].astype(np.float64)
        u = np.unique(y); y = (y == u[-1]).astype(np.float64)   # -> {0,1}
    elif name == "covtype":
        from sklearn.datasets import fetch_covtype
        dd = fetch_covtype()
        X = dd.data[:, :nfeat].astype(np.float64)
        y = (dd.target == 2).astype(np.float64)
    else:
        raise ValueError(name)
    X = (X - X.mean(0)) / X.std(0)
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.ascontiguousarray(A), y


def batch_fit(A, y, lam, iters=100, tol=1e-11):
    n, d = A.shape
    x = np.zeros(d)
    for it in range(iters):
        s = 1.0 / (1.0 + np.exp(-np.clip(A @ x, -30, 30)))
        W = s * (1.0 - s)
        g = A.T @ (s - y) / n + lam * x
        H = (A.T * W) @ A / n + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        x -= step
        if np.linalg.norm(step) < tol:
            break
    return x, it + 1


def sandwich_at(A, y, x, lam):
    n, d = A.shape
    s = 1.0 / (1.0 + np.exp(-np.clip(A @ x, -30, 30)))
    W = s * (1.0 - s)
    H = (A.T * W) @ A / n + lam * np.eye(d)
    G = (s - y)[:, None] * A + lam * x[None, :]
    S = G.T @ G / n
    Hi = np.linalg.inv(H)
    return Hi @ S @ Hi, H, S


def adam_eta0(H, S, safety):
    Dm = np.diag(np.diag(S) ** -0.25)
    return safety / float(np.linalg.eigvalsh(Dm @ H @ Dm).max())


@njit(cache=True)
def stream_arm(A, y, n, d, lam, eta0, alpha, mode, c1, gamma, c2, b1c, b2c, eps, cps, t0):
    """One streaming pass.  mode 0 = SA-Adam; mode 1 = deployed Adam.  Tail-averages
    from step t0; records (xbar, Hhat, Shat) at checkpoints cps.  Single pass, O(d^2)."""
    nck = cps.shape[0]
    x = np.zeros(d); m = np.zeros(d); v = np.zeros(d); sumx = np.zeros(d); grad = np.zeros(d)
    Hs = np.zeros((d, d)); Ss = np.zeros((d, d))
    rec_x = np.empty((nck, d)); rec_H = np.empty((nck, d, d)); rec_S = np.empty((nck, d, d))
    lb1 = 0.0; lb2 = 0.0; cptr = 0
    for t in range(1, n + 1):
        i = t - 1
        z = 0.0
        for j in range(d):
            z += A[i, j] * x[j]
        if z > 30.0:
            s = 1.0
        elif z < -30.0:
            s = 0.0
        else:
            s = 1.0 / (1.0 + math.exp(-z))
        w = s * (1.0 - s); sy = s - y[i]
        for j in range(d):
            grad[j] = sy * A[i, j] + lam * x[j]
        if t > t0:
            for j in range(d):
                aij = A[i, j]; gj = grad[j]
                for k in range(d):
                    Hs[j, k] += w * aij * A[i, k]
                    Ss[j, k] += gj * grad[k]
                Hs[j, j] += lam
        eta = eta0 * t ** (-alpha)
        if mode == 0:
            b1 = 1.0 - c1 * t ** (-gamma)
            b2 = 1.0 - c2 / t
        else:
            b1 = b1c; b2 = b2c
        lb1 += math.log(b1); lb2 += math.log(b2)
        km = 1.0 - math.exp(lb1); kv = 1.0 - math.exp(lb2)
        if km < 1e-15:
            km = 1e-15
        if kv < 1e-15:
            kv = 1e-15
        for j in range(d):
            m[j] = b1 * m[j] + (1.0 - b1) * grad[j]
            v[j] = b2 * v[j] + (1.0 - b2) * grad[j] * grad[j]
            x[j] -= eta * (m[j] / km) / (math.sqrt(v[j] / kv) + eps)
            if t > t0:
                sumx[j] += x[j]
        if cptr < nck and t == cps[cptr]:
            inv = 1.0 / (t - t0)
            for j in range(d):
                rec_x[cptr, j] = sumx[j] * inv
                for k in range(d):
                    rec_H[cptr, j, k] = Hs[j, k] * inv
                    rec_S[cptr, j, k] = Ss[j, k] * inv
            cptr += 1
    return rec_x, rec_H, rec_S


def plugin_sigma(H, S):
    Hi = np.linalg.inv(H)
    return Hi @ S @ Hi


def make_figure(xb, hw_b, est_sa, hw_sa, cps, x_tr, se_tr, n, d, mem_one, mem_batch,
                t_stream, batch_passes, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15})
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # (a) forest: one-pass vs stored-batch sandwich CIs
    order = np.argsort(xb)
    yy = np.arange(d)
    ax[0].errorbar(xb[order], yy + 0.15, xerr=hw_b[order], fmt="o", ms=4, capsize=2,
                   color="k", label="stored-batch sandwich")
    ax[0].errorbar(est_sa[order], yy - 0.15, xerr=hw_sa[order], fmt="s", ms=4, capsize=2,
                   color="C1", label="SA-Adam (one-pass)")
    ax[0].axvline(0, color="gray", lw=1, ls=":")
    ax[0].set_yticks(yy); ax[0].set_yticklabels([f"$x_{{{j}}}$" for j in order], fontsize=10)
    ax[0].set(xlabel="coefficient (95% CI)", title="(a) One-pass CIs $=$ stored-batch CIs")
    ax[0].legend(frameon=True, fontsize=15, loc="lower right")

    # (b) online time-to-significance
    tts = np.full(d, np.nan)
    for j in range(d):
        sig = np.abs(x_tr[:, j]) > Z * se_tr[:, j]
        for k in range(len(cps)):
            if sig[k] and sig[k:].all():
                tts[j] = cps[k]; break
    o2 = np.argsort(tts)
    ax[1].hlines(np.arange(d), cps[0], tts[o2], color="C1", lw=1.4, alpha=.6)
    ax[1].plot(tts[o2], np.arange(d), "o", ms=5, color="C1")
    ax[1].axvline(n, color="k", ls="--", lw=1.2)
    ax[1].text(n, 0.5, "batch:\noffline,\nall at $n$ ", fontsize=14, va="bottom", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_yticks(np.arange(d)); ax[1].set_yticklabels([f"$x_{{{j}}}$" for j in o2], fontsize=10)
    ax[1].set(xlabel="samples seen when CI first excludes 0",
              title="(b) Online: features resolve as data streams in")

    # (c) memory scaling: O(d^2) flat vs O(nd) -> batch infeasible at production scale
    ns = np.logspace(5, 9.3, 60)
    batch_mb = ns * d * 8 / 1024**2
    one_mb = mem_one / 1024**2
    ax[2].loglog(ns, batch_mb, color="k", lw=2, label="stored-batch  $O(nd)$")
    ax[2].axhline(one_mb, color="C1", lw=2, label="one-pass state  $O(d^2)$")
    ax[2].axhline(16 * 1024, color="r", ls=":", lw=1.5, label="16 GB (commodity RAM)")
    ax[2].axvline(n, color="gray", ls="--", lw=1)
    ax[2].text(n, batch_mb[0], f" $n{{=}}${n/1e6:.0f}M\n (here)", fontsize=12, va="bottom")
    ax[2].set(xlabel="stream length $n$", ylabel="memory (MB)",
              title="(c) Memory: batch grows, one-pass is flat")
    ax[2].legend(frameon=True, fontsize=15, loc="lower left", bbox_to_anchor=(0.0, 0.07))
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"figure -> {os.path.normpath(out_pdf)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="susy", choices=["susy", "covtype"])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--nsub", type=int, default=0)
    ap.add_argument("--nfeat", type=int, default=10)
    ap.add_argument("--eta_safety", type=float, default=12.0)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    A, y = load_dataset(args.dataset, args.nfeat)
    if args.nsub and args.nsub < A.shape[0]:
        idx = rng.choice(A.shape[0], args.nsub, replace=False)
        A = np.ascontiguousarray(A[idx]); y = y[idx].copy()
    n, d = A.shape
    lam = args.lam
    print(f"{args.dataset}: n={n:,}, d={d}, lambda={lam}, pos frac={y.mean():.3f}")

    # ---- stored-batch reference ----
    t = time.time()
    xb, iters = batch_fit(A, y, lam)
    Sig_b, Hb, Sb = sandwich_at(A, y, xb, lam)
    t_batch = time.time() - t
    se_b = np.sqrt(np.diag(Sig_b) / n)
    eta0 = adam_eta0(Hb, Sb, args.eta_safety)
    print(f"batch: {iters} Newton iters in {t_batch:.2f}s, ||xb||={np.linalg.norm(xb):.3f}, "
          f"eta0(adam)={eta0:.3f}")

    # ---- streaming (one pass; shuffle once) ----
    perm = rng.permutation(n)
    As = np.ascontiguousarray(A[perm]); ys = np.ascontiguousarray(y[perm])
    t0 = max(300, n // 200)
    neff = n - t0
    cps = np.unique(np.clip(np.round(np.geomspace(2 * t0, n, 28)).astype(np.int64), 1, n))
    stream_arm(As[:6], ys[:6], 6, d, lam, eta0, ALPHA, 0, C1, GAMMA, C2,
               B1_DEP, B2_DEP, EPS, np.array([6], dtype=np.int64), 2)

    ests, ses, dt = {}, {}, {}
    rec_sa = None
    for mode, name in ((0, "SA-Adam"), (1, "Adam(deployed)")):
        tt = time.time()
        rx, rH, rS = stream_arm(As, ys, n, d, lam, eta0, ALPHA, mode, C1, GAMMA, C2,
                                B1_DEP, B2_DEP, EPS, cps, t0)
        dt[name] = time.time() - tt
        ests[name] = rx[-1]
        ses[name] = np.sqrt(np.diag(plugin_sigma(rH[-1], rS[-1])) / neff)
        if mode == 0:
            rec_sa = (rx, rH, rS)
    rx, rH, rS = rec_sa
    se_tr = np.empty((len(cps), d))
    for k in range(len(cps)):
        se_tr[k] = np.sqrt(np.diag(plugin_sigma(rH[k], rS[k])) / (cps[k] - t0))

    # ---- metrics ----
    sig_b = np.abs(xb) > Z * se_b
    print(f"\n=== one-pass vs stored-batch sandwich (n={n:,}, d={d}) ===")
    print(f"{'arm':<16}{'est vs batch':>14}{'CIhw vs batch':>16}{'sig agree':>12}{'pass time':>12}")
    for name in ("SA-Adam", "Adam(deployed)"):
        rel = np.linalg.norm(ests[name] - xb) / np.linalg.norm(xb)
        hw = Z * ses[name]
        relhw = float(np.median(np.abs(hw - Z * se_b) / (Z * se_b)))
        agree = int(np.sum((np.abs(ests[name]) > hw) == sig_b))
        print(f"{name:<16}{rel:>14.4f}{relhw:>16.4f}{agree:>9}/{d}{dt[name]:>11.2f}s")

    mem_one = (2 * d * d + 5 * d) * 8
    mem_batch = n * d * 8
    n_ram = 16 * 1024**3 / (d * 8)
    print(f"\nmemory: one-pass state {mem_one/1024:.1f} KB (O(d^2))  vs  "
          f"stored design {mem_batch/1024**2:.0f} MB (O(nd))  ->  {mem_batch/mem_one:,.0f}x")
    print(f"passes: 1 (streaming) vs {iters} (Newton); batch storage hits 16 GB at n={n_ram:.1e}")

    figs = os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(figs, exist_ok=True)
    make_figure(xb, Z * se_b, ests["SA-Adam"], Z * ses["SA-Adam"], cps, rx, se_tr,
                n, d, mem_one, mem_batch, dt["SA-Adam"], iters,
                os.path.join(figs, "exp4_streaming_logistic.pdf"))
    with open(os.path.join(figs, "exp4_streaming_logistic_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["coord", "batch_est", "batch_cihw", "saadam_est", "saadam_cihw"])
        for j in range(d):
            wr.writerow([j, f"{xb[j]:.5f}", f"{Z*se_b[j]:.5f}",
                         f"{ests['SA-Adam'][j]:.5f}", f"{Z*ses['SA-Adam'][j]:.5f}"])


if __name__ == "__main__":
    main()
