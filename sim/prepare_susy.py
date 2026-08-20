#!/usr/bin/env python
"""Build the SUSY cache consumed by exp4_streaming_logistic.py.

exp4 expects an .npz at ``/tmp/susy.npz`` (the module constant ``SUSY_NPZ``)
holding

    X : (5000000, 18) float  -- the 18 raw SUSY features, UCI column order
                                (8 low-level kinematic + 10 high-level derived)
    y : (5000000,)    float  -- the class label (1 = signal, 0 = background)

exp4 standardizes X itself and prepends the intercept, so this script does NO
preprocessing beyond the CSV -> array conversion; the coefficient order reported
in the paper is exactly the UCI feature order.

Data (not redistributed here; ~2.4 GB uncompressed, 900 MB gzipped):
    SUSY, UCI Machine Learning Repository, donated by Baldi, Sadowski &
    Whiteson (2014), https://archive.ics.uci.edu/dataset/279/susy
    Direct file: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
    The CSV is headerless: column 0 is the label, columns 1..18 the features.

Run:  python prepare_susy.py --csv /path/to/SUSY.csv.gz          # -> /tmp/susy.npz
      python prepare_susy.py --csv SUSY.csv.gz --out ~/susy.npz
"""
import argparse
import gzip
import os
import numpy as np

NFEAT = 18


def read_csv(path):
    """Return (X, y). Uses pandas when available (~1 min); falls back to a
    chunked numpy parse (slower but dependency-free)."""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        df = pd.read_csv(path, header=None, dtype=np.float64)
        arr = df.to_numpy()
        return arr[:, 1:NFEAT + 1], arr[:, 0]

    print("pandas not found -- falling back to a chunked numpy parse "
          "(several minutes)")
    opener = gzip.open if str(path).endswith(".gz") else open
    rows, chunk = [], []
    with opener(path, "rt") as fh:
        for line in fh:
            chunk.append(line)
            if len(chunk) == 200_000:
                rows.append(np.loadtxt(chunk, delimiter=",", dtype=np.float64))
                chunk = []
                print(f"  {sum(r.shape[0] for r in rows):,} rows")
    if chunk:
        rows.append(np.loadtxt(chunk, delimiter=",", dtype=np.float64))
    arr = np.vstack(rows)
    return arr[:, 1:NFEAT + 1], arr[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="path to SUSY.csv or SUSY.csv.gz from the UCI repository")
    ap.add_argument("--out", default="/tmp/susy.npz",
                    help="output .npz (must match SUSY_NPZ in exp4_streaming_logistic.py)")
    args = ap.parse_args()

    X, y = read_csv(args.csv)
    if X.shape[1] != NFEAT:
        raise SystemExit(f"expected {NFEAT} feature columns, got {X.shape[1]}")
    print(f"SUSY: n={X.shape[0]:,}, d={X.shape[1]}, pos frac={y.mean():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, X=X.astype(np.float32), y=y.astype(np.float32))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
