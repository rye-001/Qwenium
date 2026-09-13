#!/usr/bin/env python3
"""Closed-form draft head probe — fit stage (numpy, no gradients).

Reads the binary dump from draft_head_capture.cpp (per-prompt sequences of
(final_norm hidden h_i, generated token y_i)), builds (h_t, x_{t+1}, x_{t+2},
x_{t+3}) tuples, ridge-fits W: h_t -> h_{t+1} and W2: h_t -> h_{t+2} on a
random 80/20 split by POSITION (row), and writes the held-out predictions +
true labels to a binary file for draft_head_score.cpp to score with the
model's real output head. See docs/note-closed-form-draft-head.md.

Usage: python3 fit_draft_heads.py <capture.bin> <score_input.bin>
"""
import struct
import sys
import numpy as np


def read_capture(path):
    with open(path, "rb") as f:
        hidden, vocab, num_prompts = struct.unpack("<iii", f.read(12))
        prompts = []
        for _ in range(num_prompts):
            (n_records,) = struct.unpack("<i", f.read(4))
            hs = np.empty((n_records, hidden), dtype=np.float32)
            ys = np.empty((n_records,), dtype=np.int32)
            for r in range(n_records):
                hs[r] = np.frombuffer(f.read(hidden * 4), dtype=np.float32)
                (ys[r],) = struct.unpack("<i", f.read(4))
            prompts.append((hs, ys))
    return hidden, vocab, prompts


def build_samples(prompts):
    """For each prompt, i ranges over positions with i+2 valid.
    h_t = h[i]; x_{t+1}=y[i]; x_{t+2}=y[i+1]; x_{t+3}=y[i+2];
    target for W (h_t->h_{t+1}) = h[i+1]; target for W2 (h_t->h_{t+2}) = h[i+2].
    """
    H, T1, T2, L2, L3, L1 = [], [], [], [], [], []
    for hs, ys in prompts:
        L = len(ys)
        for i in range(L - 2):
            H.append(hs[i])
            T1.append(hs[i + 1])
            T2.append(hs[i + 2])
            L1.append(ys[i])
            L2.append(ys[i + 1])
            L3.append(ys[i + 2])
    return (np.stack(H), np.stack(T1), np.stack(T2),
            np.array(L1), np.array(L2), np.array(L3))


def ridge_fit(X, Y, alpha):
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d, dtype=np.float64)
    B = X.T @ Y
    M = np.linalg.solve(A, B)
    return M  # predict via X @ M


def main():
    if len(sys.argv) != 3:
        print("usage: fit_draft_heads.py <capture.bin> <score_input.bin>", file=sys.stderr)
        sys.exit(1)
    capture_path, score_path = sys.argv[1], sys.argv[2]

    hidden, vocab, prompts = read_capture(capture_path)
    total = sum(len(ys) for _, ys in prompts)
    print(f"hidden={hidden} vocab={vocab} num_prompts={len(prompts)} "
          f"total_records={total}")
    for i, (hs, ys) in enumerate(prompts):
        print(f"  prompt {i}: {len(ys)} records")

    H, T1, T2, L1, L2, L3 = build_samples(prompts)
    n = H.shape[0]
    print(f"usable (h_t, x_t+1, x_t+2, x_t+3) samples: {n} (hidden dim {hidden})")
    if n < hidden:
        print(f"UNDERDETERMINED: {n} samples < hidden dim {hidden} — "
              f"reporting nothing further, this is not a finding.")
        sys.exit(0)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_test = max(1, int(0.2 * n))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    print(f"train={len(train_idx)} test={len(test_idx)} (80/20 split by position, seed=0)")

    Xtr, Xte = H[train_idx].astype(np.float64), H[test_idx].astype(np.float64)
    T1tr = T1[train_idx].astype(np.float64)
    T2tr = T2[train_idx].astype(np.float64)
    L2te, L3te = L2[test_idx], L3[test_idx]
    L1te, L2tr_labels = L1[test_idx], L2[train_idx]

    alpha = 10.0  # fixed ridge strength; hidden-space regression, not tuned
    W = ridge_fit(Xtr, T1tr, alpha)
    W2 = ridge_fit(Xtr, T2tr, alpha)
    print(f"ridge alpha={alpha}")

    pred1_te = (Xte @ W).astype(np.float32)   # predicted h_{t+1} -> scores p2
    pred2_te = (Xte @ W2).astype(np.float32)  # predicted h_{t+2} -> scores p3

    # Baselines (trivial, no ggml needed):
    baseline_repeat = float(np.mean(L2te == L1te))  # x_{t+2} == x_{t+1}?
    mode_token = np.bincount(L2[train_idx]).argmax()
    baseline_mode2 = float(np.mean(L2te == mode_token))
    mode_token3 = np.bincount(L3[train_idx]).argmax()
    baseline_mode3 = float(np.mean(L3te == mode_token3))
    print(f"baseline x_t+2==x_t+1: {baseline_repeat:.4f}")
    print(f"baseline x_t+2==most-frequent-train-token: {baseline_mode2:.4f}")
    print(f"baseline x_t+3==most-frequent-train-token: {baseline_mode3:.4f}")

    with open(score_path, "wb") as f:
        f.write(struct.pack("<iii", hidden, vocab, len(test_idx)))
        for row, lbl in zip(pred1_te, L2te):
            f.write(row.tobytes())
            f.write(struct.pack("<i", int(lbl)))
        for row, lbl in zip(pred2_te, L3te):
            f.write(row.tobytes())
            f.write(struct.pack("<i", int(lbl)))
    print(f"wrote {score_path}")


if __name__ == "__main__":
    main()
