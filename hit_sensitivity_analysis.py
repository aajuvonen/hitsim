"""
hit_sensitivity_analysis.py

Sensitivity analysis for HIT.

This script synthesizes context-response-latency tuples and computes
the HIT score under controlled parameter sweeps. use matplotlib (no
seaborn), single-plot figures, and do not specify colors or styles.

Author : A. Artturi Juvonen
Date   : 2025-10-23
"""


import numpy as np
import pandas as pd
from itertools import product
from math import log2, isfinite
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)  # reproducible

def entropy_empirical(counts, base=2):
    """Plug-in entropy from counts (discrete)."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    p = p[p > 0]
    log = np.log2 if base == 2 else np.log
    return -np.sum(p * log(p))

def miller_madow_correction(k_nonzero, n):
    """Miller-Madow bias correction term for entropy (in bits)."""
    if n <= 0 or k_nonzero <= 0:
        return 0.0
    return (k_nonzero - 1) / (2 * n) / np.log(2)  # convert nats->bits

def entropy_mm(counts):
    """Miller-Madow corrected entropy (bits)."""
    n = counts.sum()
    h = entropy_empirical(counts, base=2)
    k = (counts > 0).sum()
    return h + miller_madow_correction(k, n)

def joint_counts(x, y, kx, ky):
    """Return joint contingency table counts for discrete x in [0,kx), y in [0,ky)."""
    m = np.zeros((kx, ky), dtype=int)
    for a, b in zip(x, y):
        m[a, b] += 1
    return m

def mutual_information_from_counts(n_xy, mm=False):
    """Compute I(X;Y) from joint counts. Optionally Miller-Madow via entropy parts."""
    n = n_xy.sum()
    if n == 0:
        return 0.0
    n_x = n_xy.sum(axis=1)
    n_y = n_xy.sum(axis=0)
    if mm:
        Hx = entropy_mm(n_x)
        Hy = entropy_mm(n_y)
        Hxy = entropy_mm(n_xy.flatten())
    else:
        Hx = entropy_empirical(n_x, base=2)
        Hy = entropy_empirical(n_y, base=2)
        Hxy = entropy_empirical(n_xy.flatten(), base=2)
    return Hx + Hy - Hxy

def sample_context(n, kC, dist='uniform', alpha=1.2):
    """Sample contexts 0..kC-1 from uniform or Zipf-like (truncated power-law over finite support)."""
    if dist == 'uniform':
        p = np.ones(kC) / kC
    else:
        # Zipf-like over finite support: p(i) ~ (i+1)^(-alpha), sorted by rank
        ranks = np.arange(1, kC+1)
        p = ranks ** (-alpha)
        p = p / p.sum()
    return rng.choice(kC, size=n, p=p), p

def sample_policy_responses(C, kR, accuracy=1.0):
    """
    Construct a simple 'correct action' mapping and inject errors.
    - Map each context c to a canonical action a = c mod kR.
    - With prob=accuracy choose the canonical action, else a random wrong action.
    """
    n = len(C)
    canonical = C % kR
    if accuracy >= 1.0:
        return canonical
    out = canonical.copy()
    flip_mask = rng.random(n) > accuracy
    # choose wrong action uniformly from remaining kR-1 actions
    for idx in np.where(flip_mask)[0]:
        wrong_choices = [a for a in range(kR) if a != canonical[idx]]
        out[idx] = rng.choice(wrong_choices)
    return out

def sample_latency(n, mode='fixed', base=1.0):
    """Generate latency samples: fixed (base), jittered (+/-1 around base), Poisson with mean base."""
    if mode == 'fixed':
        T = np.full(n, base)
    elif mode == 'jittered':
        # integer jitter ±1 around base, lower bounded at 0.5 to avoid zero
        jitter = rng.integers(-1, 2, size=n)
        T = np.maximum(0.5, base + jitter)
    elif mode == 'poisson':
        # Poisson with mean=base, shift by 0.5 to avoid zeros, as time/cost should be positive
        T = rng.poisson(lam=base, size=n).astype(float) + 0.5
    else:
        raise ValueError("Unknown latency mode")
    return T

def compute_hit_from_samples(C, R, T, use_mm=False):
    """Compute normalized Hbar, Ibar, Tbar and HIT = (Hbar * Ibar) / Tbar for one configuration."""
    kC = int(C.max()) + 1 if len(C) else 0
    kR = int(R.max()) + 1 if len(R) else 0
    # Entropy of C
    counts_C = np.bincount(C, minlength=kC)
    Hc = entropy_mm(counts_C) if use_mm else entropy_empirical(counts_C, base=2)
    Hc_max = np.log2(kC) if kC > 0 else 1.0
    Hbar = Hc / Hc_max if Hc_max > 0 else 0.0
    # Mutual information I(C;R)
    n_cr = joint_counts(C, R, kC, kR)
    Icr = mutual_information_from_counts(n_cr, mm=use_mm)
    Imax = min(np.log2(kC) if kC>0 else 0.0, np.log2(kR) if kR>0 else 0.0)
    Ibar = Icr / Imax if Imax > 0 else 0.0
    # Latency normalization by max observed latency in this configuration
    T = np.asarray(T, dtype=float)
    Tbar = T.mean() / T.max() if T.size > 0 else 1.0
    # HIT
    HIT = (Hbar * Ibar) / Tbar if Tbar > 0 else 0.0
    return dict(Hc=Hc, Hbar=Hbar, Icr=Icr, Ibar=Ibar, Tmean=T.mean(), Tmax=T.max(), Tbar=Tbar, HIT=HIT)

# Sweep parameters (reduced grid for runtime; extensible)
kC_vals = [3, 10]
kR_vals = [3, 10]
dists = [('uniform', None), ('zipf', 1.2)]
accuracies = [1.0, 0.8, 0.6]
latency_modes = ['fixed', 'jittered', 'poisson']
sample_sizes = [300, 1000]

rows = []
for kC, kR, (dist, alpha), acc, lmode, n in product(kC_vals, kR_vals, dists, accuracies, latency_modes, sample_sizes):
    # sample contexts
    C, pC = sample_context(n, kC, dist=('uniform' if dist=='uniform' else 'zipf'), alpha=(alpha if alpha else 1.2))
    # responses per policy
    R = sample_policy_responses(C, kR, accuracy=acc)
    # latency
    T = sample_latency(n, mode=lmode, base=1.0)
    # compute metrics (plugin and MM-corrected)
    res_naive = compute_hit_from_samples(C, R, T, use_mm=False)
    res_mm = compute_hit_from_samples(C, R, T, use_mm=True)
    rows.append({
        'kC': kC, 'kR': kR, 'dist': dist if dist=='uniform' else f'zipf-{alpha}',
        'accuracy': acc, 'latency': lmode, 'n': n,
        **{f'{k}': v for k, v in res_naive.items()},
        **{f'{k}_mm': v for k, v in res_mm.items()},
    })

df = pd.DataFrame(rows)

# Save raw results
csv_path = './data/hit_sensitivity_results.csv'
df.to_csv(csv_path, index=False)

# Summaries to mirror the claims in the user's earlier text

def summarize_effect_of_kR(df):
    # Compare HIT when kR=3 vs kR=10 under matched kC, dist, accuracy, latency, n
    keys = ['kC','dist','accuracy','latency','n']
    # pivot
    sub = df.pivot_table(index=keys, columns='kR', values='HIT', aggfunc='mean')
    sub = sub.dropna()
    if {3,10}.issubset(set(sub.columns)):
        sub['delta_HIT_kR_10_vs_3'] = sub[10] - sub[3]
        sub['rel_change'] = sub['delta_HIT_kR_10_vs_3'] / sub[3]
        return sub[['delta_HIT_kR_10_vs_3','rel_change']].reset_index()
    return pd.DataFrame()

def summarize_zipf_effect(df):
    # Compare zipf-1.2 vs uniform
    keys = ['kC','kR','accuracy','latency','n']
    sub = df[df['dist'].isin(['uniform','zipf-1.2'])]
    sub = sub.pivot_table(index=keys, columns='dist', values='HIT', aggfunc='mean')
    sub = sub.dropna()
    if {'uniform','zipf-1.2'}.issubset(set(sub.columns)):
        sub['delta_zipf_vs_uniform'] = sub['zipf-1.2'] - sub['uniform']
        sub['rel_change'] = sub['delta_zipf_vs_uniform'] / sub['uniform']
        return sub[['delta_zipf_vs_uniform','rel_change']].reset_index()
    return pd.DataFrame()

def summarize_accuracy_effect(df):
    # linearity: slope approx between 1.0, 0.8, 0.6
    keys = ['kC','kR','dist','latency','n']
    sums = []
    for _, g in df.groupby(keys):
        g2 = g[['accuracy','HIT']].sort_values('accuracy', ascending=False)
        if set(g2['accuracy']) == {1.0, 0.8, 0.6}:
            # compute relative change per 0.2 decrement
            hit1, hit08, hit06 = g2['HIT'].values
            rel1 = (hit08 - hit1)/hit1 if hit1 else np.nan
            rel2 = (hit06 - hit08)/hit08 if hit08 else np.nan
            sums.append({**{k:v for k,v in zip(keys, _)},
                         'rel_drop_1_to_08': rel1, 'rel_drop_08_to_06': rel2,
                         'abs_drop_1_to_06': hit06 - hit1})
    return pd.DataFrame(sums)

def summarize_latency_effect(df):
    # Compare latency modes at fixed other params
    keys = ['kC','kR','dist','accuracy','n']
    sub = df.pivot_table(index=keys, columns='latency', values=['Tmean','Tbar','HIT'], aggfunc='mean')
    sub = sub.dropna()
    return sub.reset_index()

sum_kR = summarize_effect_of_kR(df)
sum_zipf = summarize_zipf_effect(df)
sum_acc = summarize_accuracy_effect(df)
sum_lat = summarize_latency_effect(df)

# Save summaries
sum_kR_path = './data/summary_kR_effect.csv'
sum_zipf_path = './data/summary_zipf_effect.csv'
sum_acc_path = './data/summary_accuracy_effect.csv'
sum_lat_path = './data/summary_latency_effect.csv'

sum_kR.to_csv(sum_kR_path, index=False)
sum_zipf.to_csv(sum_zipf_path, index=False)
sum_acc.to_csv(sum_acc_path, index=False)
sum_lat.to_csv(sum_lat_path, index=False)

# A simple plot: HIT vs accuracy aggregated over settings, to verify near-linearity
acc_order = [1.0, 0.8, 0.6]
hit_by_acc = df.groupby('accuracy')['HIT'].mean().reindex(acc_order)
plt.figure()
plt.plot(acc_order, hit_by_acc.values, marker='o')
plt.xlabel('Policy accuracy')
plt.ylabel('Mean HIT')
plt.title('Mean HIT vs. policy accuracy (aggregated)')
plot_path = './data/hit_vs_accuracy.png'
plt.savefig(plot_path, bbox_inches='tight')
plot_path
