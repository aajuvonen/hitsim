"""
hit_7agent_ipd.py
HIT Benchmark. 7-Agent Iterated Prisoner's Dilemma
Strategies include: 5 classical + WSLS-reactive + WSLS-payoff
Seed: 42, Rounds: 2000

Author : A. Artturi Juvonen
Date   : 2025-05-13
"""

import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from sklearn.metrics import mutual_info_score

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
np.random.seed(42)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def entropy_bits(seq):
    """Shannon entropy for binary sequences (bits)."""
    if len(seq) == 0:
        return 0.0
    probs = np.bincount(seq, minlength=2) / len(seq)
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()

def mi_bits(ctx, rsp):
    """Mutual information between binary context and response (bits)."""
    if len(ctx) == 0:
        return 0.0
    return mutual_info_score(ctx, rsp) / np.log(2)  # convert nats → bits

# ------------------------------------------------------------------
# Strategy definitions
# ------------------------------------------------------------------
def always_cooperate(_, __):
    return 1

def always_defect(_, __):
    return 0

def tit_for_tat(opp_hist, _):
    return 1 if not opp_hist else opp_hist[-1]

def random_strategy(_, __):
    return np.random.choice([0, 1])

def grim_trigger(opp_hist, _):
    return 0 if 0 in opp_hist else 1

def wsls_reactive(opp_hist, self_hist):
    """Repeat if last round matched, else switch."""
    if not self_hist:
        return 1
    return self_hist[-1] if self_hist[-1] == opp_hist[-1] else 1 - self_hist[-1]

def wsls_payoff(opp_hist, self_hist):
    """Payoff-based WSLS: repeat if mutual coop or mutual defect."""
    if not self_hist:
        return 1
    if self_hist[-1] == opp_hist[-1]:  # mutual coop (1,1) or mutual defect (0,0)
        return self_hist[-1]
    else:
        return 1 - self_hist[-1]

# ------------------------------------------------------------------
# Agent list
# ------------------------------------------------------------------
strategies = {
    "Always Cooperate":        always_cooperate,
    "Always Defect":           always_defect,
    "Tit-for-Tat":             tit_for_tat,
    "Random":                  random_strategy,
    "Grim Trigger":            grim_trigger,
    "WSLS-reactive":           wsls_reactive,
    "WSLS-payoff":             wsls_payoff
}
agents = list(strategies.keys())
rounds = 2000

# ------------------------------------------------------------------
# Simulation setup
# ------------------------------------------------------------------
opp_hist = {a: defaultdict(list) for a in agents}
self_hist= {a: defaultdict(list) for a in agents}
my_act   = defaultdict(list)  # (i,j) → i’s action vs j
opp_act  = defaultdict(list)  # (i,j) → j’s action vs i

for _ in range(rounds):
    for i, j in itertools.combinations(agents, 2):
        ai = strategies[i](opp_hist[i][j], self_hist[i][j])
        aj = strategies[j](opp_hist[j][i], self_hist[j][i])

        # update histories
        opp_hist[i][j].append(aj);  opp_hist[j][i].append(ai)
        self_hist[i][j].append(ai); self_hist[j][i].append(aj)

        # record actions
        my_act [(i, j)].append(ai); my_act [(j, i)].append(aj)
        opp_act[(i, j)].append(aj); opp_act[(j, i)].append(ai)

# ------------------------------------------------------------------
# Compute HIT for each agent
# ------------------------------------------------------------------
stats = defaultdict(lambda: {"H":0.0, "I":0.0, "n":0})

for i in agents:
    for j in agents:
        if i == j:
            continue
        ctx = np.array(opp_act[(i, j)][:-1])  # C_t = j’s previous action
        rsp = np.array(my_act [(i, j)][1:])   # R_t = i’s current action
        if len(ctx) == 0:
            continue
        Hc = entropy_bits(ctx)
        Hr = entropy_bits(rsp)
        Ib = mi_bits(ctx, rsp)

        H_norm = Hc  # binary alphabet ⇒ H_max = 1
        denom  = min(Hc, Hr)
        I_norm = Ib / denom if denom > 0 else 0.0

        stats[i]["H"] += H_norm
        stats[i]["I"] += I_norm
        stats[i]["n"] += 1

# ------------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------------
rows = []
for agent, d in stats.items():
    Hbar = d["H"] / d["n"]
    Ibar = d["I"] / d["n"]
    HIT  = Hbar * Ibar
    rows.append([agent, round(Hbar, 3), round(Ibar, 3), round(HIT, 3)])

df = pd.DataFrame(rows, columns=["Strategy", "H_bar", "I_bar", "HIT"])
df = df.sort_values("Strategy").reset_index(drop=True)

print(df.to_string(index=False))
