"""
hit_5agent_c2_pn.py
HIT Benchmark. 5-Agent C2 simulation using Petri nets.

Author : A. Artturi Juvonen
Date   : 2025-05-14
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.metrics import mutual_info_score

# Agent and environment setup
agents = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
ctx_vals = ["calm", "suspicious", "hostile"]
ctx_p = [0.5, 0.3, 0.2]
actions = ["monitor", "query", "engage"]
rng = np.random.default_rng(123)

# Agent policies
def delta_policy(c):
    return rng.choice(actions) if rng.random() < 0.2 else {
        "calm": "monitor", "suspicious": "query", "hostile": "engage"
    }[c]

def echo_policy(c): return delta_policy(c)

def bravo_policy(c):
    return "monitor" if c == "calm" else \
           rng.choice(["monitor", "query"], p=[0.3, 0.7]) if c == "suspicious" else "engage"

def charlie_policy(c):
    return rng.choice(["monitor", "query"], p=[0.8, 0.2]) if c == "calm" else \
           rng.choice(["monitor", "query"], p=[0.4, 0.6]) if c == "suspicious" else "engage"

def alpha_policy(c):
    return {"calm": "monitor", "suspicious": "query", "hostile": "engage"}[c]

policy = {
    "Alpha": alpha_policy,
    "Bravo": bravo_policy,
    "Charlie": charlie_policy,
    "Delta": delta_policy,
    "Echo": echo_policy
}

# Simulation
N = 3000
log = []
flow_info = {a: {"in": 0, "out": 0} for a in agents}
flow_cmd = {a: {"in": 0, "out": 0} for a in agents}

for _ in range(N):
    env = rng.choice(ctx_vals, p=ctx_p)
    d_ctx = rng.choice([env] + [x for x in ctx_vals if x != env], p=[0.9, 0.05, 0.05])
    e_ctx = rng.choice([env] + [x for x in ctx_vals if x != env], p=[0.9, 0.05, 0.05])
    d_act = delta_policy(d_ctx)
    e_act = echo_policy(e_ctx)

    log.append(("Delta", d_ctx, d_act, 1))
    log.append(("Echo", e_ctx, e_act, 1))
    flow_info["Delta"]["out"] += 1
    flow_info["Bravo"]["in"] += 1
    flow_info["Echo"]["out"] += 1
    flow_info["Charlie"]["in"] += 1

    b_own = rng.choice(ctx_vals, p=[0.8 if env == x else 0.1 for x in ctx_vals])
    c_own = rng.choice(ctx_vals, p=[0.8 if env == x else 0.1 for x in ctx_vals])
    b_fused = rng.choice([d_ctx, b_own], p=[0.7, 0.3])
    c_fused = rng.choice([e_ctx, c_own], p=[0.7, 0.3])

    # Peer fusion
    b_fused = rng.choice([b_fused, c_fused], p=[0.7, 0.3])
    c_fused = rng.choice([c_fused, b_fused], p=[0.7, 0.3])

    b_act = bravo_policy(b_fused)
    c_act = charlie_policy(c_fused)
    log.append(("Bravo", b_fused, b_act, 2))
    log.append(("Charlie", c_fused, c_act, 2))

    flow_info["Bravo"]["out"] += 1
    flow_info["Charlie"]["out"] += 1
    flow_info["Alpha"]["in"] += 2

    flow_cmd["Bravo"]["out"] += 1
    flow_cmd["Delta"]["in"] += 1
    flow_cmd["Charlie"]["out"] += 1
    flow_cmd["Echo"]["in"] += 1

    a_fused = rng.choice([b_fused, c_fused])
    a_act = alpha_policy(a_fused)
    log.append(("Alpha", a_fused, a_act, 3))

    for target in ["Bravo", "Charlie", "Delta", "Echo"]:
        flow_cmd["Alpha"]["out"] += 1
        flow_cmd[target]["in"] += 1

# Convert to DataFrame
df = pd.DataFrame(log, columns=["Agent", "Ctx", "Act", "Lat"])

# HIT computation
Hmax = math.log2(len(ctx_vals))
Imax = math.log2(len(actions))

def entropy(seq):
    n = len(seq)
    c = Counter(seq)
    return -sum((v / n) * math.log2(v / n) for v in c.values())

def mutual_info(cseq, rseq):
    return mutual_info_score(cseq, rseq) / math.log(2)

results = []
for a in agents:
    sub = df[df.Agent == a]
    h = entropy(sub.Ctx)
    i = mutual_info(sub.Ctx, sub.Act)
    hbar = h / Hmax
    ibar = i / Imax
    tbar = sub.Lat.mean() / 3
    HIT = (hbar * ibar) / tbar
    lam_info = math.log2(1 + flow_info[a]["out"]) - math.log2(1 + flow_info[a]["in"])
    lam_cmd = math.log2(1 + flow_cmd[a]["out"]) - math.log2(1 + flow_cmd[a]["in"])
    results.append({
        "Agent": a, "H": h, "I": i, "H̄": hbar, "Ī": ibar, "T̄": tbar,
        "HIT": HIT,
        "λ_info": lam_info, "λ_cmd": lam_cmd,
        "HIT_info": HIT * lam_info, "HIT_cmd": HIT * lam_cmd
    })

df_results = pd.DataFrame(results)
print(df_results.round(3))
