"""
hit_single_agent_ap.py
Single-agent air-policing toy model used to study HIT metrics.
Every contact starts "unknown," the agent may probe up to four
times, and actions accrue cost according to the ROE table.

Author : A. Artturi Juvonen
Date   : 2025-05-13
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.metrics import mutual_info_score

# -------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------
np.random.seed(0)

N_CONTACTS   = 5000          # trials
MAX_ROUNDS   = 4             # ≤ 4 probes
CONF_THRESH  = 0.70          # stop if max belief ≥ 0.7

CLASSES  = ["H", "A", "F"]   # Hostile, Ambiguous, Friendly
P_CLASS  = [1/3, 1/3, 1/3]
H_MAX    = math.log2(len(CLASSES))

FEATURES = {
    "geography"      : ["own", "international", "foreign"],
    "squawk"         : ["per_plan", "standby", "incorrect"],
    "iff"            : ["positive", "nonresponsive", "negative"],
    "radar_contact"  : ["none", "secondary", "primary_own", "primary_peer"],
    "cooperation"    : ["complies", "uncooperative", "evasive"],
    "combat_behavior": ["none", "defensive", "offensive", "flares",
                         "warning_fire", "missile_launch"],
    "flight_profile" : ["low_slow", "low_fast", "cruise", "supersonic",
                         "erratic"],
}

NOISE_MISREAD = 0.20   # 20 % mis-read
NOISE_DROPOUT = 0.05   # 5 % dropout (None)

BASE_COST   = {"ignore": 2, "interrogate": 1, "intervene": 2, "intercept": 3}
GEO_FACTOR  = {"own": 1, "international": 2, "foreign": 4}
RISK_FACTOR = {"H": 1, "A": 2, "F": 5}
T_MAX       = 90       # “reasonable worst-case”

# -------------------------------------------------------------
# 2. Rule-based vote table  (H, A, F)
# -------------------------------------------------------------
RULE = {
    "geography": {
        "own":            [0.0, 0.0,  1.0],
        "international":  [0.5, 0.5,  0.5],
        "foreign":        [1.0, 0.3, -0.5],
    },
    "squawk": {
        "none":     [0.0, 0.0, 0.0],
        "per_plan": [0.0, 0.0, 1.0],
        "standby":  [0.3, 0.7, 0.0],
        "incorrect":[1.0, 0.5,-0.5],
    },
    "iff": {
        "positive":      [0.0, 0.0, 1.0],
        "nonresponsive": [0.4, 0.6, 0.0],
        "negative":      [1.0, 0.3,-0.5],
    },
    "radar_contact": {
        "none":         [0.0, 0.2, 0.4],
        "secondary":    [0.3, 0.5, 0.3],
        "primary_own":  [0.6, 0.3, 0.1],
        "primary_peer": [0.9, 0.2, 0.0],
    },
    "cooperation": {
        "complies":     [0.0, 0.2, 0.8],
        "uncooperative":[0.4, 0.5, 0.1],
        "evasive":      [0.9, 0.6,-0.3],
    },
    "combat_behavior": {
        "none":         [0.0, 0.2, 0.5],
        "defensive":    [0.4, 0.4, 0.0],
        "offensive":    [1.0, 0.3,-0.5],
        "flares":       [0.4, 0.3, 0.1],
        "warning_fire": [0.9, 0.2,-0.4],
        "missile_launch":[1.1,0.1,-0.6],
    },
    "flight_profile": {
        "low_slow":  [0.1, 0.4, 0.4],
        "low_fast":  [0.5, 0.4, 0.0],
        "cruise":    [0.1, 0.1, 0.7],
        "supersonic": [0.8,0.2,-0.3],
        "erratic":   [0.6, 0.4,-0.2],
    },
}

# -------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------
def sample_features(true_id):
    geo = np.random.choice(FEATURES["geography"])
    f = {
        "geography": geo,
        "squawk": "none",
        "iff": "nonresponsive",
        "radar_contact": np.random.choice(FEATURES["radar_contact"]),
        "cooperation": "complies" if true_id == "F" else \
                       np.random.choice(["uncooperative", "evasive"]),
        "combat_behavior": "none",
        "flight_profile": np.random.choice(FEATURES["flight_profile"]),
    }
    return f

def inject_noise(f):
    for k, v in f.items():
        if k == "geography":      # cannot mis-read
            continue
        if k == "iff":            # may drop but not mis-read
            if np.random.rand() < NOISE_DROPOUT:
                f[k] = None
            continue
        if np.random.rand() < NOISE_DROPOUT:
            f[k] = None
        elif np.random.rand() < NOISE_MISREAD:
            f[k] = np.random.choice([x for x in FEATURES[k] if x != v])
    return f

def classify(f):
    vote = np.zeros(3)
    for k, v in f.items():
        if v is None or k not in RULE:
            continue
        vote += np.array(RULE[k][v])
    p = np.exp(vote); p /= p.sum()
    return CLASSES[int(np.argmax(p))], p

def choose_action(label, conf):
    if label == "F" and conf >= 0.90:
        return "ignore"
    if label == "H" and conf >= 0.70:
        return "intercept"
    if label == "A" and conf >= 0.70:
        return "intervene"
    return "interrogate"

def cost(act, geo, label):
    return BASE_COST[act] * GEO_FACTOR[geo] * RISK_FACTOR[label]

def entropy(seq):
    c = Counter(seq); n=len(seq)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

# -------------------------------------------------------------
# 4. Simulation
# -------------------------------------------------------------
log = []

for _ in range(N_CONTACTS):
    true_id = np.random.choice(CLASSES, p=P_CLASS)
    feats   = sample_features(true_id)
    total_T = 0.0
    for t in range(MAX_ROUNDS):
        feats = inject_noise(feats)
        label, prob = classify(feats)
        conf  = float(prob.max())
        act   = choose_action(label, conf)
        total_T += cost(act, feats["geography"], label)
        if act != "interrogate" and conf >= CONF_THRESH:
            break
    log.append({"true": true_id, "declared": label,
                "action": act, "T": total_T})

df = pd.DataFrame(log)

# -------------------------------------------------------------
# 5. HIT Metrics
# -------------------------------------------------------------
I_CR = mutual_info_score(df["true"], df["declared"]) / math.log(2)
I_RA = mutual_info_score(df["declared"], df["action"]) / math.log(2)
T_avg = df["T"].mean()
HIT = (H_MAX * I_CR * I_RA) / (T_avg / T_MAX)

print(f"Hmax={H_MAX:.3f}  I(C;R)={I_CR:.3f}  I(R;A)={I_RA:.3f}  "
      f"T̄={T_avg:.1f}  HIT={HIT:.3f}")
print("Accuracy =",
      f"{(df['true']==df['declared']).mean()*100:.1f}%")
