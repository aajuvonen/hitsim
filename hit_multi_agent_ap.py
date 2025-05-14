"""
hit_multi_agent_ap.py
Four full-capacity agents observe a single contact, share belief vectors
over a full-mesh network with one-step latency, and act under a costed ROE.
The script logs per-agent HIT components.

Author : A. Artturi Juvonen
Date   : 2025-05-13
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.metrics import mutual_info_score

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
np.random.seed(0)
N_CONTACTS   = 5000          # independent trials
N_AGENTS     = 4             # full mesh
MAX_ROUNDS   = 4             # ≤ 4 probes per contact
CONF_THRESH  = 0.70          # stop when max belief ≥ 0.7

# Class prior (H, A, F) = (1/3, 1/3, 1/3)
CLASSES  = ["H", "A", "F"]
P_CLASS  = [1/3, 1/3, 1/3]
H_MAX    = math.log2(len(CLASSES))   # = log2 3

# Feature definitions --------------------------------------------------
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

NOISE_MISREAD = 0.20   # 20 % mis-read on noise-sensitive features
NOISE_DROPOUT = 0.05   # 5 % dropout (None)

# Cost model -----------------------------------------------------------
BASE_COST   = {"ignore": 2, "interrogate": 1, "intervene": 2, "intercept": 3}
GEO_FACTOR  = {"own": 1, "international": 2, "foreign": 4}
RISK_FACTOR = {"H": 1, "A": 2, "F": 5}
T_MAX       = 90       # “reasonable worst-case” path ≈ 3 rounds

# Rule-based vote table  (H, A, F) weights -----------------------------
RULE = {
    "geography": {
        "own":            [0.0, 0.0,  1.0],
        "international":  [0.5, 0.5,  0.5],
        "foreign":        [1.0, 0.3, -0.5],
    },
    "squawk": {
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
        "supersonic":[0.8, 0.2,-0.3],
        "erratic":   [0.6, 0.4,-0.2],
    },
}

# ----------------------------------------------------------------------
# 2. Helper functions
# ----------------------------------------------------------------------
def sample_features(true_id):
    """Generate initial feature dictionary for the contact."""
    geo = np.random.choice(FEATURES["geography"])
    feats = {
        "geography": geo,
        "squawk": "none",
        "iff": "nonresponsive",
        "radar_contact": np.random.choice(FEATURES["radar_contact"]),
        "cooperation": "complies" if true_id == "F" else \
                       np.random.choice(["uncooperative", "evasive"]),
        "combat_behavior": "none",
        "flight_profile": np.random.choice(FEATURES["flight_profile"]),
    }
    return feats

def inject_noise(feats):
    """Mis-read or drop features except geography and iff mis-read."""
    for k, v in feats.items():
        if k == "geography":            # cannot be corrupted
            continue
        if k == "iff":                  # iff can drop but not mis-read
            if np.random.rand() < NOISE_DROPOUT:
                feats[k] = None
            continue
        # dropout
        if np.random.rand() < NOISE_DROPOUT:
            feats[k] = None
        # mis-read
        elif np.random.rand() < NOISE_MISREAD:
            feats[k] = np.random.choice([x for x in FEATURES[k] if x != v])
    return feats

def classify(feats):
    """Rule-based softmax classification → belief vector and label."""
    vote = np.zeros(3)
    for key, val in feats.items():
        if val is None or key not in RULE:
            continue
        vote += np.array(RULE[key].get(val, [0,0,0]))
    probs = np.exp(vote) / np.sum(np.exp(vote))
    label = CLASSES[int(np.argmax(probs))]
    return label, probs

def choose_action(label, conf):
    """ROE decision policy based on label and confidence."""
    if label == "F" and conf >= 0.90:
        return "ignore"
    if label == "H" and conf >= 0.70:
        return "intercept"
    if label == "A" and conf >= 0.70:
        return "intervene"
    return "interrogate"

def action_cost(action, geo, label):
    return BASE_COST[action] * GEO_FACTOR[geo] * RISK_FACTOR[label]

def entropy(series):
    """Shannon entropy (bits) for discrete labels."""
    counts = Counter(series)
    n = len(series)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

# ----------------------------------------------------------------------
# 3. Simulation loop  ——  UNIDIRECTIONAL RING
# ----------------------------------------------------------------------
# records = []

# ring_next = {i: (i + 1) % N_AGENTS for i in range(N_AGENTS)}  # i → i+1 (mod n)

# for _ in range(N_CONTACTS):
#     true_id      = np.random.choice(CLASSES, p=P_CLASS)
#     local_feats  = [sample_features(true_id) for _ in range(N_AGENTS)]
#     last_msg     = [None] * N_AGENTS
#     declared     = [None] * N_AGENTS
#     action       = [None] * N_AGENTS
#     total_T      = [0.0] * N_AGENTS

#     for t in range(MAX_ROUNDS):

#         for i in range(N_AGENTS):
#             # 1) sensor corruption
#             local_feats[i] = inject_noise(local_feats[i])

#             # 2) local classification
#             lab_i, prob_i = classify(local_feats[i])

#             # 3) receive single message from predecessor (one-step latency)
#             if t > 0:
#                 pred = (i - 1) % N_AGENTS
#                 if last_msg[pred] is not None:
#                     prob_i *= last_msg[pred]           # Bayesian fusion
#                     prob_i  = prob_i / np.sum(prob_i)
#                     lab_i   = CLASSES[int(np.argmax(prob_i))]

#             # 4) decide action
#             conf_i = float(np.max(prob_i))
#             act_i  = choose_action(lab_i, conf_i)

#             # 5) cost accumulation
#             total_T[i] += action_cost(act_i, local_feats[i]["geography"], lab_i)

#             # 6) log & prepare outbound message to successor
#             declared[i] = lab_i
#             action[i]   = act_i
#             last_msg[i] = prob_i                       # travels to successor next step

#         # stop if everybody is satisfied
#         if all(action[i] != "interrogate" for i in range(N_AGENTS)):
#             break

#     # store trial outcome
#     for i in range(N_AGENTS):
#         records.append(
#             dict(agent=i, true_id=true_id, declared=declared[i],
#                  action=action[i], cost=total_T[i])
#         )

# ----------------------------------------------------------------------
# 3. Simulation loop: MESH TOPOLOGY
# ----------------------------------------------------------------------
records = []

for _ in range(N_CONTACTS):
    true_id = np.random.choice(CLASSES, p=P_CLASS)
    # Each agent keeps its own copy of features (independent noise)
    local_feats = [sample_features(true_id) for _ in range(N_AGENTS)]
    # Message buffer: list of belief vectors sent in previous step
    last_msg = [None] * N_AGENTS
    # Per-agent accumulators
    belief_hist = [[] for _ in range(N_AGENTS)]
    declared = [None] * N_AGENTS
    action   = [None] * N_AGENTS
    total_T  = [0.0] * N_AGENTS

    for t in range(MAX_ROUNDS):
        for i in range(N_AGENTS):
            # 1) sensor corruption each step
            local_feats[i] = inject_noise(local_feats[i])
            # 2) local classification
            lab_i, prob_i = classify(local_feats[i])
            # 3) receive all peer messages (one-step latency)
            if t > 0:
                for j in range(N_AGENTS):
                    if i == j or last_msg[j] is None:
                        continue
                    prob_i *= last_msg[j]  # Bayesian fusion (independent)
                prob_i = prob_i / np.sum(prob_i)
                lab_i  = CLASSES[int(np.argmax(prob_i))]
            # 4) decide action
            conf_i = float(np.max(prob_i))
            act_i  = choose_action(lab_i, conf_i)
            # 5) cost
            total_T[i] += action_cost(act_i, local_feats[i]["geography"], lab_i)
            # 6) log & prepare message for next round
            belief_hist[i].append(lab_i)
            declared[i] = lab_i
            action[i]   = act_i
            last_msg[i] = prob_i
        # stopping condition: if all agents took non-interrogate with confidence
        if all(action[i] != "interrogate" for i in range(N_AGENTS)):
            break

    # store trial results
    for i in range(N_AGENTS):
        records.append({
            "agent": i,
            "true_id": true_id,
            "declared": declared[i],
            "action": action[i],
            "cost": total_T[i],
        })

df = pd.DataFrame(records)

# ----------------------------------------------------------------------
# 4. Per-agent HIT computation
# ----------------------------------------------------------------------
summary = []
for i in range(N_AGENTS):
    sub = df[df["agent"] == i]
    I_CR = mutual_info_score(sub["true_id"], sub["declared"]) / math.log(2)
    I_RA = mutual_info_score(sub["declared"], sub["action"]) / math.log(2)
    T_avg = sub["cost"].mean()
    HIT = (H_MAX * I_CR * I_RA) / (T_avg / T_MAX)
    summary.append({
        "agent": i,
        "H_bits": round(H_MAX,3),
        "I(C;R)": round(I_CR,3),
        "I(R;A)": round(I_RA,3),
        "T_avg": round(T_avg,1),
        "HIT": round(HIT,3),
        "accuracy_%": round((sub["true_id"] == sub["declared"]).mean()*100,1),
    })

print(pd.DataFrame(summary).to_string(index=False))
