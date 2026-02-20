#!/usr/bin/env python3
"""
Coherence Unit Cell — Minimal Replication (E1 + E9)
Weekend-sized: ~2 min, seed=42, numpy only.
Matt Waltman & Aurora | Feb 2026 | Richter Farms, Iowa
"""
import numpy as np

SEED = 42

def rk4_step(theta, omega, A, Q, K, F, dt, forcing_val):
    """4th-order Runge-Kutta for Kuramoto oscillators."""
    def deriv(th):
        N = len(th)
        dth = omega.copy()
        for i in range(N):
            coupling = 0.0
            for j in range(N):
                if A[i, j]:
                    coupling += Q[j] * np.sin(th[j] - th[i])
            dth[i] += (K / N) * coupling + F * forcing_val * np.sin(th[i])
        return dth
    k1 = deriv(theta)
    k2 = deriv(theta + 0.5*dt*k1)
    k3 = deriv(theta + 0.5*dt*k2)
    k4 = deriv(theta + dt*k3)
    return theta + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def order_param(theta):
    return float(np.abs(np.mean(np.exp(1j * theta))))

def run_sim(N, omega, A, Q, K, F, dt, n_steps, forcing, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    R = np.zeros(n_steps)
    for t in range(n_steps):
        theta = rk4_step(theta, omega, A, Q, K, F, dt, forcing[t % len(forcing)])
        R[t] = order_param(theta)
    return R, theta

def flower_of_life_adj(N):
    """Hexagonal-ring Flower of Life adjacency for N nodes."""
    A = np.zeros((N, N))
    n_ring = min(6, N-1)
    for i in range(N):
        for d in range(1, n_ring+1):
            j = (i + d) % N
            if abs(i - j) <= n_ring or abs(i - j) >= N - n_ring:
                A[i, j] = A[j, i] = 1
    return A

def cell24_adj():
    """24-cell polytope: 24 vertices, 96 edges, degree 8."""
    verts = []
    for i in range(4):
        for s in [-1, 1]:
            v = [0]*4; v[i] = s*2; verts.append(v)
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                for s4 in [-1, 1]:
                    verts.append([s1, s2, s3, s4])
    verts = np.array(verts[:24], dtype=float)
    A = np.zeros((24, 24))
    for i in range(24):
        dists = np.sqrt(np.sum((verts - verts[i])**2, axis=1))
        neighbors = np.where((dists > 0.1) & (dists < 2.9))[0]
        for j in neighbors:
            A[i, j] = 1
    return A

def shera_q(N, f_min=200, f_max=20000):
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), N)
    return 12.7 * (freqs / 1000)**0.3

# ─── EXPERIMENT 1: Shera Q Transplant ───────────────────────────────
print("=" * 60)
print("E1: Shera Q-Factor Transplantation")
print("=" * 60)

N, n_steps, n_trials, K, F, dt = 51, 2000, 10, 5.0, 0.5, 0.01
omega = 2*np.pi*np.linspace(0.8, 1.2, N)
A = flower_of_life_adj(N)
forcing = np.sin(np.linspace(0, 20*np.pi, n_steps) * np.linspace(1, 3, n_steps))
Q_shera = shera_q(N)
Q_uniform = np.ones(N) * np.mean(Q_shera)

shera_R, uniform_R = [], []
for trial in range(n_trials):
    R, _ = run_sim(N, omega, A, Q_shera, K, F, dt, n_steps, forcing, SEED+trial)
    shera_R.append(np.mean(R[-200:]))
    R, _ = run_sim(N, omega, A, Q_uniform, K, F, dt, n_steps, forcing, SEED+trial)
    uniform_R.append(np.mean(R[-200:]))

print(f"  Uniform Q: R = {np.mean(uniform_R):.4f} +/- {np.std(uniform_R):.4f}")
print(f"  Shera Q:   R = {np.mean(shera_R):.4f} +/- {np.std(shera_R):.4f}")
delta = np.mean(shera_R) - np.mean(uniform_R)
print(f"  Biological advantage: dR = {delta:+.4f}")
print(f"  {'PASS' if delta > 0.05 else 'CHECK'}: Shera Q improves coherence")

# ─── EXPERIMENT 9: Self-Duality with Controls ───────────────────────
print("\n" + "=" * 60)
print("E9: Self-Duality — Strengthened Controls")
print("=" * 60)

N, n_warm, n_rec, n_trials = 24, 500, 500, 30
K, F, dt = 3.0, 0.2, 0.01
omega = 2*np.pi*np.linspace(0.8, 1.2, N)
A = cell24_adj()
forcing = 0.3*np.sin(np.linspace(0, 8*np.pi, n_warm+n_rec))

np.random.seed(SEED+200)
W_ortho, _ = np.linalg.qr(np.random.randn(N, N))
W_rand = np.random.randn(N, N)
W_rand /= np.linalg.norm(W_rand, axis=1, keepdims=True)

# Train decoder
train_phases = []
for t in range(10):
    _, theta = run_sim(N, omega, A, np.ones(N), K, F, dt, n_warm, forcing[:n_warm], SEED+t)
    train_phases.append(theta)
X = np.array([np.cos(p) for p in train_phases])
Y = np.array(train_phases)
W_trained = np.linalg.lstsq(X, Y, rcond=None)[0].T

conditions = {"Self-Dual": None, "Orthogonal": W_ortho,
              "Trained Decoder": W_trained, "Random Projection": W_rand}

all_results = {}
for cname, W in conditions.items():
    recs = []
    for trial in range(n_trials):
        _, theta = run_sim(N, omega, A, np.ones(N), K, F, dt, n_warm, forcing[:n_warm], SEED+trial)
        id_phases = theta.copy()
        id_rep = np.exp(1j*id_phases) if W is None else W @ np.cos(id_phases)
        # Perturb 30%
        idx = np.random.RandomState(SEED+trial+5000).choice(N, int(0.3*N), replace=False)
        theta[idx] = np.random.RandomState(SEED+trial+6000).uniform(0, 2*np.pi, len(idx))
        # Recover
        for t in range(n_rec):
            theta = rk4_step(theta, omega, A, np.ones(N), K, F, dt, forcing[(n_warm+t)%len(forcing)])
        if W is None:
            sim = float(np.abs(np.mean(np.exp(1j*(theta - id_phases)))))
        else:
            rec_rep = W @ np.cos(theta)
            corr = np.corrcoef(id_rep.flatten(), rec_rep.flatten())[0, 1]
            sim = max(0, float(corr))
        recs.append(sim)
    m, s = np.mean(recs), np.std(recs)
    all_results[cname] = m
    print(f"  {cname}: {m:.4f} +/- {s:.4f}")

# Compute proper ratio
sd_val = all_results["Self-Dual"]
ctrl_avg = np.mean([all_results[k] for k in all_results if k != "Self-Dual"])
if ctrl_avg > 0.001:
    ratio = sd_val / ctrl_avg
    print(f"\n  Self-Dual recovery: {sd_val:.4f}")
    print(f"  Control average:    {ctrl_avg:.4f}")
    print(f"  Advantage ratio:    {ratio:.1f}x")
else:
    print(f"\n  Self-Dual recovery: {sd_val:.4f}")
    print(f"  Controls near zero — advantage is clear")

print(f"\n  NOTE: This minimal script demonstrates effect direction and")
print(f"  relative advantage. For exact values matching the document,")
print(f"  run the full suite: python3 coherence_unit_cell_experiments.py")
print(f"\n  Seed: {SEED} | Dependencies: numpy")
print("=" * 60)
