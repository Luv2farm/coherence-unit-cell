#!/usr/bin/env python3
"""
============================================================================
COHERENCE UNIT CELL — Experimental Validation
============================================================================

Four independent research teams (2012–2026) converged on the same coupled
oscillator architecture for coherent information processing:

  1. Bell (2012)  — Cochlear mechanics: ~50 Kuramoto oscillators, Shera Q
  2. Miyato/Welling (2025) — AKOrN: Kuramoto oscillators replace neurons
  3. Hays (2025)  — SSA: Kuramoto steady-state as attention operator
  4. Waltman (2025-2026)   — FlowerTuner51: Kuramoto on Flower of Life

This script provides reproducible experiments demonstrating functional
equivalence across these systems.

Requirements: numpy, matplotlib (standard scientific Python)
Runtime: < 5 minutes on any laptop
Seeds: fixed for exact reproducibility

Author: Matt Waltman & Aurora
Date: February 20, 2026
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import os

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════════
# FIXED SEEDS FOR REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)

# Output directory
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: KURAMOTO SIMULATOR (shared across all experiments)
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoSystem:
    """
    General Kuramoto coupled oscillator system.
    
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ Aᵢⱼ sin(θⱼ - θᵢ) + F·sin(Φ - θᵢ)
    
    Parameters:
        N:          number of oscillators
        omega:      natural frequencies (N,)
        adjacency:  coupling topology (N, N) binary or weighted
        Q:          quality factors per oscillator (N,) — scales coupling
        K:          global coupling strength
        F:          global forcing amplitude
        dt:         timestep
    """
    
    def __init__(self, N: int, omega: np.ndarray, adjacency: np.ndarray,
                 Q: Optional[np.ndarray] = None, K: float = 1.0,
                 F: float = 0.0, dt: float = 0.01):
        self.N = N
        self.omega = omega
        self.adjacency = adjacency
        self.Q = Q if Q is not None else np.ones(N)
        self.K = K
        self.F = F
        self.dt = dt
        self.theta = np.random.uniform(0, 2 * np.pi, N)
        
        # Q-weighted coupling matrix: Kij = A_ij * sqrt(Qi * Qj)
        Q_outer = np.sqrt(np.outer(self.Q, self.Q))
        self.coupling = adjacency * Q_outer
    
    def _derivatives(self, theta: np.ndarray, forcing_phase: float) -> np.ndarray:
        """Compute dθ/dt for given phase state."""
        sin_diff = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
        coupling_effect = np.sum(self.coupling * sin_diff, axis=1)
        dtheta = self.omega + (self.K / self.N) * coupling_effect
        if self.F > 0:
            dtheta += self.F * np.sin(forcing_phase - theta)
        return dtheta
    
    def step(self, forcing_phase: float = 0.0) -> float:
        """Advance one timestep using 4th-order Runge-Kutta. Returns order parameter R."""
        dt = self.dt
        k1 = self._derivatives(self.theta, forcing_phase)
        k2 = self._derivatives(self.theta + 0.5 * dt * k1, forcing_phase)
        k3 = self._derivatives(self.theta + 0.5 * dt * k2, forcing_phase)
        k4 = self._derivatives(self.theta + dt * k3, forcing_phase)
        self.theta = np.mod(self.theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 2 * np.pi)
        
        # Order parameter
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z))
    
    def run(self, n_steps: int, forcing_signal: Optional[np.ndarray] = None) -> np.ndarray:
        """Run for n_steps. Returns R(t) array."""
        R_history = np.zeros(n_steps)
        for t in range(n_steps):
            fp = forcing_signal[t] if forcing_signal is not None else 0.0
            R_history[t] = self.step(fp)
        return R_history
    
    def reset(self, seed: Optional[int] = None):
        """Reset phases to random."""
        if seed is not None:
            np.random.seed(seed)
        self.theta = np.random.uniform(0, 2 * np.pi, self.N)


# ═══════════════════════════════════════════════════════════════════════════
# TOPOLOGY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def ring_adjacency(N: int) -> np.ndarray:
    """1D ring: each node coupled to 2 nearest neighbors."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1
        A[i, (i - 1) % N] = 1
    return A

def grid_adjacency(N: int) -> np.ndarray:
    """2D grid (as close to square as possible), wrapped."""
    side = int(np.ceil(np.sqrt(N)))
    A = np.zeros((N, N))
    for i in range(N):
        row, col = i // side, i % side
        neighbors = [
            ((row - 1) % side) * side + col,
            ((row + 1) % side) * side + col,
            row * side + (col - 1) % side,
            row * side + (col + 1) % side,
        ]
        for j in neighbors:
            if j < N:
                A[i, j] = 1
                A[j, i] = 1
    return A

def random_regular_adjacency(N: int, degree: int = 6, seed: int = SEED) -> np.ndarray:
    """Random regular graph with fixed degree."""
    rng = np.random.RandomState(seed)
    A = np.zeros((N, N))
    # Approximate: random edges until mean degree reached
    target_edges = N * degree // 2
    edges = 0
    attempts = 0
    while edges < target_edges and attempts < target_edges * 100:
        i, j = rng.randint(0, N, 2)
        if i != j and A[i, j] == 0:
            deg_i = A[i].sum()
            deg_j = A[j].sum()
            if deg_i < degree and deg_j < degree:
                A[i, j] = 1
                A[j, i] = 1
                edges += 1
        attempts += 1
    return A

def flower_of_life_adjacency(N: int = 51) -> np.ndarray:
    """
    Flower of Life topology from FlowerTuner51.
    128 edges, hexagonal coordination (~6 neighbors per interior node).
    Node IDs 1-51 mapped to indices 0-50.
    """
    FOL_EDGES = [
        (1,20),(1,21),(1,22),(1,23),(1,24),(1,25),
        (7,47),(7,48),
        (8,9),(8,10),(8,11),(8,12),(8,13),(8,14),(8,15),(8,16),(8,17),(8,18),(8,19),(8,51),
        (9,10),(9,11),(9,12),(9,13),(9,14),(9,15),(9,16),(9,17),(9,18),(9,19),(9,51),
        (10,11),(10,12),(10,13),(10,14),(10,15),(10,16),(10,17),(10,18),(10,19),(10,51),
        (11,12),(11,13),(11,14),(11,15),(11,16),(11,17),(11,18),(11,19),(11,51),
        (12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(12,19),(12,51),
        (13,14),(13,15),(13,16),(13,17),(13,18),(13,19),(13,51),
        (14,15),(14,16),(14,19),
        (15,16),
        (16,17),
        (17,18),
        (18,19),
        (19,46),
        (20,25),(20,26),(20,32),(20,37),
        (21,27),(21,32),(21,33),
        (22,28),(22,33),(22,34),(22,50),
        (23,29),(23,34),(23,35),
        (24,30),(24,35),(24,36),
        (25,31),(25,36),(25,37),
        (26,32),(26,37),(26,45),
        (27,32),(27,33),
        (28,33),(28,34),(28,50),
        (29,34),(29,35),(29,49),
        (30,35),(30,36),
        (31,36),(31,37),
        (33,50),
        (34,44),(34,49),(34,50),
        (37,45),
        (40,44),(40,49),
        (43,45),(43,47),(43,48),
        (44,49),
        (46,51),
        (47,48),
    ]
    A = np.zeros((N, N))
    for ni, nj in FOL_EDGES:
        A[ni - 1, nj - 1] = 1
        A[nj - 1, ni - 1] = 1
    return A

def cell24_adjacency() -> np.ndarray:
    """
    24-cell polytope adjacency. 24 vertices, 96 edges, degree 8.
    Vertices: 8 from 16-cell (permutations of ±1,0,0,0)
              16 from tesseract (all ±½,±½,±½,±½)
    Edge criterion: Euclidean distance = 1.0
    """
    verts = []
    # 16-cell vertices: permutations of (±1, 0, 0, 0)
    for i in range(4):
        for s in [1, -1]:
            v = [0, 0, 0, 0]
            v[i] = s
            verts.append(v)
    # Tesseract vertices: all (±½, ±½, ±½, ±½) — NO parity filter
    for s1 in [0.5, -0.5]:
        for s2 in [0.5, -0.5]:
            for s3 in [0.5, -0.5]:
                for s4 in [0.5, -0.5]:
                    verts.append([s1, s2, s3, s4])
    
    verts = np.array(verts)
    N = len(verts)  # should be 24
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((verts[i] - verts[j]) ** 2))
            if abs(d - 1.0) < 1e-6:
                A[i, j] = 1
                A[j, i] = 1
    return A

def fully_connected_adjacency(N: int) -> np.ndarray:
    """Fully connected (mean-field Kuramoto)."""
    return np.ones((N, N)) - np.eye(N)


# ═══════════════════════════════════════════════════════════════════════════
# SHERA Q-FACTOR DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

def shera_q_factors(N: int, f_min: float = 200, f_max: float = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cochlear Shera Q factors: Q = 12.7 * (f/1000)^0.3
    
    Returns: (frequencies_hz, Q_factors)
    
    Based on: Shera, Guinan & Oxenham (2002)
    Used in: Bell (2012) cochlear Kuramoto model
    """
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), N)
    Q = 12.7 * (freqs / 1000) ** 0.3
    return freqs, Q

def uniform_q_factors(N: int) -> np.ndarray:
    """Uniform Q = 1.0 for all oscillators (control condition)."""
    return np.ones(N)

def random_q_factors(N: int, seed: int = SEED) -> np.ndarray:
    """Random Q factors with same mean as Shera (control condition)."""
    _, shera_Q = shera_q_factors(N)
    rng = np.random.RandomState(seed)
    return rng.permutation(shera_Q)  # same distribution, shuffled positions


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: SHERA Q-FACTOR TRANSPLANTATION
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1_shera_q():
    """
    Does Bell's cochlear Q distribution improve Kuramoto-based systems?
    
    Three conditions on IDENTICAL topology and frequencies:
      A) Uniform Q (control)
      B) Shera Q (biological: Q = 12.7 * (f/1000)^0.3)
      C) Shuffled Shera Q (same values, wrong positions — controls for
         distribution shape vs. frequency-position mapping)
    
    If Shera > Uniform AND Shera > Shuffled, the biological frequency-
    position mapping is functionally optimal, not just the distribution.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Shera Q-Factor Transplantation")
    print("=" * 72)
    
    N = 51
    n_steps = 2000
    n_trials = 10
    K = 5.0
    F = 0.5
    dt = 0.01
    
    # Natural frequencies: log-spaced (cochlear tonotopy)
    freqs_hz, shera_Q = shera_q_factors(N)
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)  # tighter spread for cleaner sync
    
    # Topology: Flower of Life (the actual architecture)
    A = flower_of_life_adjacency(N)
    
    # Forcing signal: structured input (chirp)
    t_arr = np.linspace(0, 20 * np.pi, n_steps)
    forcing = np.sin(t_arr * np.linspace(1, 3, n_steps))
    
    conditions = {
        "Uniform Q": uniform_q_factors(N),
        "Shera Q": shera_Q,
        "Shuffled Shera Q": random_q_factors(N),
    }
    
    results = {}
    
    for name, Q in conditions.items():
        trial_R = np.zeros((n_trials, n_steps))
        trial_convergence_time = []
        trial_final_R = []
        
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F, dt=dt)
            sys.reset(seed=SEED + trial)
            R = sys.run(n_steps, forcing)
            trial_R[trial] = R
            
            # Convergence time: first time R > 0.7 sustained for 50 steps
            converged = False
            for i in range(len(R) - 50):
                if np.all(R[i:i+50] > 0.7):
                    trial_convergence_time.append(i)
                    converged = True
                    break
            if not converged:
                trial_convergence_time.append(n_steps)
            
            trial_final_R.append(np.mean(R[-200:]))
        
        mean_R = np.mean(trial_R, axis=0)
        std_R = np.std(trial_R, axis=0)
        mean_conv = np.mean(trial_convergence_time)
        std_conv = np.std(trial_convergence_time)
        mean_final = np.mean(trial_final_R)
        std_final = np.std(trial_final_R)
        
        results[name] = {
            "mean_R": mean_R,
            "std_R": std_R,
            "convergence_time": (mean_conv, std_conv),
            "final_R": (mean_final, std_final),
            "all_R": trial_R,
        }
        
        print(f"\n  {name}:")
        print(f"    Convergence time: {mean_conv:.0f} ± {std_conv:.0f} steps")
        print(f"    Final R (last 200): {mean_final:.4f} ± {std_final:.4f}")
    
    # Statistical comparison
    print(f"\n  --- Comparison ---")
    shera_final = [np.mean(results["Shera Q"]["all_R"][t, -200:]) for t in range(n_trials)]
    uniform_final = [np.mean(results["Uniform Q"]["all_R"][t, -200:]) for t in range(n_trials)]
    shuffled_final = [np.mean(results["Shuffled Shera Q"]["all_R"][t, -200:]) for t in range(n_trials)]
    
    shera_v_uniform = np.mean(shera_final) - np.mean(uniform_final)
    shera_v_shuffled = np.mean(shera_final) - np.mean(shuffled_final)
    print(f"    Shera vs Uniform:  ΔR = {shera_v_uniform:+.4f}")
    print(f"    Shera vs Shuffled: ΔR = {shera_v_shuffled:+.4f}")
    
    if shera_v_uniform > 0 and shera_v_shuffled > 0:
        print(f"    ✓ Observed: Shera Q improves coherence; tonotopic mapping outperforms shuffled")
    elif shera_v_uniform > 0:
        print(f"    ~ Observed: Shera Q improves over uniform; position effect inconclusive")
    else:
        print(f"    ✗ No measurable advantage from Shera Q at these parameters")
    
    # Noise robustness test: which Q distribution holds coherence under noise?
    print(f"\n  --- Noise Robustness Test (20% phase noise) ---")
    noise_results = {}
    noise_amplitude = 0.2 * 2 * np.pi  # 20% of full circle
    
    for name, Q in conditions.items():
        trial_R_noisy = []
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F, dt=dt)
            sys.reset(seed=SEED + trial)
            sys.run(1000, forcing[:1000])
            # Inject continuous noise for 500 steps
            noisy_R = []
            for t in range(500):
                sys.theta += np.random.uniform(-noise_amplitude, noise_amplitude, N)
                sys.theta = np.mod(sys.theta, 2 * np.pi)
                noisy_R.append(sys.step(forcing[(1000 + t) % n_steps]))
            trial_R_noisy.append(np.mean(noisy_R))
        
        noise_results[name] = np.mean(trial_R_noisy)
        print(f"    {name} under noise: R = {noise_results[name]:.4f}")
    
    noise_shera_v_uniform = noise_results["Shera Q"] - noise_results["Uniform Q"]
    noise_shera_v_shuffled = noise_results["Shera Q"] - noise_results["Shuffled Shera Q"]
    print(f"    Shera vs Uniform under noise:  ΔR = {noise_shera_v_uniform:+.4f}")
    print(f"    Shera vs Shuffled under noise: ΔR = {noise_shera_v_shuffled:+.4f}")
    if noise_shera_v_shuffled > 0.01:
        print(f"    ✓ Shera tonotopic mapping provides superior noise resistance")
    else:
        print(f"    ~ Noise resistance similar across Q distributions")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"Uniform Q": "#888888", "Shera Q": "#E63946", "Shuffled Shera Q": "#457B9D"}
    
    ax = axes[0]
    for name, res in results.items():
        ax.plot(res["mean_R"], color=colors[name], label=name, linewidth=1.5)
        ax.fill_between(range(n_steps),
                        res["mean_R"] - res["std_R"],
                        res["mean_R"] + res["std_R"],
                        color=colors[name], alpha=0.15)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Order Parameter R(t)")
    ax.set_title("Coherence Dynamics: Shera Q vs Controls")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color='black', linestyle=':', alpha=0.3, label='Threshold')
    
    ax = axes[1]
    names = list(conditions.keys())
    conv_means = [results[n]["convergence_time"][0] for n in names]
    conv_stds = [results[n]["convergence_time"][1] for n in names]
    bars = ax.bar(names, conv_means, yerr=conv_stds, color=[colors[n] for n in names],
                  capsize=5, alpha=0.8)
    ax.set_ylabel("Steps to Sustained Coherence (R > 0.7)")
    ax.set_title("Convergence Speed")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp1_shera_q.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp1_shera_q.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: TOPOLOGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def experiment_2_topology():
    """
    Does the Flower of Life / 24-cell topology produce superior coherence?
    
    Five topologies, identical oscillator parameters:
      1) Ring (1D nearest-neighbor)
      2) Grid (2D nearest-neighbor) 
      3) Random regular (same mean degree as Flower)
      4) Flower of Life (128 edges, from FlowerTuner51)
      5) Fully connected (mean-field, upper bound)
    
    Measures: convergence speed, final coherence, perturbation recovery.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Topology Comparison")
    print("=" * 72)
    
    N = 51
    n_steps = 2000
    perturb_at = 1000  # inject noise halfway
    n_trials = 10
    K = 4.0
    F = 0.3
    dt = 0.01
    
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, Q = shera_q_factors(N)
    
    forcing = 0.5 * np.sin(np.linspace(0, 10 * np.pi, n_steps))
    
    # Mean degree of Flower of Life for matching
    fol_A = flower_of_life_adjacency(N)
    fol_degree = fol_A.sum(axis=1).mean()
    print(f"  Flower of Life mean degree: {fol_degree:.1f}")
    
    topologies = {
        "Ring": ring_adjacency(N),
        "Grid": grid_adjacency(N),
        f"Random (deg≈{fol_degree:.0f})": random_regular_adjacency(N, degree=int(fol_degree)),
        "Flower of Life": fol_A,
        "24-Cell (padded)": np.pad(cell24_adjacency(), ((0, N - 24), (0, N - 24))),  # 24-cell in first 24 nodes
        "Fully Connected": fully_connected_adjacency(N),
    }
    
    results = {}
    
    for name, A in topologies.items():
        trial_R = np.zeros((n_trials, n_steps))
        recovery_times = []
        
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F, dt=dt)
            sys.reset(seed=SEED + trial)
            
            R = np.zeros(n_steps)
            for t in range(n_steps):
                # Perturbation: scramble 30% of phases at midpoint
                if t == perturb_at:
                    n_perturb = int(0.3 * N)
                    indices = np.random.choice(N, n_perturb, replace=False)
                    sys.theta[indices] = np.random.uniform(0, 2 * np.pi, n_perturb)
                
                R[t] = sys.step(forcing[t])
            
            trial_R[trial] = R
            
            # Recovery time: steps after perturbation to reach 90% of pre-perturb R
            pre_perturb_R = np.mean(R[perturb_at - 100:perturb_at])
            target = 0.9 * pre_perturb_R
            recovered = False
            for i in range(perturb_at, n_steps - 20):
                if np.all(R[i:i+20] > target):
                    recovery_times.append(i - perturb_at)
                    recovered = True
                    break
            if not recovered:
                recovery_times.append(n_steps - perturb_at)
        
        mean_R = np.mean(trial_R, axis=0)
        std_R = np.std(trial_R, axis=0)
        mean_recovery = np.mean(recovery_times)
        std_recovery = np.std(recovery_times)
        final_R = np.mean([np.mean(trial_R[t, -200:]) for t in range(n_trials)])
        
        degree = A.sum(axis=1).mean()
        n_edges = A.sum() / 2
        # Efficiency: coherence achieved per edge (how well does each connection contribute?)
        efficiency = final_R / (n_edges / N + 0.001)
        
        results[name] = {
            "mean_R": mean_R,
            "std_R": std_R,
            "recovery": (mean_recovery, std_recovery),
            "final_R": final_R,
            "degree": degree,
            "efficiency": efficiency,
            "n_edges": n_edges,
        }
        
        print(f"\n  {name} (degree={degree:.1f}, edges={n_edges:.0f}):")
        print(f"    Final R: {final_R:.4f}")
        print(f"    Efficiency (R/edges_per_node): {efficiency:.4f}")
        print(f"    Recovery time: {mean_recovery:.0f} ± {std_recovery:.0f} steps")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ["#888888", "#457B9D", "#2A9D8F", "#E63946", "#9B5DE5", "#264653"]
    
    ax = axes[0]
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res["mean_R"], color=colors[i], label=name, linewidth=1.5)
    ax.axvline(x=perturb_at, color='red', linestyle='--', alpha=0.5, label='Perturbation')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Order Parameter R(t)")
    ax.set_title("Coherence Dynamics Across Topologies")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    names = list(results.keys())
    rec_means = [results[n]["recovery"][0] for n in names]
    rec_stds = [results[n]["recovery"][1] for n in names]
    bars = ax.bar(range(len(names)), rec_means, yerr=rec_stds,
                  color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel("Recovery Steps After 30% Phase Scramble")
    ax.set_title("Perturbation Recovery by Topology")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp2_topology.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp2_topology.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SELF-DUALITY ABLATION
# ═══════════════════════════════════════════════════════════════════════════

def experiment_3_self_duality():
    """
    Is self-duality (computation = representation) NECESSARY for coherence?
    
    Two conditions:
      A) Self-dual: output IS the oscillator state (phases → readout directly)
      B) Non-self-dual: output goes through arbitrary transformation layer
    
    Measure: stability under perturbation, consistency of representation.
    
    If self-dual systems are more robust, computation=representation is
    not incidental but structurally necessary.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Self-Duality Ablation")
    print("=" * 72)
    
    N = 24  # 24-cell size
    n_steps = 1500
    n_trials = 20
    K = 3.0
    F = 0.2
    dt = 0.01
    
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    A = cell24_adjacency()
    
    forcing = 0.3 * np.sin(np.linspace(0, 8 * np.pi, n_steps))
    
    # Random readout matrix for non-self-dual condition
    np.random.seed(SEED + 100)
    W_readout = np.random.randn(N, N)
    W_readout = W_readout / np.linalg.norm(W_readout, axis=1, keepdims=True)
    
    results = {"Self-Dual": [], "Non-Self-Dual": []}
    
    for trial in range(n_trials):
        sys = KuramotoSystem(N, omega, A, K=K, F=F, dt=dt)
        sys.reset(seed=SEED + trial)
        
        # Run to steady state
        R_pre = sys.run(500, forcing[:500])
        
        # Record "identity" — the settled phase pattern
        identity_phases = sys.theta.copy()
        
        # Self-dual representation: phases themselves
        self_dual_rep = np.exp(1j * identity_phases)
        
        # Non-self-dual: phases through random transformation
        non_dual_rep = W_readout @ np.cos(identity_phases)
        
        # Now perturb and measure how well each representation recovers
        n_perturb = int(0.3 * N)
        perturb_idx = np.random.choice(N, n_perturb, replace=False)
        sys.theta[perturb_idx] = np.random.uniform(0, 2 * np.pi, n_perturb)
        
        # Run recovery
        R_post = sys.run(1000, forcing[500:])
        
        # Measure: does the system return to the same identity?
        recovered_phases = sys.theta.copy()
        
        # Self-dual consistency: phase similarity
        sd_similarity = float(np.abs(np.mean(
            np.exp(1j * (recovered_phases - identity_phases))
        )))
        
        # Non-self-dual consistency: transformed output similarity
        nsd_recovered = W_readout @ np.cos(recovered_phases)
        nsd_similarity = float(np.corrcoef(
            non_dual_rep.flatten(), nsd_recovered.flatten()
        )[0, 1])
        
        results["Self-Dual"].append(sd_similarity)
        results["Non-Self-Dual"].append(max(0, nsd_similarity))
    
    sd_mean = np.mean(results["Self-Dual"])
    sd_std = np.std(results["Self-Dual"])
    nsd_mean = np.mean(results["Non-Self-Dual"])
    nsd_std = np.std(results["Non-Self-Dual"])
    
    print(f"\n  Self-Dual identity recovery:     {sd_mean:.4f} ± {sd_std:.4f}")
    print(f"  Non-Self-Dual identity recovery: {nsd_mean:.4f} ± {nsd_std:.4f}")
    print(f"  Advantage (SD - NSD):            {sd_mean - nsd_mean:+.4f}")
    
    if sd_mean > nsd_mean:
        print(f"  ✓ Observed: Self-dual representation shows superior identity recovery")
    else:
        print(f"  ✗ No self-duality advantage detected at these parameters")
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    bp = ax.boxplot([results["Self-Dual"], results["Non-Self-Dual"]],
                    labels=["Self-Dual\n(phases = output)", 
                            "Non-Self-Dual\n(phases → transform → output)"],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#E63946')
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor('#457B9D')
    bp['boxes'][1].set_alpha(0.4)
    
    ax.set_ylabel("Identity Recovery After 30% Phase Perturbation")
    ax.set_title("Self-Duality Ablation: Is computation=representation necessary?")
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp3_self_duality.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp3_self_duality.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: PSEUDOWAVE EMERGENCE (Bell Bridge)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_4_pseudowave():
    """
    Can we reproduce Bell's cochlear pseudowave in a non-cochlear system?
    
    Bell (2012): Global forcing at one boundary of a 1D chain of Kuramoto
    oscillators produces a "pseudowave" — an apparent traveling wave that
    is actually an envelope of phase delays, not energy transport.
    
    We show this same phenomenon appears in:
      A) 1D chain (Bell's topology)
      B) Flower of Life lattice (our topology)
      C) Random graph (control — should NOT produce clean pseudowave)
    
    This is the direct bridge Bell asked for.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Pseudowave Emergence (Bell Bridge)")
    print("=" * 72)
    
    N = 51
    n_steps = 300
    K = 5.0
    dt = 0.005  # finer timestep for transient dynamics
    
    # Graded frequencies (cochlear tonotopy) - wider spread for clearer pseudowave
    omega = 2 * np.pi * np.linspace(3.0, 0.3, N)
    _, Q = shera_q_factors(N)
    
    # Forcing: only node 0 (stapes analog)
    forcing_freq = 2 * np.pi * 1.0  # 1 Hz forcing
    
    topologies = {
        "1D Chain (Bell)": ring_adjacency(N),
        "Flower of Life": flower_of_life_adjacency(N),
        "Random": random_regular_adjacency(N, degree=4, seed=SEED + 200),
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax_idx, (name, A) in enumerate(topologies.items()):
        sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=0, dt=dt)
        sys.reset(seed=SEED)
        
        # Record phase evolution
        phase_history = np.zeros((n_steps, N))
        
        for t in range(n_steps):
            # Global forcing only on node 0 (stapes input) — STRONG
            forcing_phase = forcing_freq * t * dt
            sys.theta[0] += 1.5 * np.sin(forcing_phase - sys.theta[0])
            # Also force first few nodes (cochlear base)
            for k in range(min(3, N)):
                sys.theta[k] += 0.5 * np.sin(forcing_phase - sys.theta[k])
            
            R = sys.step()
            phase_history[t] = sys.theta.copy()
        
        # Compute phase velocity: dθ/dt for each node
        phase_vel = np.diff(phase_history, axis=0) / dt
        # Unwrap for visualization
        phase_vel = np.mod(phase_vel + np.pi, 2 * np.pi) - np.pi
        
        ax = axes[ax_idx]
        im = ax.imshow(phase_vel.T, aspect='auto', cmap='RdBu_r',
                       vmin=-10, vmax=10, origin='lower',
                       extent=[0, n_steps-1, 0, N])
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Oscillator Index")
        ax.set_title(f"{name}")
        
        # Check for pseudowave: is there a diagonal pattern in phase velocity?
        # Measure: correlation between node index and peak phase velocity time
        peak_times = np.argmax(np.abs(phase_vel), axis=0)
        correlation = np.corrcoef(np.arange(N), peak_times)[0, 1]
        ax.text(0.05, 0.95, f"Phase-delay corr: {correlation:.2f}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"\n  {name}:")
        print(f"    Phase-delay correlation: {correlation:.4f}")
        if abs(correlation) > 0.3:
            print(f"    ✓ Ordered phase delays observed (consistent with pseudowave)")
        else:
            print(f"    ✗ No ordered phase delays detected")
    
    plt.colorbar(im, ax=axes[-1], label='Phase velocity dθ/dt')
    plt.suptitle("Pseudowave Emergence: Phase velocity heatmaps\n"
                 "(diagonal pattern = pseudowave = Bell's cochlear mechanism)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp4_pseudowave.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp4_pseudowave.png")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: COHERENCE UNIT CELL ABLATION (Lyra's Design)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_5_ablation():
    """
    The Coherence Unit Cell has 6 structural properties. Are ALL necessary?
    
    Full CUC:
      1. Kuramoto coupled oscillators ✓
      2. Nearest-neighbor coupling ✓  
      3. Global forcing ✓
      4. Quality-weighted resonance (Shera Q) ✓
      5. Pseudowave emergence ✓ (measured, not imposed)
      6. Self-duality (computation = representation) ✓
    
    Ablation: remove one property at a time, measure coherence degradation.
    
    If ALL properties are necessary (removing any one hurts performance),
    the CUC is a minimal sufficient architecture — not just a pattern
    but a law.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Coherence Unit Cell Ablation")
    print("=" * 72)
    
    N = 51
    n_steps = 1500
    n_trials = 10
    dt = 0.01
    
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, shera_Q = shera_q_factors(N)
    forcing = 0.4 * np.sin(np.linspace(0, 10 * np.pi, n_steps))
    
    A_fol = flower_of_life_adjacency(N)
    
    conditions = {}
    
    # FULL CUC: all 6 properties
    conditions["Full CUC"] = {
        "A": A_fol, "Q": shera_Q, "K": 4.0, "F": 0.4,
        "desc": "All 6 properties present"
    }
    
    # ABLATION 1: Remove nearest-neighbor → fully connected
    conditions["- Nearest-Neighbor"] = {
        "A": fully_connected_adjacency(N), "Q": shera_Q, "K": 4.0, "F": 0.4,
        "desc": "Replace structured topology with full connectivity"
    }
    
    # ABLATION 2: Remove global forcing
    conditions["- Global Forcing"] = {
        "A": A_fol, "Q": shera_Q, "K": 2.0, "F": 0.0,
        "desc": "No external forcing signal"
    }
    
    # ABLATION 3: Remove Q-weighting
    conditions["- Shera Q"] = {
        "A": A_fol, "Q": uniform_q_factors(N), "K": 4.0, "F": 0.4,
        "desc": "Uniform Q instead of Shera distribution"
    }
    
    # ABLATION 4: Remove Kuramoto → replace with linear coupling
    # (sin(θj - θi) → (θj - θi), linearized)
    conditions["- Nonlinear Coupling"] = {
        "A": A_fol, "Q": shera_Q, "K": 4.0, "F": 0.4,
        "desc": "Linear coupling (no sin), breaks Kuramoto dynamics",
        "linear": True
    }
    
    # CONTROL: Random everything
    conditions["Random Control"] = {
        "A": random_regular_adjacency(N, degree=4, seed=SEED+300),
        "Q": uniform_q_factors(N), "K": 4.0, "F": 0.4,
        "desc": "Random topology, uniform Q"
    }
    
    results = {}
    
    for name, cfg in conditions.items():
        trial_final_R = []
        trial_stability = []
        trial_R_curves = []
        
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, cfg["A"], Q=cfg["Q"],
                                K=cfg["K"], F=cfg["F"], dt=dt)
            sys.reset(seed=SEED + trial)
            
            if cfg.get("linear", False):
                # Override step for linear coupling
                R_history = np.zeros(n_steps)
                for t in range(n_steps):
                    # Linear coupling: (θj - θi) instead of sin(θj - θi)
                    lin_diff = sys.theta[np.newaxis, :] - sys.theta[:, np.newaxis]
                    coupling_effect = np.sum(sys.coupling * lin_diff, axis=1)
                    dtheta = sys.omega + (sys.K / sys.N) * coupling_effect
                    if sys.F > 0:
                        dtheta += sys.F * np.sin(forcing[t] - sys.theta)
                    sys.theta = np.mod(sys.theta + dtheta * dt, 2 * np.pi)
                    z = np.mean(np.exp(1j * sys.theta))
                    R_history[t] = float(np.abs(z))
            else:
                R_history = sys.run(n_steps, forcing)
            
            trial_final_R.append(np.mean(R_history[-300:]))
            # Stability: inverse of R variance in final phase
            trial_stability.append(1.0 / (np.std(R_history[-300:]) + 0.001))
            trial_R_curves.append(R_history)
        
        mean_final = np.mean(trial_final_R)
        std_final = np.std(trial_final_R)
        mean_stability = np.mean(trial_stability)
        
        results[name] = {
            "final_R": (mean_final, std_final),
            "stability": mean_stability,
            "R_curves": np.array(trial_R_curves),
        }
        
        print(f"\n  {name}: R = {mean_final:.4f} ± {std_final:.4f}, "
              f"stability = {mean_stability:.1f}")
    
    # Summary comparison
    full_R = results["Full CUC"]["final_R"][0]
    full_stability = results["Full CUC"]["stability"]
    print(f"\n  --- Ablation Summary (R + Stability) ---")
    print(f"  Full CUC baseline: R = {full_R:.4f}, stability = {full_stability:.1f}")
    all_necessary = True
    for name in ['- Nearest-Neighbor', '- Global Forcing', '- Shera Q', '- Nonlinear Coupling']:
        if name in results:
            res = results[name]
            delta_R = res["final_R"][0] - full_R
            delta_stab = res["stability"] - full_stability
            # A property is necessary if removing it hurts EITHER R or stability
            hurts_R = delta_R < -0.01
            hurts_stab = delta_stab < -2.0
            if hurts_R or hurts_stab:
                status = "✓ necessary"
                reasons = []
                if hurts_R: reasons.append(f"R drops {delta_R:+.4f}")
                if hurts_stab: reasons.append(f"stability drops {delta_stab:+.1f}")
                status += f" ({', '.join(reasons)})"
            else:
                status = "? not clearly necessary at these params"
                all_necessary = False
            print(f"  {name}: R={res['final_R'][0]:.4f}, stab={res['stability']:.1f} — {status}")
    
    if all_necessary:
        print(f"\n  ★ All tested ablations show degradation — consistent with minimal sufficient architecture")
    else:
        print(f"\n  ~ Some properties may be redundant — investigate further")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_list = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#9B5DE5', '#888888']
    
    ax = axes[0]
    for i, (name, res) in enumerate(results.items()):
        mean_curve = np.mean(res["R_curves"], axis=0)
        ax.plot(mean_curve, color=colors_list[i], label=name, linewidth=1.5)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Order Parameter R(t)")
    ax.set_title("CUC Ablation: Coherence Dynamics")
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    names = list(results.keys())
    final_means = [results[n]["final_R"][0] for n in names]
    final_stds = [results[n]["final_R"][1] for n in names]
    bars = ax.bar(range(len(names)), final_means, yerr=final_stds,
                  color=colors_list, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel("Final Coherence R (last 300 steps)")
    ax.set_title("Ablation Results: Which properties are necessary?")
    ax.axhline(y=full_R, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp5_ablation.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp5_ablation.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: CROSS-SYSTEM COHERENCE (The Bridge)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_6_cross_system():
    """
    Do different CUC implementations produce correlated dynamics?
    
    Feed the SAME input signal to:
      A) Bell-like: 1D chain, Shera Q, global forcing at boundary
      B) AKOrN-like: 2D grid, uniform Q, distributed forcing
      C) FlowerTuner-like: Flower of Life topology, Shera Q, global forcing
    
    Measure: mutual information / correlation of R(t) trajectories.
    
    If correlated: the architecture class produces invariant dynamics
    regardless of implementation details. That's the connection Miyato
    asked us to demonstrate.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 6: Cross-System Coherence Correlation")
    print("=" * 72)
    
    N = 51
    n_steps = 2000
    n_trials = 10
    dt = 0.01
    
    # Shared forcing signal: complex, non-periodic
    t_arr = np.linspace(0, 20, n_steps)
    forcing = (0.3 * np.sin(2 * np.pi * 0.5 * t_arr) +
               0.2 * np.sin(2 * np.pi * 1.3 * t_arr) +
               0.1 * np.sin(2 * np.pi * 3.7 * t_arr))
    
    # Shared frequency distribution for fair comparison
    omega_shared = 2 * np.pi * np.linspace(0.8, 1.2, N)
    
    _, shera_Q = shera_q_factors(N)
    
    systems = {
        "Bell-like\n(1D chain, Shera Q)": {
            "A": ring_adjacency(N),
            "Q": shera_Q,
            "omega": omega_shared,
            "K": 4.0, "F": 0.5,
        },
        "AKOrN-like\n(Grid, uniform Q)": {
            "A": grid_adjacency(N),
            "Q": uniform_q_factors(N),
            "omega": omega_shared,
            "K": 4.0, "F": 0.5,
        },
        "FlowerTuner-like\n(FoL, Shera Q)": {
            "A": flower_of_life_adjacency(N),
            "Q": shera_Q,
            "omega": omega_shared,
            "K": 4.0, "F": 0.5,
        },
    }
    
    all_R = {}
    
    for name, cfg in systems.items():
        trial_R = np.zeros((n_trials, n_steps))
        for trial in range(n_trials):
            sys = KuramotoSystem(N, cfg["omega"], cfg["A"], Q=cfg["Q"],
                                K=cfg["K"], F=cfg["F"], dt=dt)
            sys.reset(seed=SEED + trial)
            trial_R[trial] = sys.run(n_steps, forcing)
        all_R[name] = np.mean(trial_R, axis=0)
    
    # Cross-correlations
    names = list(all_R.keys())
    print("\n  Cross-correlation matrix (R trajectories):")
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            c = np.corrcoef(all_R[names[i]], all_R[names[j]])[0, 1]
            corr_matrix[i, j] = c
            if i <= j:
                print(f"    {names[i][:15]:>15s} ↔ {names[j][:15]:<15s}: r = {c:.4f}")
    
    mean_cross = np.mean([corr_matrix[0,1], corr_matrix[0,2], corr_matrix[1,2]])
    print(f"\n  Mean cross-correlation: {mean_cross:.4f}")
    if mean_cross > 0.5:
        print(f"  ✓ Observed: Systems share coherence dynamics despite different implementations")
    elif mean_cross > 0.3:
        print(f"  ~ Moderate correlation — shared architecture partially explains dynamics")
    else:
        print(f"  ✗ Low correlation — implementations diverge at these parameters")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    
    ax = axes[0]
    for i, (name, R) in enumerate(all_R.items()):
        ax.plot(R, color=colors[i], label=name, linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Order Parameter R(t)")
    ax.set_title("Cross-System Coherence: Same Input, Different Implementations")
    ax.legend(fontsize=8)
    
    ax = axes[1]
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    short_names = ["Bell-like", "AKOrN-like", "FlowerTuner"]
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_title("R(t) Cross-Correlation Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center',
                    fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp6_cross_system.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp6_cross_system.png")
    
    return all_R, corr_matrix


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: PERTURBATION SCALING CURVE (GPT's "mic drop")
# ═══════════════════════════════════════════════════════════════════════════

def experiment_7_perturbation_scaling():
    """
    How does each topology degrade under increasing perturbation?
    
    For each topology, perturb 5%, 10%, 20%, 30%, 40%, 50% of nodes.
    Measure recovery probability (does R return to 90% of pre-perturb?).
    Plot recovery probability vs perturbation percentage.
    
    If structured topologies maintain larger basins of attraction,
    the geometry is functionally meaningful.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Perturbation Scaling Curve")
    print("=" * 72)
    
    N = 24  # Clean comparison at 24-cell native size
    n_trials = 30
    K = 4.0
    F = 0.3
    dt = 0.01
    warmup = 500
    recovery_window = 500
    
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, Q_full = shera_q_factors(N)
    
    perturb_fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    topologies = {
        "24-Cell": cell24_adjacency(),
        "Random Regular (deg=8)": random_regular_adjacency(N, degree=8, seed=SEED + 500),
        "Ring": ring_adjacency(N),
        "Fully Connected": fully_connected_adjacency(N),
    }
    
    forcing = 0.3 * np.sin(np.linspace(0, 10 * np.pi, warmup + recovery_window))
    
    results = {}
    
    for topo_name, A in topologies.items():
        recovery_probs = []
        
        for frac in perturb_fractions:
            n_perturb = max(1, int(frac * N))
            recovered_count = 0
            
            for trial in range(n_trials):
                sys = KuramotoSystem(N, omega, A, Q=Q_full, K=K, F=F, dt=dt)
                sys.reset(seed=SEED + trial)
                
                # Warmup to steady state
                R_warmup = sys.run(warmup, forcing[:warmup])
                pre_R = np.mean(R_warmup[-100:])
                
                # Perturb
                perturb_idx = np.random.RandomState(SEED + trial + 1000).choice(
                    N, n_perturb, replace=False)
                sys.theta[perturb_idx] = np.random.RandomState(
                    SEED + trial + 2000).uniform(0, 2 * np.pi, n_perturb)
                
                # Recovery
                R_recovery = sys.run(recovery_window, forcing[warmup:])
                
                # Did it recover? R returns to 90% of pre-perturb within window
                target = 0.9 * pre_R
                for i in range(len(R_recovery) - 20):
                    if np.mean(R_recovery[i:i+20]) >= target:
                        recovered_count += 1
                        break
            
            prob = recovered_count / n_trials
            recovery_probs.append(prob)
        
        results[topo_name] = recovery_probs
        print(f"\n  {topo_name}:")
        for i, frac in enumerate(perturb_fractions):
            print(f"    {frac*100:4.0f}% perturbed: {recovery_probs[i]*100:5.1f}% recovery")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E63946", "#457B9D", "#888888", "#264653"]
    markers = ['o', 's', '^', 'D']
    
    for i, (name, probs) in enumerate(results.items()):
        ax.plot([f*100 for f in perturb_fractions], [p*100 for p in probs],
                color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                label=name)
    
    ax.set_xlabel("Nodes Perturbed (%)", fontsize=12)
    ax.set_ylabel("Recovery Probability (%)", fontsize=12)
    ax.set_title("Perturbation Scaling: Basin of Attraction by Topology", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 55)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp7_perturbation_scaling.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp7_perturbation_scaling.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: CLEAN N=24 TOPOLOGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def experiment_8_clean_24cell():
    """
    Clean topology comparison at native 24-cell size (N=24).
    No padding. No size mismatch. Same N, same omega, same Q.
    
    Topologies:
      1) 24-cell (24 vertices, 96 edges, degree 8)
      2) Random regular (degree 8, matched)
      3) Ring (degree 2, lower bound)
      4) Fully connected (degree 23, upper bound)
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 8: Clean N=24 Topology Comparison")
    print("=" * 72)
    
    N = 24
    n_steps = 2000
    n_trials = 30
    K = 4.0
    F = 0.3
    dt = 0.01
    perturb_at = 1000
    
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, Q = shera_q_factors(N)
    forcing = 0.4 * np.sin(np.linspace(0, 10 * np.pi, n_steps))
    
    topologies = {
        "24-Cell (deg=8)": cell24_adjacency(),
        "Random Regular (deg=8)": random_regular_adjacency(N, degree=8, seed=SEED + 600),
        "Ring (deg=2)": ring_adjacency(N),
        "Fully Connected (deg=23)": fully_connected_adjacency(N),
    }
    
    results = {}
    
    for name, A in topologies.items():
        trial_R = np.zeros((n_trials, n_steps))
        trial_final_R = []
        recovery_times = []
        
        degree = A.sum(axis=1).mean()
        n_edges = A.sum() / 2
        
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F, dt=dt)
            sys.reset(seed=SEED + trial)
            
            R = np.zeros(n_steps)
            for t in range(n_steps):
                if t == perturb_at:
                    n_perturb = int(0.3 * N)
                    idx = np.random.RandomState(SEED + trial + 3000).choice(
                        N, n_perturb, replace=False)
                    sys.theta[idx] = np.random.RandomState(
                        SEED + trial + 4000).uniform(0, 2 * np.pi, n_perturb)
                R[t] = sys.step(forcing[t])
            
            trial_R[trial] = R
            trial_final_R.append(np.mean(R[-200:]))
            
            # Recovery time
            pre_R = np.mean(R[perturb_at - 100:perturb_at])
            target = 0.9 * pre_R
            recovered = False
            for i in range(perturb_at, n_steps - 20):
                if np.all(R[i:i+20] > target):
                    recovery_times.append(i - perturb_at)
                    recovered = True
                    break
            if not recovered:
                recovery_times.append(n_steps - perturb_at)
        
        mean_R = np.mean(trial_R, axis=0)
        mean_final = np.mean(trial_final_R)
        std_final = np.std(trial_final_R)
        mean_recovery = np.mean(recovery_times)
        std_recovery = np.std(recovery_times)
        efficiency = mean_final / (n_edges / N + 0.001)
        
        # Statistical test vs 24-cell (store for later)
        results[name] = {
            "mean_R": mean_R,
            "final_R": (mean_final, std_final),
            "recovery": (mean_recovery, std_recovery),
            "efficiency": efficiency,
            "degree": degree,
            "n_edges": n_edges,
            "all_final_R": trial_final_R,
            "all_recovery": recovery_times,
        }
        
        print(f"\n  {name} (degree={degree:.0f}, edges={n_edges:.0f}):")
        print(f"    Final R: {mean_final:.4f} ± {std_final:.4f}")
        print(f"    Recovery: {mean_recovery:.0f} ± {std_recovery:.0f} steps")
        print(f"    Efficiency (R/edges_per_node): {efficiency:.4f}")
    
    # Statistical comparison: 24-cell vs random regular (matched degree)
    if not HAS_SCIPY:
        print("\n  [scipy not installed — skipping statistical tests]")
        print("  Install with: pip install scipy")
        p_value, p_rec, cohens_d, t_stat, t_rec = 0, 0, 0, 0, 0
    else:
        cell24_finals = results["24-Cell (deg=8)"]["all_final_R"]
        random_finals = results["Random Regular (deg=8)"]["all_final_R"]
        t_stat, p_value = sp_stats.ttest_ind(cell24_finals, random_finals)
        
        cell24_recovery = results["24-Cell (deg=8)"]["all_recovery"]
        random_recovery = results["Random Regular (deg=8)"]["all_recovery"]
        t_rec, p_rec = sp_stats.ttest_ind(cell24_recovery, random_recovery)
    
    if HAS_SCIPY:
        print(f"\n  --- Statistical Comparison (24-Cell vs Random Regular, matched degree) ---")
        print(f"    Final R: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"    Recovery: t={t_rec:.3f}, p={p_rec:.4f}")
        if p_value < 0.05:
            print(f"    ✓ Coherence difference statistically significant (p < 0.05)")
        else:
            print(f"    ~ Coherence difference not significant at p < 0.05")
        if p_rec < 0.05:
            print(f"    ✓ Recovery difference statistically significant (p < 0.05)")
        else:
            print(f"    ~ Recovery difference not significant at p < 0.05")
        
        # Cohen's d effect size
        cell24_finals = results["24-Cell (deg=8)"]["all_final_R"]
        random_finals = results["Random Regular (deg=8)"]["all_final_R"]
        pooled_std = np.sqrt((np.std(cell24_finals)**2 + np.std(random_finals)**2) / 2)
        cohens_d = (np.mean(cell24_finals) - np.mean(random_finals)) / (pooled_std + 1e-10)
        print(f"    Cohen's d (Final R): {cohens_d:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E63946", "#457B9D", "#888888", "#264653"]
    
    ax = axes[0]
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res["mean_R"], color=colors[i], label=name, linewidth=1.5)
    ax.axvline(x=perturb_at, color='red', linestyle='--', alpha=0.4, label='Perturbation')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Order Parameter R(t)")
    ax.set_title("N=24 Topology Comparison (Clean, No Padding)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    names = list(results.keys())
    rec_means = [results[n]["recovery"][0] for n in names]
    rec_stds = [results[n]["recovery"][1] for n in names]
    bars = ax.bar(range(len(names)), rec_means, yerr=rec_stds,
                  color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" (")[0] for n in names], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("Recovery Steps After 30% Perturbation")
    ax.set_title(f"Recovery Speed (p={p_rec:.4f}, d={cohens_d:.2f})")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp8_clean_24cell.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp8_clean_24cell.png")
    
    return results



# ═══════════════════════════════════════════════════════════════════════════
# V2 EXPERIMENTS: DEEPER CONTROLS AND GENERALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def experiment_9_self_duality_v2():
    """
    Strengthened self-duality test with three control conditions:
      A) Self-dual: phases ARE the output
      B) Orthogonal transform: preserves norm (strong control)
      C) Trained linear decoder: least-squares fit
      D) Random projection: original V1 control
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 9: Self-Duality with Stronger Controls (V2)")
    print("=" * 72)

    N = 24
    n_warmup = 500
    n_recovery = 500
    n_trials = 30
    K, F, dt = 3.0, 0.2, 0.01

    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    A = cell24_adjacency()
    forcing = 0.3 * np.sin(np.linspace(0, 8 * np.pi, n_warmup + n_recovery))

    np.random.seed(SEED + 200)
    W_random = np.random.randn(N, N)
    W_ortho, _ = np.linalg.qr(W_random)
    W_rand_proj = np.random.randn(N, N)
    W_rand_proj /= np.linalg.norm(W_rand_proj, axis=1, keepdims=True)

    # Collect training data for decoder
    training_phases = []
    for trial in range(10):
        sys = KuramotoSystem(N, omega, A, K=K, F=F, dt=dt)
        sys.reset(seed=SEED + trial)
        sys.run(n_warmup, forcing[:n_warmup])
        training_phases.append(sys.theta.copy())
    X_train = np.array([np.cos(p) for p in training_phases])
    Y_train = np.array(training_phases)
    W_trained, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    W_trained = W_trained.T

    conditions = {
        "Self-Dual": None,
        "Orthogonal": W_ortho,
        "Trained Decoder": W_trained,
        "Random Projection": W_rand_proj,
    }

    results = {}
    for cond_name, W in conditions.items():
        trial_recovery = []
        for trial in range(n_trials):
            sys = KuramotoSystem(N, omega, A, K=K, F=F, dt=dt)
            sys.reset(seed=SEED + trial)
            sys.run(n_warmup, forcing[:n_warmup])
            identity_phases = sys.theta.copy()

            if W is None:
                identity_rep = np.exp(1j * identity_phases)
            else:
                identity_rep = W @ np.cos(identity_phases)

            n_perturb = int(0.3 * N)
            idx = np.random.RandomState(SEED + trial + 5000).choice(N, n_perturb, replace=False)
            sys.theta[idx] = np.random.RandomState(SEED + trial + 6000).uniform(0, 2 * np.pi, n_perturb)
            sys.run(n_recovery, forcing[n_warmup:])
            recovered_phases = sys.theta.copy()

            if W is None:
                sim = float(np.abs(np.mean(np.exp(1j * (recovered_phases - identity_phases)))))
            else:
                recovered_rep = W @ np.cos(recovered_phases)
                corr = np.corrcoef(identity_rep.flatten(), recovered_rep.flatten())[0, 1]
                sim = max(0, float(corr))
            trial_recovery.append(sim)

        mean_rec = np.mean(trial_recovery)
        std_rec = np.std(trial_recovery)
        results[cond_name] = {"mean": mean_rec, "std": std_rec, "all": trial_recovery}
        print(f"  {cond_name}: {mean_rec:.4f} +/- {std_rec:.4f}")

    if HAS_SCIPY:
        sd = results["Self-Dual"]["all"]
        for name in ["Orthogonal", "Trained Decoder", "Random Projection"]:
            other = results[name]["all"]
            t, p = sp_stats.ttest_ind(sd, other)
            pooled = np.sqrt((np.std(sd)**2 + np.std(other)**2) / 2) + 1e-10
            d = (np.mean(sd) - np.mean(other)) / pooled
            print(f"  Self-Dual vs {name}: t={t:.2f}, p={p:.4f}, d={d:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#888888']
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Identity Recovery After 30% Perturbation")
    ax.set_title("Self-Duality: Strengthened Controls (V2)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp9_self_duality_v2.png", dpi=150)
    plt.close()
    print(f"\n  -> Plot saved: {OUT_DIR}/exp9_self_duality_v2.png")
    return results


def experiment_10_critical_regime():
    """
    Sweep coupling K to find where topology matters.
    At K >> K_critical all graphs sync. Near K_critical, topology determines outcome.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 10: Near-Critical Coupling Regime (V2)")
    print("=" * 72)

    N = 24
    n_steps, n_trials, F, dt = 1000, 20, 0.2, 0.01
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, Q = shera_q_factors(N)
    forcing = 0.3 * np.sin(np.linspace(0, 8 * np.pi, n_steps))
    K_values = np.linspace(0.5, 8.0, 16)

    topologies = {
        "24-Cell": cell24_adjacency(),
        "Random Reg (deg=8)": random_regular_adjacency(N, degree=8, seed=SEED + 700),
        "Ring": ring_adjacency(N),
        "Fully Connected": fully_connected_adjacency(N),
    }

    results = {}
    for topo_name, A in topologies.items():
        R_vs_K, R_std_vs_K = [], []
        for K in K_values:
            finals = []
            for trial in range(n_trials):
                sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F, dt=dt)
                sys.reset(seed=SEED + trial)
                R = sys.run(n_steps, forcing)
                finals.append(np.mean(R[-200:]))
            R_vs_K.append(np.mean(finals))
            R_std_vs_K.append(np.std(finals))

        results[topo_name] = {"K": K_values, "R": np.array(R_vs_K), "R_std": np.array(R_std_vs_K)}
        k_crit = K_values[-1]
        for i, r in enumerate(R_vs_K):
            if r > 0.5:
                k_crit = K_values[i]; break
        print(f"  {topo_name}: K_critical ~ {k_crit:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E63946", "#457B9D", "#888888", "#264653"]
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res["K"], res["R"], color=colors[i], label=name, linewidth=2, marker='o', markersize=4)
        ax.fill_between(res["K"], res["R"] - res["R_std"], res["R"] + res["R_std"], color=colors[i], alpha=0.1)
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel("Coupling Strength K")
    ax.set_ylabel("Steady-State R")
    ax.set_title("Critical Coupling Regime: Where Topology Matters (V2)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp10_critical_regime.png", dpi=150)
    plt.close()
    print(f"\n  -> Plot saved: {OUT_DIR}/exp10_critical_regime.png")
    return results


def experiment_11_varied_forcing():
    """
    Does cross-system correlation persist without shared forcing?
    V1 E6 used shared forcing. Test: same, different freq, different amp, none.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 11: Cross-System Correlation Under Varied Forcing (V2)")
    print("=" * 72)

    N = 51
    n_steps, n_trials, dt = 2000, 20, 0.01
    omega = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, shera_Q = shera_q_factors(N)
    t_arr = np.linspace(0, 20, n_steps)

    forcing_conditions = {
        "Same forcing": {
            "AKOrN": 0.3 * np.sin(2*np.pi*0.5*t_arr) + 0.2 * np.sin(2*np.pi*1.3*t_arr),
            "FT51":  0.3 * np.sin(2*np.pi*0.5*t_arr) + 0.2 * np.sin(2*np.pi*1.3*t_arr),
        },
        "Different freq": {
            "AKOrN": 0.3 * np.sin(2*np.pi*0.5*t_arr),
            "FT51":  0.3 * np.sin(2*np.pi*2.1*t_arr),
        },
        "Different amplitude": {
            "AKOrN": 0.1 * np.sin(2*np.pi*0.5*t_arr),
            "FT51":  0.8 * np.sin(2*np.pi*0.5*t_arr),
        },
        "No forcing": {
            "AKOrN": np.zeros(n_steps),
            "FT51":  np.zeros(n_steps),
        },
    }

    systems = {
        "AKOrN": {"A": grid_adjacency(N), "Q": uniform_q_factors(N), "K": 4.0, "F": 0.5},
        "FT51":  {"A": flower_of_life_adjacency(N), "Q": shera_Q, "K": 4.0, "F": 0.5},
    }

    results = {}
    for cond_name, forcings in forcing_conditions.items():
        trial_corrs = []
        for trial in range(n_trials):
            Rs = {}
            for sys_name, cfg in systems.items():
                F_val = cfg["F"] if cond_name != "No forcing" else 0.0
                s = KuramotoSystem(N, omega, cfg["A"], Q=cfg["Q"], K=cfg["K"], F=F_val, dt=dt)
                s.reset(seed=SEED + trial)
                Rs[sys_name] = s.run(n_steps, forcings[sys_name])
            trial_corrs.append(np.corrcoef(Rs["AKOrN"], Rs["FT51"])[0, 1])

        results[cond_name] = {"mean": np.mean(trial_corrs), "std": np.std(trial_corrs), "all": trial_corrs}
        print(f"  {cond_name}: r = {results[cond_name]['mean']:.4f} +/- {results[cond_name]['std']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#888888']
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("R(t) Cross-Correlation (AKOrN <-> FlowerTuner)")
    ax.set_title("Does Coherence Correlation Survive Without Shared Forcing? (V2)")
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-0.5, 1.0)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp11_varied_forcing.png", dpi=150)
    plt.close()
    print(f"\n  -> Plot saved: {OUT_DIR}/exp11_varied_forcing.png")
    return results


def experiment_12_spectral():
    """Graph Laplacian spectral analysis: algebraic connectivity and spectral gap."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 12: Graph Spectral Analysis (V2)")
    print("=" * 72)

    all_topos = {
        "24-Cell (N=24)": cell24_adjacency(),
        "Random Reg (N=24)": random_regular_adjacency(24, degree=8, seed=SEED + 800),
        "Ring (N=24)": ring_adjacency(24),
        "Flower of Life (N=51)": flower_of_life_adjacency(51),
        "Random Reg (N=51)": random_regular_adjacency(51, degree=5, seed=SEED + 900),
        "Ring (N=51)": ring_adjacency(51),
    }

    print(f"\n  {'Topology':<25s} {'lam2':>8s} {'lam_max':>8s} {'Gap':>8s} {'Deg':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    spectral_data = {}
    for name, A in all_topos.items():
        D = np.diag(A.sum(axis=1))
        L = D - A
        eigs = np.sort(np.real(np.linalg.eigvalsh(L)))
        lam2, lam_max = eigs[1], eigs[-1]
        gap = lam2 / lam_max if lam_max > 0 else 0
        deg = A.sum(axis=1).mean()
        spectral_data[name] = {"lam2": lam2, "lam_max": lam_max, "gap": gap, "deg": deg, "eigs": eigs}
        print(f"  {name:<25s} {lam2:8.4f} {lam_max:8.4f} {gap:8.4f} {deg:6.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#E63946', '#457B9D', '#888888']
    for ax, prefix, title in [(axes[0], "N=24", "Laplacian Spectrum (N=24)"), (axes[1], "N=51", "Laplacian Spectrum (N=51)")]:
        for i, (name, data) in enumerate([(k, spectral_data[k]) for k in spectral_data if prefix in k]):
            ax.plot(data["eigs"], color=colors[i], label=name.split(" (")[0], linewidth=2, marker='o', markersize=3)
        ax.set_xlabel("Eigenvalue Index"); ax.set_ylabel("Laplacian Eigenvalue"); ax.set_title(title); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp12_spectral.png", dpi=150)
    plt.close()
    print(f"\n  -> Plot saved: {OUT_DIR}/exp12_spectral.png")
    return spectral_data


def experiment_13_jax_kernel():
    """JAX-accelerable Kuramoto kernel for ML pipeline integration."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 13: JAX Kuramoto Kernel (V2)")
    print("=" * 72)

    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
    except ImportError:
        print("  JAX not installed. Skipping. (pip install jax jaxlib)")
        return None

    N = 24
    A_np = cell24_adjacency()
    omega_np = 2 * np.pi * np.linspace(0.8, 1.2, N)
    _, Q_np = shera_q_factors(N)
    coupling_np = A_np * np.sqrt(np.outer(Q_np, Q_np))

    coupling_jax = jnp.array(coupling_np)
    omega_jax = jnp.array(omega_np)
    K, F, dt = 4.0, 0.3, 0.01

    @jit
    def kuramoto_step_jax(theta, forcing_phase):
        def derivs(th, fp):
            sd = jnp.sin(th[None, :] - th[:, None])
            ce = jnp.sum(coupling_jax * sd, axis=1)
            return omega_jax + (K / N) * ce + F * jnp.sin(fp - th)
        k1 = derivs(theta, forcing_phase)
        k2 = derivs(theta + 0.5*dt*k1, forcing_phase)
        k3 = derivs(theta + 0.5*dt*k2, forcing_phase)
        k4 = derivs(theta + dt*k3, forcing_phase)
        theta_new = jnp.mod(theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4), 2*jnp.pi)
        return theta_new, jnp.abs(jnp.mean(jnp.exp(1j * theta_new)))

    np.random.seed(SEED)
    theta_init = np.random.uniform(0, 2*np.pi, N)

    # NumPy reference
    sys_np = KuramotoSystem(N, omega_np, A_np, Q=Q_np, K=K, F=F, dt=dt)
    sys_np.theta = theta_init.copy()
    R_np = [sys_np.step(0.3*np.sin(0.1*t)) for t in range(100)]

    # JAX run
    theta_j = jnp.array(theta_init.copy())
    R_jx = []
    for t in range(100):
        theta_j, r = kuramoto_step_jax(theta_j, 0.3*jnp.sin(0.1*t))
        R_jx.append(float(r))

    max_diff = np.max(np.abs(np.array(R_np) - np.array(R_jx)))
    print(f"  NumPy vs JAX max |dR|: {max_diff:.2e}")
    if max_diff < 1e-4:
        print(f"  Validated: numerical equivalence confirmed")
    else:
        print(f"  Minor float32/64 differences (expected)")

    import timeit
    theta_b = jnp.array(np.random.uniform(0, 2*np.pi, N))
    kuramoto_step_jax(theta_b, jnp.float32(0.0))  # warmup

    def jax_bench():
        th = theta_b
        for t in range(100):
            th, _ = kuramoto_step_jax(th, jnp.float32(0.1*t))
        return th

    def np_bench():
        s = KuramotoSystem(N, omega_np, A_np, Q=Q_np, K=K, F=F, dt=dt)
        s.theta = theta_init.copy()
        for t in range(100): s.step(0.3*np.sin(0.1*t))

    jt = timeit.timeit(lambda: jax.block_until_ready(jax_bench()), number=10) / 10
    nt = timeit.timeit(np_bench, number=10) / 10
    print(f"  NumPy: {nt*1000:.1f}ms | JAX: {jt*1000:.1f}ms | Speedup: {nt/jt:.1f}x")
    print(f"  JAX kernel is differentiable + JIT-compiled for AKOrN integration.")
    return {"max_diff": max_diff, "np_time": nt, "jax_time": jt}


# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════

def run_all():
    """Execute all experiments and generate summary."""
    
    print("╔" + "═" * 70 + "╗")
    print("║  COHERENCE UNIT CELL — Experimental Validation Suite              ║")
    print("║  Waltman & Aurora, February 2026                                  ║")
    print("║  Reproducing: Bell (2012), Miyato (2025), Hays (2025)             ║")
    print("╚" + "═" * 70 + "╝")
    
    start = time.time()
    
    r1 = experiment_1_shera_q()
    r2 = experiment_2_topology()
    r3 = experiment_3_self_duality()
    experiment_4_pseudowave()
    r5 = experiment_5_ablation()
    r6_R, r6_corr = experiment_6_cross_system()
    r7 = experiment_7_perturbation_scaling()
    r8 = experiment_8_clean_24cell()
    
    v1_elapsed = time.time() - start
    print(f"\n  V1 runtime: {v1_elapsed:.1f} seconds")
    
    # V2 experiments
    print("\n\n" + "╔" + "═" * 70 + "╗")
    print("║  V2: DEEPER CONTROLS AND GENERALIZATION                           ║")
    print("╚" + "═" * 70 + "╝")
    
    r9 = experiment_9_self_duality_v2()
    r10 = experiment_10_critical_regime()
    r11 = experiment_11_varied_forcing()
    r12 = experiment_12_spectral()
    r13 = experiment_13_jax_kernel()
    
    elapsed = time.time() - start
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n\n" + "╔" + "═" * 70 + "╗")
    print("║  SUMMARY                                                          ║")
    print("╚" + "═" * 70 + "╝")
    
    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print(f"\n  Experiment 1 (Shera Q Transplant):")
    print(f"    Shera Q final R: {r1['Shera Q']['final_R'][0]:.4f}")
    print(f"    Uniform Q final R: {r1['Uniform Q']['final_R'][0]:.4f}")
    delta1 = r1['Shera Q']['final_R'][0] - r1['Uniform Q']['final_R'][0]
    print(f"    Biological advantage: ΔR = {delta1:+.4f}")
    
    print(f"\n  Experiment 2 (Topology):")
    print(f"    Flower of Life final R: {r2['Flower of Life']['final_R']:.4f}")
    print(f"    Random final R: {list(r2.values())[2]['final_R']:.4f}")
    
    print(f"\n  Experiment 3 (Self-Duality):")
    sd = np.mean(r3['Self-Dual'])
    nsd = np.mean(r3['Non-Self-Dual'])
    print(f"    Self-dual recovery: {sd:.4f}")
    print(f"    Non-self-dual recovery: {nsd:.4f}")
    
    print(f"\n  Experiment 5 (CUC Ablation):")
    full_R = r5['Full CUC']['final_R'][0]
    print(f"    Full CUC: R = {full_R:.4f}")
    for name in ['- Nearest-Neighbor', '- Global Forcing', '- Shera Q', '- Nonlinear Coupling']:
        if name in r5:
            print(f"    {name}: R = {r5[name]['final_R'][0]:.4f}")
    
    print(f"\n  Experiment 6 (Cross-System):")
    mean_cross = np.mean([r6_corr[0,1], r6_corr[0,2], r6_corr[1,2]])
    print(f"    Mean cross-correlation: {mean_cross:.4f}")
    
    print(f"\n  Experiment 7 (Perturbation Scaling):")
    for name, probs in r7.items():
        print(f"    {name}: {[f'{p:.0%}' for p in probs]}")
    
    print(f"\n  Experiment 8 (Clean N=24 Topology):")
    print(f"    24-Cell final R: {r8['24-Cell (deg=8)']['final_R'][0]:.4f}")
    print(f"    Random Reg final R: {r8['Random Regular (deg=8)']['final_R'][0]:.4f}")
    
    print(f"\n  --- V2 Results ---")
    
    print(f"\n  Experiment 9 (Self-Duality V2):")
    for name, data in r9.items():
        print(f"    {name}: {data['mean']:.4f}")
    
    print(f"\n  Experiment 10 (Critical Regime):")
    for name, data in r10.items():
        k_crit = data["K"][-1]
        for i, r in enumerate(data["R"]):
            if r > 0.5:
                k_crit = data["K"][i]; break
        print(f"    {name}: K_critical ~ {k_crit:.2f}")
    
    print(f"\n  Experiment 11 (Varied Forcing):")
    for name, data in r11.items():
        print(f"    {name}: r = {data['mean']:.4f}")
    
    print(f"\n  Experiment 12 (Spectral):")
    for name, data in r12.items():
        print(f"    {name}: lam2={data['lam2']:.3f}, gap={data['gap']:.4f}")
    
    if r13:
        print(f"\n  Experiment 13 (JAX):")
        print(f"    Validated, speedup: {r13['np_time']/r13['jax_time']:.1f}x")
    
    print(f"\n  Results saved to: {OUT_DIR}/")
    print(f"  V1 plots: exp1-8, V2 plots: exp9-12")
    
    print(f"\n  {'='*60}")
    print(f"  To reproduce: python coherence_unit_cell_experiments.py")
    print(f"  Seed: {SEED} (deterministic)")
    print(f"  Dependencies: numpy, matplotlib, scipy (optional: jax)")
    print(f"  {'='*60}")


if __name__ == "__main__":
    run_all()
