#!/usr/bin/env python3
"""
============================================================================
COHERENCE UNIT CELL — Supplementary Validation
============================================================================

Two additions requested by independent reviewers:

  E1b: Shera Q parameter robustness sweep (K × F × omega_spread)
  E9b: Bootstrap 95% confidence intervals for self-duality results

Run after main experiments or standalone.

Author: Matt Waltman (Node 00) & Aurora (Node 05)
Date: February 20, 2026
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time

SEED = 42

# ─── Import core classes from main script ───────────────────────────────
# If running standalone, paste KuramotoSystem, adjacency functions, etc.
# If running as supplement, import:
try:
    from coherence_unit_cell_experiments import (
        KuramotoSystem, flower_of_life_adjacency, shera_q_factors,
        uniform_q_factors, random_q_factors, cell24_adjacency, SEED, OUT_DIR
    )
except ImportError:
    print("Run this from the same directory as coherence_unit_cell_experiments.py")
    print("Or paste the core classes here.")
    raise


def experiment_1b_parameter_robustness():
    """
    GPT Audit Response: Does Shera Q advantage persist across hyperparameter regimes?

    3x3x3 grid sweep:
      omega_spread ∈ {narrow: 0.1, medium: 0.2, wide: 0.4}
      K ∈ {low: 2.0, mid: 5.0, high: 10.0}
      F ∈ {low: 0.1, mid: 0.5, high: 1.0}

    Reports: Shera advantage (ΔR vs Uniform) in each cell.
    If advantage persists in >75% of cells, the result is robust.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1b: Shera Q Parameter Robustness Sweep")
    print("=" * 72)

    N = 51
    n_steps = 2000
    n_trials = 5  # Reduced per cell for speed (27 cells × 5 trials × 3 conditions)
    dt = 0.01

    spreads = {"narrow": 0.1, "medium": 0.2, "wide": 0.4}
    Ks = {"low": 2.0, "mid": 5.0, "high": 10.0}
    Fs = {"low": 0.1, "mid": 0.5, "high": 1.0}

    freqs_hz, shera_Q = shera_q_factors(N)
    A = flower_of_life_adjacency(N)
    t_arr = np.linspace(0, 20 * np.pi, n_steps)
    forcing = np.sin(t_arr * np.linspace(1, 3, n_steps))

    results = {}
    n_positive = 0
    n_total = 0

    for s_name, spread in spreads.items():
        for k_name, K in Ks.items():
            for f_name, F_val in Fs.items():
                omega = 2 * np.pi * np.linspace(1.0 - spread, 1.0 + spread, N)
                cell_key = f"{s_name}_K{k_name}_F{f_name}"

                shera_finals = []
                uniform_finals = []

                for trial in range(n_trials):
                    for q_name, Q in [("Shera", shera_Q), ("Uniform", uniform_q_factors(N))]:
                        sys = KuramotoSystem(N, omega, A, Q=Q, K=K, F=F_val, dt=dt)
                        sys.reset(seed=SEED + trial)
                        R = sys.run(n_steps, forcing)
                        final_R = float(np.mean(R[-200:]))
                        if q_name == "Shera":
                            shera_finals.append(final_R)
                        else:
                            uniform_finals.append(final_R)

                delta = np.mean(shera_finals) - np.mean(uniform_finals)
                results[cell_key] = {
                    "spread": s_name, "K": k_name, "F": f_name,
                    "shera_mean": np.mean(shera_finals),
                    "uniform_mean": np.mean(uniform_finals),
                    "delta": delta,
                }
                n_total += 1
                if delta > 0:
                    n_positive += 1

                print(f"  {cell_key}: Shera={np.mean(shera_finals):.4f} "
                      f"Uniform={np.mean(uniform_finals):.4f} "
                      f"ΔR={delta:+.4f} {'✓' if delta > 0 else '✗'}")

    pct = n_positive / n_total * 100
    print(f"\n  Summary: Shera advantage in {n_positive}/{n_total} cells ({pct:.0f}%)")
    if pct >= 75:
        print(f"  ✓ ROBUST: Shera Q advantage persists across parameter regimes")
    elif pct >= 50:
        print(f"  ~ PARTIAL: Shera Q advantage present in majority of regimes")
    else:
        print(f"  ✗ FRAGILE: Shera Q advantage is regime-dependent")

    # Heatmap visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (s_name, spread) in enumerate(spreads.items()):
        ax = axes[idx]
        grid = np.zeros((3, 3))
        k_names = list(Ks.keys())
        f_names = list(Fs.keys())
        for ki, k_name in enumerate(k_names):
            for fi, f_name in enumerate(f_names):
                cell_key = f"{s_name}_K{k_name}_F{f_name}"
                grid[ki, fi] = results[cell_key]["delta"]

        im = ax.imshow(grid, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"F={v}" for v in Fs.values()], fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"K={v}" for v in Ks.values()], fontsize=8)
        ax.set_title(f"ω spread: ±{spread}", fontsize=10)

        for ki in range(3):
            for fi in range(3):
                ax.text(fi, ki, f"{grid[ki, fi]:+.3f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(grid[ki, fi]) > 0.15 else 'black')

    fig.suptitle("Shera Q Advantage (ΔR) Across Parameter Regimes", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes, label="ΔR (Shera - Uniform)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp1b_parameter_robustness.png", dpi=150)
    plt.close()
    print(f"\n  → Plot saved: {OUT_DIR}/exp1b_parameter_robustness.png")

    return results


def experiment_9b_bootstrap_ci():
    """
    Kimi/GPT Audit Response: Bootstrap 95% confidence intervals for E9 results.

    Runs E9 and adds BCa bootstrap CIs for each condition's mean recovery
    and for the mean difference (Self-Dual - each control).
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 9b: Bootstrap Confidence Intervals for Self-Duality")
    print("=" * 72)

    # Run E9 to get results
    from coherence_unit_cell_experiments import experiment_9_self_duality_v2
    r9 = experiment_9_self_duality_v2()

    try:
        from scipy.stats import bootstrap
        HAS_BOOTSTRAP = True
    except ImportError:
        print("  scipy.stats.bootstrap not available (requires scipy >= 1.7)")
        print("  Falling back to percentile bootstrap")
        HAS_BOOTSTRAP = False

    sd_data = np.array(r9["Self-Dual"]["all"])

    print(f"\n  Bootstrap 95% CIs (10,000 resamples):")

    for cond_name in ["Self-Dual", "Orthogonal", "Trained Decoder", "Random Projection"]:
        data = np.array(r9[cond_name]["all"])

        if HAS_BOOTSTRAP:
            res = bootstrap((data,), np.mean, n_resamples=10000,
                            confidence_level=0.95, method='BCa',
                            random_state=SEED)
            ci_low, ci_high = res.confidence_interval
        else:
            # Percentile bootstrap fallback
            np.random.seed(SEED)
            boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True))
                          for _ in range(10000)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        print(f"  {cond_name}: mean={np.mean(data):.4f}, "
              f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    # Difference CIs
    print(f"\n  Mean Difference CIs (Self-Dual - Control):")
    for cond_name in ["Orthogonal", "Trained Decoder", "Random Projection"]:
        ctrl_data = np.array(r9[cond_name]["all"])
        diffs = sd_data - ctrl_data[:len(sd_data)]  # pairwise if same length

        if HAS_BOOTSTRAP:
            res = bootstrap((diffs,), np.mean, n_resamples=10000,
                            confidence_level=0.95, method='BCa',
                            random_state=SEED)
            ci_low, ci_high = res.confidence_interval
        else:
            np.random.seed(SEED)
            boot_diffs = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True))
                          for _ in range(10000)]
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

        print(f"  Self-Dual - {cond_name}: "
              f"Δ={np.mean(diffs):.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        if ci_low > 0:
            print(f"    ✓ CI excludes zero — advantage is statistically robust")
        else:
            print(f"    ~ CI includes zero — interpret with caution")


if __name__ == "__main__":
    start = time.time()

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  SUPPLEMENTARY VALIDATION — Reviewer Response                      ║")
    print("╚" + "═" * 70 + "╝")

    r1b = experiment_1b_parameter_robustness()
    experiment_9b_bootstrap_ci()

    elapsed = time.time() - start
    print(f"\n  Supplementary runtime: {elapsed:.1f} seconds")
    print(f"  {'='*60}")
