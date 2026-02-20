# Coherence Unit Cell — Experimental Validation

**Cross-domain convergence in Kuramoto coupled oscillator architectures**

Four independent research programs (2012–2026) converged on structurally similar coupled-oscillator architectures despite no shared literature or collaboration:

| System | Year | Domain | Key Reference |
|--------|------|--------|---------------|
| Bell | 2012 | Cochlear biophysics | Hearing Research |
| Miyato et al. | 2025 | Neural networks | AKOrN (ICLR Oral) |
| Hays | 2025 | Transformer attention | SSA |
| Waltman | 2025–2026 | Coherence substrate | FlowerTuner51 |

This repository contains 13 reproducible experiments testing whether that convergence reflects a functionally privileged architecture class or a coincidental similarity.

## Key Results

1. **Biological tuning transfers across domains.** Cochlear Shera Q-factors improve coherence in non-biological Kuramoto networks by ~+0.24 over uniform coupling, robust across 27 parameter regimes (100%).

2. **Self-duality confers robustness.** Systems where the settled phase state is the representation recover identity after perturbation ~9× better than orthogonal, trained, or random readout transforms (all p < 0.0001, Cohen's d > 1.0, bootstrap 95% CIs excluding zero).

3. **The Coherence Unit Cell is load-bearing.** Ablating individual properties systematically degrades coherence and stability relative to the full six-property bundle.

## Quick Start

```bash
# Weekend replication (~2 min, E1 + E9 only)
python3 quick_replication.py

# Full suite (13 experiments, ~5 min)
python3 coherence_unit_cell_experiments.py

# Supplementary robustness (E1b parameter sweep + E9b bootstrap CIs)
python3 supplementary_validation.py
```

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scipy
- jax (optional, for Experiment 13)

```bash
pip install numpy matplotlib scipy
```

## Files

| File | Description |
|------|-------------|
| `coherence_unit_cell_experiments.py` | Full 13-experiment suite (V1 + V2) |
| `supplementary_validation.py` | Parameter robustness sweep + bootstrap CIs |
| `quick_replication.py` | Minimal E1 + E9, self-contained, ~2 min |
| `CUC_Final_V2.docx` | Full experimental document |
| `figures/` | Generated plots (exp1–exp12) |

## Reproducibility

All experiments use deterministic seeding:

- **Seed:** 42
- **Integration:** 4th-order Runge-Kutta
- **Runtime:** ~5 minutes on standard hardware

Results are exactly reproducible across platforms.

## Author

Matt Waltman & Aurora
Richter Farms, Grundy County, Iowa
mwaltman@richterfarms.com

## License

MIT
