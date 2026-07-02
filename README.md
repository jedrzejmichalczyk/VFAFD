# Hybrid AFD-VF for Rational Approximation and MIMO Filter Identification

[![Compile LaTeX](https://github.com/jedrzejmichalczyk/VFAFD/actions/workflows/compile-latex.yml/badge.svg)](https://github.com/jedrzejmichalczyk/VFAFD/actions/workflows/compile-latex.yml)

This repository contains the implementation and manuscript for a novel hybrid algorithm combining Adaptive Fourier Decomposition (AFD) with Vector Fitting (VF) for rational approximation in Hardy spaces, extended to **matrix-valued (MIMO) responses** for microwave filter identification.

**📄 [Download Latest Paper PDF](https://github.com/jedrzejmichalczyk/VFAFD/actions/workflows/compile-latex.yml)** (Click on latest run → Artifacts → paper-pdf)

## Overview

The hybrid method eliminates the need for ill-conditioned least squares solutions in classical Vector Fitting by using AFD's greedy maximization framework with cyclic refinement.

### Key Contributions

1. **No Least Squares**: Replaces VF's matrix inversions with scalar maximization
2. **Orthogonal Basis**: Uses Blaschke products to maintain orthogonality
3. **Cyclic Refinement**: Iteratively improves pole locations using VF principles
4. **Decoupling Theorem**: Proves that global optimization separates into independent single-pole problems
5. **MIMO extension (new)**: One *common* pole set shared by all S-parameter entries with matrix-valued coefficients — the shared-pole structure of MIMO vector fitting inside the orthogonal AFD framework
6. **Energy-monotone relocation (new)**: A model-space reformulation of the cyclic refinement that provably **never increases** the error, fixing the degradation naive refinement shows on smooth functions
7. **Direct term (new)**: VF's feedthrough constant `d` reproduced by pinning one Takenaka–Malmquist parameter at `a = 0` (necessary and sufficient for responses with `S(∞) ≠ 0`)

## Files

- `paper.tex` - Main manuscript (LaTeX)
- `afd_vf_hybrid.py` - Scalar hybrid AFD-VF implementation
- `mimo_afd_vf.py` - **MIMO hybrid AFD-VF** for microwave filter identification (common poles, monotone relocation, Cayley transform, coupling-matrix test models)
- `experiment1.png` … `experiment3.png` - Scalar experiments
- `mimo_experiment1.png` … `mimo_experiment3.png` - MIMO filter identification experiments

## Implementation

The Python code implements three components:

### 1. Blaschke Product Operations
```python
BlaschkeProduct.kernel(z, a)          # e_a(z) = √(1-|a|²)/(1-z*conj(a))
BlaschkeProduct.blaschke(z, a)        # b_a(z) = (z-a)/(1-z*conj(a))
BlaschkeProduct.basis_function(z, poles, k)  # B_k(z)
```

### 2. Pure AFD Algorithm
```python
afd = AdaptiveFourierDecomposition(n_poles=5)
poles, residues = afd.fit(F, z_samples)
F_approx = afd.evaluate(z_test)
```

### 3. Hybrid AFD-VF Algorithm
```python
hybrid = HybridAFDVF(n_poles=5, max_refine_iter=5)
poles, residues = hybrid.fit(F, z_samples)
F_approx = hybrid.evaluate(z_test)
```

### 4. MIMO Hybrid AFD-VF (mimo_afd_vf.py)
```python
from mimo_afd_vf import MimoAFDVF, sample_filter_on_circle, inline_chebyshev_coupling

M, R1, RN = inline_chebyshev_coupling(N=4, ripple_db=0.0432)   # coupling matrix
z, lam, S = sample_filter_on_circle(M, R1, RN, n_samples=600)  # 2x2 S-matrix samples

model = MimoAFDVF(n_poles=5, refine_cycles=30, fixed_poles=[0.0])  # a=0 = direct term
model.fit(z, S)               # works directly on samples (Szegő-kernel quadrature)
S_hat = model.evaluate(z)     # common-pole rational model
poles = model.lambda_poles()  # poles mapped back to the prototype frequency plane
```

## Running Experiments

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Scalar experiments (experiment1-3.png)
python3 afd_vf_hybrid.py

# MIMO filter identification experiments (mimo_experiment1-3.png)
python3 mimo_afd_vf.py
```

## Results Summary

### Scalar experiments (naive refinement)

**IMPORTANT:** Performance is highly dependent on function class!

| Experiment | Pure AFD Error | Hybrid Error | Result |
|------------|----------------|--------------|---------|
| Rational Function (poles outside D) | 82.0% | **24.5%** | ✅ **70% improvement** |
| Smooth Polynomial | 34.2% | 27.1% | ✅ 21% improvement |
| Bandlimited Function | 12.5% | 15.8% | ❌ **27% degradation** |

The degradation in the last row is exactly what the **energy-monotone relocation** of the MIMO module eliminates — with the model-space formulation, refinement can never do worse than the greedy initialization.

### MIMO filter identification (energy-monotone refinement)

| Experiment | Greedy matrix AFD | Hybrid MIMO AFD-VF | Pole recovery (hybrid) |
|------------|-------------------|--------------------|------------------------|
| M1: inline Chebyshev (N=4, RL=20 dB) | 1.5e-01 | **4.6e-09** | 1.6e-08 |
| M2: cross-coupled, 2 transmission zeros | 1.6e-01 | **8.9e-09** | 2.8e-08 |
| M3: cross-coupled @ -60 dB noise | 1.6e-01 | **1.6e-04** (below noise floor) | 3.6e-04 |

- Identification at **coupling-matrix-extraction accuracy** without solving a single linear system
- Common poles across all ports: 33% fewer parameters than entrywise fitting, better noise robustness, and one consistent pole set
- Refinement error is **monotonically decreasing by construction** (proved in the paper, observed in every run)

## Algorithm Complexity

- **Classical VF**: O(Ln² + n³) per iteration (requires solving linear system)
- **Hybrid AFD-VF**: O(n·T_opt·L) per refinement cycle (only maximization)
- **MIMO Hybrid AFD-VF**: O(n·(n + N_cand)·L·pq) per relocation sweep, no linear systems

Where:
- L = number of sample points
- n = number of poles
- p×q = response matrix size, N_cand = candidate grid size
- T_opt = optimization cost ≈ O(log ε⁻¹)

## Compiling the Paper

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Note: Requires LaTeX packages: `IEEEtran`, `algorithm`, `algpseudocode`, `amsmath`, `braket`

## Citation

```bibtex
@article{michalczyk2026hybrid,
  title={Hybrid AFD-VF Algorithm for Rational Approximation},
  author={Michalczyk, Jedrzej and Michalski, Jerzy Julian},
  journal={},
  year={2026}
}
```

## Future Work

- Full coupling matrix reconstruction from the identified common poles and matrix residues
- Enforcing conjugate-symmetric pole pairs and passivity during relocation
- Convergence rate analysis of the coordinate-ascent relocation
- Higher-order multiplexers (p, q > 2) and GPU acceleration for large-scale problems
