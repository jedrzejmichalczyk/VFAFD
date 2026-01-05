# Hybrid AFD-VF for Rational Approximation

This repository contains the implementation and manuscript for a novel hybrid algorithm combining Adaptive Fourier Decomposition (AFD) with Vector Fitting (VF) for rational approximation in Hardy spaces.

## Overview

The hybrid method eliminates the need for ill-conditioned least squares solutions in classical Vector Fitting by using AFD's greedy maximization framework with cyclic refinement.

### Key Contributions

1. **No Least Squares**: Replaces VF's matrix inversions with scalar maximization
2. **Orthogonal Basis**: Uses Blaschke products to maintain orthogonality
3. **Cyclic Refinement**: Iteratively improves pole locations using VF principles
4. **Decoupling Theorem**: Proves that global optimization separates into independent single-pole problems

## Files

- `paper.tex` - Main manuscript (LaTeX)
- `afd_vf_hybrid.py` - Python implementation of the algorithms
- `experiment1.png` - Results for rational function test
- `experiment2.png` - Results for smooth function test

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

## Running Experiments

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run experiments
python3 afd_vf_hybrid.py
```

This will:
- Validate test functions for H²(D) membership (poles must be outside unit disk!)
- Run three numerical experiments
- Generate comparison plots (experiment1.png, experiment2.png, experiment3.png)
- Print comprehensive performance metrics and summary table

## Results Summary

**IMPORTANT:** Performance is highly dependent on function class!

| Experiment | Pure AFD Error | Hybrid Error | Result |
|------------|----------------|--------------|---------|
| Rational Function (poles outside D) | 82.0% | **24.5%** | ✅ **70% improvement** |
| Smooth Polynomial | 34.2% | 27.1% | ✅ 21% improvement |
| Bandlimited Function | 12.5% | 15.8% | ❌ **27% degradation** |

### Key Findings

1. **When Hybrid Excels**: Rational functions with well-separated poles → 70% error reduction
2. **When Hybrid Helps Moderately**: Smooth functions with moderate pole separation → 21% error reduction
3. **When Hybrid Hurts**: Very smooth/bandlimited functions → 27% error increase!

**Recommendation**: Use hybrid refinement only for functions with known distinct pole structure (e.g., microwave filters). For smooth approximation, stick with pure AFD.

## Algorithm Complexity

- **Classical VF**: O(Ln² + n³) per iteration (requires solving linear system)
- **Hybrid AFD-VF**: O(n·T_opt·L) per refinement cycle (only maximization)

Where:
- L = number of sample points
- n = number of poles
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

- Extension to matrix-valued functions (MIMO systems)
- Application to microwave filter coupling matrix extraction
- Convergence rate analysis for different function classes
- GPU acceleration for large-scale problems
