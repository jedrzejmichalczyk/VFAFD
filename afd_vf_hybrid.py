"""
Hybrid AFD-VF Algorithm for Rational Approximation
Combines Adaptive Fourier Decomposition with Vector Fitting
"""

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
import time


class BlaschkeProduct:
    """Represents Blaschke products for AFD"""

    @staticmethod
    def kernel(z: np.ndarray, a: complex) -> np.ndarray:
        """Normalized reproducing kernel e_a(z) = √(1-|a|²)/(1-z*conj(a))"""
        return np.sqrt(1 - np.abs(a)**2) / (1 - z * np.conj(a))

    @staticmethod
    def blaschke(z: np.ndarray, a: complex) -> np.ndarray:
        """Elementary Blaschke factor b_a(z) = (z-a)/(1-z*conj(a))"""
        return (z - a) / (1 - z * np.conj(a))

    @staticmethod
    def basis_function(z: np.ndarray, poles: List[complex], k: int) -> np.ndarray:
        """B_k(z) = (∏_{u=0}^{k-1} b_{a_u}(z)) * e_{a_k}(z)"""
        if k == 0:
            return BlaschkeProduct.kernel(z, poles[0])

        # Product of Blaschke factors
        result = np.ones_like(z, dtype=complex)
        for u in range(k):
            result *= BlaschkeProduct.blaschke(z, poles[u])

        # Multiply by kernel
        result *= BlaschkeProduct.kernel(z, poles[k])
        return result


class AdaptiveFourierDecomposition:
    """Pure AFD implementation"""

    def __init__(self, n_poles: int, max_iter: int = 100):
        self.n_poles = n_poles
        self.max_iter = max_iter
        self.poles = []
        self.residues = []

    def inner_product(self, f1: np.ndarray, f2: np.ndarray, z_samples: np.ndarray) -> complex:
        """Approximate H² inner product ⟨f1, f2⟩"""
        # Use trapezoidal rule on the unit circle
        return np.trapz(f1 * np.conj(f2), z_samples) / (2j * np.pi)

    def find_optimal_pole(self, F_k: Callable, z_samples: np.ndarray,
                         search_radius: float = 0.95) -> Tuple[complex, float]:
        """Find a_k = argmax |⟨F_k, e_a⟩| = argmax √(1-|a|²)|F_k(a)|"""

        def objective(x):
            """Negative of |√(1-|a|²)F_k(a)| for minimization"""
            a = x[0] + 1j*x[1]
            if np.abs(a) >= 1.0:
                return 1e10  # Penalty for points outside unit disk

            fa = F_k(a)
            val = np.sqrt(1 - np.abs(a)**2) * np.abs(fa)
            return -val  # Negative for minimization

        # Use differential evolution for global optimization
        bounds = [(-search_radius, search_radius), (-search_radius, search_radius)]
        result = differential_evolution(objective, bounds, maxiter=100,
                                       seed=42, workers=1, atol=1e-6)

        optimal_a = result.x[0] + 1j*result.x[1]
        max_value = -result.fun

        return optimal_a, max_value

    def fit(self, F: Callable, z_samples: np.ndarray) -> Tuple[List[complex], List[complex]]:
        """
        Perform AFD decomposition
        F(z) ≈ Σ_{k=0}^{n-1} B_k(z)M_k
        """
        self.poles = []
        self.residues = []

        # Current residual function
        F_values = F(z_samples)

        def F_k(z):
            """Compute current residual at point z"""
            # Reconstruct from current approximation
            approx = np.zeros_like(z, dtype=complex)
            for i, (pole, residue) in enumerate(zip(self.poles, self.residues)):
                B_i = BlaschkeProduct.basis_function(z, self.poles, i)
                approx += B_i * residue
            return F(z) - approx

        for k in range(self.n_poles):
            print(f"  AFD iteration {k+1}/{self.n_poles}")

            # Find optimal pole
            a_k, max_proj = self.find_optimal_pole(F_k, z_samples)

            # Compute residue M_k = √(1-|a_k|²) F_k(a_k)
            M_k = np.sqrt(1 - np.abs(a_k)**2) * F_k(a_k)

            self.poles.append(a_k)
            self.residues.append(M_k)

            print(f"    Pole: {a_k:.6f}, Residue: {M_k:.6f}, Max projection: {max_proj:.6e}")

        return self.poles, self.residues

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """Evaluate approximation at points z"""
        result = np.zeros_like(z, dtype=complex)
        for k, (pole, residue) in enumerate(zip(self.poles, self.residues)):
            B_k = BlaschkeProduct.basis_function(z, self.poles[:k+1], k)
            result += B_k * residue
        return result


class HybridAFDVF:
    """Hybrid AFD-VF with cyclic refinement (no least squares)"""

    def __init__(self, n_poles: int, max_refine_iter: int = 5):
        self.n_poles = n_poles
        self.max_refine_iter = max_refine_iter
        self.poles = []
        self.residues = []

    def find_optimal_pole(self, F_residual: Callable, search_radius: float = 0.95) -> complex:
        """Find a = argmax |√(1-|a|²)F(a)| without least squares"""

        def objective(x):
            a = x[0] + 1j*x[1]
            if np.abs(a) >= 1.0:
                return 1e10

            fa = F_residual(a)
            val = np.sqrt(1 - np.abs(a)**2) * np.abs(fa)
            return -val

        bounds = [(-search_radius, search_radius), (-search_radius, search_radius)]
        result = differential_evolution(objective, bounds, maxiter=100,
                                       seed=42, workers=1, atol=1e-6)

        return result.x[0] + 1j*result.x[1]

    def fit(self, F: Callable, z_samples: np.ndarray) -> Tuple[List[complex], List[complex]]:
        """
        Hybrid algorithm:
        1. Initial AFD pass
        2. Cyclic refinement of poles (VF-inspired but using AFD maximization)
        """

        print("Phase 1: Initial AFD decomposition")
        afd = AdaptiveFourierDecomposition(self.n_poles)
        self.poles, self.residues = afd.fit(F, z_samples)

        print(f"\nPhase 2: Cyclic refinement ({self.max_refine_iter} iterations)")

        for iter_num in range(self.max_refine_iter):
            print(f"  Refinement iteration {iter_num+1}/{self.max_refine_iter}")
            max_change = 0.0

            # Cycle through each pole
            for j in range(self.n_poles):
                # Compute partial residual F̃_j = F - Σ_{k≠j} B_k M_k
                def F_partial(z):
                    approx = np.zeros_like(z, dtype=complex)
                    for k in range(self.n_poles):
                        if k != j:
                            B_k = BlaschkeProduct.basis_function(z, self.poles, k)
                            approx += B_k * self.residues[k]
                    return F(z) - approx

                # Re-optimize pole j
                old_pole = self.poles[j]
                new_pole = self.find_optimal_pole(F_partial)

                # Update pole
                self.poles[j] = new_pole

                # Recompute residue
                M_j = np.sqrt(1 - np.abs(new_pole)**2) * F_partial(new_pole)
                self.residues[j] = M_j

                # Track convergence
                pole_change = np.abs(new_pole - old_pole)
                max_change = max(max_change, pole_change)

            print(f"    Max pole change: {max_change:.6e}")

            if max_change < 1e-6:
                print(f"    Converged!")
                break

        return self.poles, self.residues

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """Evaluate approximation"""
        result = np.zeros_like(z, dtype=complex)
        for k, (pole, residue) in enumerate(zip(self.poles, self.residues)):
            B_k = BlaschkeProduct.basis_function(z, self.poles[:k+1], k)
            result += B_k * residue
        return result


def test_function_1(z: np.ndarray) -> np.ndarray:
    """
    Test function: Rational function with poles OUTSIDE unit disk
    This is a proper H²(D) function.
    Poles: 1.5+0.5j (|z|=1.58), -1.2+0.8j (|z|=1.44), 2.0-1.0j (|z|=2.24)
    """
    poles = [1.5 + 0.5j, -1.2 + 0.8j, 2.0 - 1.0j]
    residues = [0.5 + 0.2j, -0.3 + 0.4j, 0.4 - 0.1j]

    result = np.zeros_like(z, dtype=complex)
    for pole, res in zip(poles, residues):
        result += res / (z - pole)
    return result


def test_function_2(z: np.ndarray) -> np.ndarray:
    """
    Test function: Smooth function in H² with poles outside D
    1 + 0.5z + 0.3z² = 0 has roots at z ≈ -0.83 ± 1.63i with |z| ≈ 1.83 > 1
    """
    return 1 / (1 + 0.5*z + 0.3*z**2)


def test_function_3(z: np.ndarray) -> np.ndarray:
    """
    Test function: Bandlimited-like function (very smooth in H²)
    All singularities at infinity, exponentially decaying Taylor series
    """
    return np.exp(-0.3*z) * np.cos(0.5*z)


def run_comparison_experiment(F: Callable, n_poles: int,
                              experiment_name: str) -> dict:
    """Run comparison between AFD and Hybrid methods"""

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")

    # Sample points on unit circle for evaluation
    n_samples = 500
    theta = np.linspace(0, 2*np.pi, n_samples)
    z_samples = 0.95 * np.exp(1j * theta)  # Slightly inside for stability

    # True function values
    F_true = F(z_samples)

    results = {}

    # Test 1: Pure AFD
    print("\n--- Running Pure AFD ---")
    t0 = time.time()
    afd = AdaptiveFourierDecomposition(n_poles)
    afd.fit(F, z_samples)
    F_afd = afd.evaluate(z_samples)
    time_afd = time.time() - t0
    error_afd = np.linalg.norm(F_true - F_afd) / np.linalg.norm(F_true)

    results['AFD'] = {
        'time': time_afd,
        'error': error_afd,
        'poles': afd.poles.copy(),
        'residues': afd.residues.copy(),
        'approximation': F_afd
    }

    print(f"AFD - Time: {time_afd:.3f}s, Relative Error: {error_afd:.6e}")

    # Test 2: Hybrid AFD-VF
    print("\n--- Running Hybrid AFD-VF ---")
    t0 = time.time()
    hybrid = HybridAFDVF(n_poles, max_refine_iter=5)
    hybrid.fit(F, z_samples)
    F_hybrid = hybrid.evaluate(z_samples)
    time_hybrid = time.time() - t0
    error_hybrid = np.linalg.norm(F_true - F_hybrid) / np.linalg.norm(F_true)

    results['Hybrid'] = {
        'time': time_hybrid,
        'error': error_hybrid,
        'poles': hybrid.poles.copy(),
        'residues': hybrid.residues.copy(),
        'approximation': F_hybrid
    }

    print(f"Hybrid - Time: {time_hybrid:.3f}s, Relative Error: {error_hybrid:.6e}")

    # Summary
    improvement = (error_afd - error_hybrid) / error_afd * 100
    print(f"\nImprovement: {improvement:.2f}% error reduction")

    results['z_samples'] = z_samples
    results['F_true'] = F_true

    return results


def plot_results(results: dict, experiment_name: str, filename: str):
    """Create comparison plots"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    z_samples = results['z_samples']
    F_true = results['F_true']

    # Plot 1: Magnitude comparison
    ax = axes[0, 0]
    ax.plot(np.angle(z_samples), np.abs(F_true), 'k-', linewidth=2, label='True')
    ax.plot(np.angle(z_samples), np.abs(results['AFD']['approximation']),
            'b--', linewidth=1.5, label='AFD')
    ax.plot(np.angle(z_samples), np.abs(results['Hybrid']['approximation']),
            'r:', linewidth=1.5, label='Hybrid')
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    ax = axes[0, 1]
    error_afd = np.abs(F_true - results['AFD']['approximation'])
    error_hybrid = np.abs(F_true - results['Hybrid']['approximation'])
    ax.semilogy(np.angle(z_samples), error_afd, 'b-', label='AFD Error')
    ax.semilogy(np.angle(z_samples), error_hybrid, 'r-', label='Hybrid Error')
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Pointwise Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Pole locations
    ax = axes[1, 0]
    circle = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--', linewidth=1)
    ax.add_patch(circle)

    poles_afd = np.array(results['AFD']['poles'])
    poles_hybrid = np.array(results['Hybrid']['poles'])

    ax.plot(poles_afd.real, poles_afd.imag, 'bo', markersize=8, label='AFD poles')
    ax.plot(poles_hybrid.real, poles_hybrid.imag, 'r^', markersize=8, label='Hybrid poles')

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Pole Locations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    # Plot 4: Comparison table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Method', 'Rel. Error', 'Time (s)'],
        ['AFD', f"{results['AFD']['error']:.2e}", f"{results['AFD']['time']:.3f}"],
        ['Hybrid', f"{results['Hybrid']['error']:.2e}", f"{results['Hybrid']['time']:.3f}"]
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(f'{experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")

    return fig


def validate_h2_function(F: Callable, name: str):
    """Check if function appears to be in H²(D) by testing for singularities inside disk"""
    print(f"\nValidating '{name}' for H²(D) membership...")

    # Test points inside unit disk
    test_points = [0, 0.3+0.4j, -0.5+0.2j, 0.7-0.3j, 0.6j, -0.8]

    for z in test_points:
        try:
            val = F(np.array([z]))[0]
            if np.isnan(val) or np.isinf(val) or np.abs(val) > 1e10:
                print(f"  WARNING: Singularity or large value at z={z}: F(z)={val}")
                print(f"  This function may NOT be in H²(D)!")
                return False
        except:
            print(f"  ERROR: Exception at z={z}")
            return False

    print(f"  ✓ Function appears valid for H²(D)")
    return True


if __name__ == "__main__":
    print("="*70)
    print("HYBRID AFD-VF ALGORITHM - NUMERICAL EXPERIMENTS")
    print("="*70)

    # Validate test functions
    validate_h2_function(test_function_1, "Rational Function")
    validate_h2_function(test_function_2, "Smooth Polynomial")
    validate_h2_function(test_function_3, "Bandlimited Function")

    # Experiment 1: Rational function with poles outside D
    results1 = run_comparison_experiment(
        test_function_1,
        n_poles=5,
        experiment_name="Rational Function (Poles Outside D)"
    )
    plot_results(results1, "Experiment 1: Rational Function", "experiment1.png")

    # Experiment 2: Smooth function
    results2 = run_comparison_experiment(
        test_function_2,
        n_poles=4,
        experiment_name="Smooth Polynomial Function"
    )
    plot_results(results2, "Experiment 2: Smooth Function", "experiment2.png")

    # Experiment 3: Very smooth function
    results3 = run_comparison_experiment(
        test_function_3,
        n_poles=6,
        experiment_name="Bandlimited Function"
    )
    plot_results(results3, "Experiment 3: Bandlimited Function", "experiment3.png")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*70)
    print(f"{'Experiment':<30} {'AFD Error':<15} {'Hybrid Error':<15} {'Improvement':>10}")
    print("-"*70)

    for i, (name, results) in enumerate([
        ("Rational Function", results1),
        ("Smooth Polynomial", results2),
        ("Bandlimited Function", results3)
    ], 1):
        afd_err = results['AFD']['error']
        hyb_err = results['Hybrid']['error']
        improv = (afd_err - hyb_err) / afd_err * 100
        print(f"{name:<30} {afd_err:<15.3e} {hyb_err:<15.3e} {improv:>9.1f}%")

    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)
