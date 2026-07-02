"""
MIMO Hybrid AFD-VF Algorithm for Microwave Filter Identification
================================================================

Extends the scalar hybrid AFD-VF idea (afd_vf_hybrid.py) to matrix-valued
functions, targeting identification of multiport microwave filters from
sampled S-parameter data.

Method summary
--------------
A p x q matrix response F(z) analytic in the unit disk is expanded in a
single scalar Takenaka-Malmquist (TM) basis with MATRIX coefficients,

    F(z) ~= sum_k  M_k B_k(z),      B_k(z) = e_{a_k}(z) prod_{u<k} b_{a_u}(z),

so all matrix entries share one common pole set (the defining property of
MIMO vector fitting).  The algorithm is least-squares free:

1. Greedy matrix AFD: each pole maximizes the Frobenius-norm energy
   extracted from the reduced remainder,  a_k = argmax (1-|a|^2) ||G_k(a)||_F^2.
2. Energy-monotone cyclic relocation (the VF ingredient): for each pole,
   project F onto the TM space of the REMAINING poles, divide the residual
   by their Blaschke product, and re-maximize.  Because the previous pole is
   kept as a fallback candidate, every step is guaranteed not to decrease
   the captured energy, i.e. the approximation error is non-increasing.
   This repairs the failure mode of the naive scalar refinement, which
   could degrade the fit for smooth functions.

Everything operates on SAMPLED data on the unit circle: interior values
are recovered with Szego-kernel quadrature, so the method applies directly
to measured frequency sweeps mapped to the circle by a Cayley transform.

Test vehicles are coupled-resonator (coupling matrix) filter models whose
true poles are known eigenvalues, enabling quantitative pole-recovery
checks.

Run:  python3 mimo_afd_vf.py
"""

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CVD-safe categorical palette (validated), near-black reserved for the
# reference ("true") curve which is not a categorical series.
C_TRUE = "#222222"
C_BLUE = "#0072B2"   # greedy MIMO AFD
C_VERM = "#D55E00"   # hybrid MIMO AFD-VF
C_GREEN = "#009E73"  # entrywise scalar baseline
C_MAG = "#B84A83"    # auxiliary

EPS = 1e-16


# ---------------------------------------------------------------------------
# Takenaka-Malmquist primitives
# ---------------------------------------------------------------------------

def kernel_e(z: np.ndarray, a: complex) -> np.ndarray:
    """Normalized Szego kernel e_a(z) = sqrt(1-|a|^2)/(1 - z conj(a))."""
    return np.sqrt(1.0 - np.abs(a) ** 2) / (1.0 - z * np.conj(a))


def blaschke_b(z: np.ndarray, a: complex) -> np.ndarray:
    """Elementary Blaschke factor b_a(z) = (z-a)/(1 - z conj(a))."""
    return (z - a) / (1.0 - z * np.conj(a))


def uniform_circle_grid(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoint-uniform grid on the unit circle (avoids z = 1)."""
    theta = 2.0 * np.pi * (np.arange(n_samples) + 0.5) / n_samples
    return np.exp(1j * theta), theta


# ---------------------------------------------------------------------------
# Cayley transform between the s-domain (lowpass prototype) and the disk
# ---------------------------------------------------------------------------
# z = (s-1)/(s+1) maps the open right half plane onto the unit disk and the
# imaginary axis s = j*lambda onto the unit circle.  Poles of a stable
# (passive) response lie in Re s < 0 and therefore map OUTSIDE the disk,
# so S(z(s)) is analytic in the disk as required by AFD.

def lambda_from_theta(theta: np.ndarray) -> np.ndarray:
    """Prototype frequency lambda corresponding to z = exp(j theta)."""
    return 1.0 / np.tan(theta / 2.0)


def tm_param_to_lambda_pole(a: complex) -> complex:
    """Map a TM parameter a in D to the prototype-frequency pole location.

    The TM basis function e_a has its pole at z = 1/conj(a); pulling that
    back through the Cayley transform gives s, and lambda = -j s.
    """
    zeta = 1.0 / np.conj(a)
    s = (1.0 + zeta) / (1.0 - zeta)
    return -1j * s


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

class MimoAFDVF:
    """Common-pole matrix rational approximation on sampled circle data.

    Parameters
    ----------
    n_poles        : number of TM basis functions (poles) to use, INCLUDING
                     any fixed poles
    refine_cycles  : maximum number of cyclic relocation sweeps (0 = pure
                     greedy matrix AFD)
    fixed_poles    : TM parameters pinned during greedy selection and
                     relocation.  Pass [0.0] to include a constant basis
                     function: this is the AFD analogue of the direct
                     (feedthrough) term d in vector fitting and is REQUIRED
                     for exact representation of responses with S(inf) != 0,
                     such as S-parameters.
    grid_radii     : coarse-search radii count
    grid_angles    : coarse-search angle count
    rtol           : stop refinement when the relative error improves by
                     less than this between sweeps
    """

    def __init__(self, n_poles: int, refine_cycles: int = 10,
                 fixed_poles: Optional[Sequence[complex]] = None,
                 grid_radii: int = 24, grid_angles: int = 64,
                 rtol: float = 1e-9, r_max: float = 0.995,
                 verbose: bool = True):
        self.n_poles = n_poles
        self.fixed_poles = list(fixed_poles or [])
        if len(self.fixed_poles) >= n_poles:
            raise ValueError("need at least one free pole")
        self.refine_cycles = refine_cycles
        self.rtol = rtol
        self.r_max = r_max
        self.verbose = verbose
        # Coarse candidate grid, denser toward the circle where filter
        # poles live after the Cayley map.
        radii = 1.0 - np.geomspace(1e-3, 1.0, grid_radii)
        angles = 2.0 * np.pi * np.arange(grid_angles) / grid_angles
        self._candidates = (radii[:, None] * np.exp(1j * angles[None, :])).ravel()

        self.poles: List[complex] = []
        self.coeffs: List[np.ndarray] = []
        self.error_history: List[float] = []

    # -- sampled-data inner products -------------------------------------

    def _interior_value(self, G: np.ndarray, a: complex) -> np.ndarray:
        """Evaluate the H^2 function with samples G at the interior point a.

        Uses the reproducing property f(a) = <f, K_a> discretized with the
        uniform trapezoidal rule (spectrally accurate on the circle).
        """
        w = 1.0 / (1.0 - a * np.conj(self._z))
        return np.einsum('l,l...->...', w, G) / len(self._z)

    def _gain(self, G: np.ndarray, a: complex) -> float:
        """Energy extracted by adding pole a: (1-|a|^2) ||G(a)||_F^2."""
        val = self._interior_value(G, a)
        return (1.0 - np.abs(a) ** 2) * float(np.sum(np.abs(val) ** 2))

    # -- pole search (scalar maximization, no linear algebra) ------------

    def _find_pole(self, G: np.ndarray,
                   extra_starts: Sequence[complex] = ()) -> complex:
        L = len(self._z)
        A = self._candidates
        # Vectorized coarse evaluation of the gain on the whole grid.
        W = 1.0 / (1.0 - np.outer(A, np.conj(self._z)))
        V = W @ G.reshape(L, -1) / L
        gains = (1.0 - np.abs(A) ** 2) * np.sum(np.abs(V) ** 2, axis=1)
        order = np.argsort(gains)[::-1]
        starts = [A[i] for i in order[:3]] + list(extra_starts)

        def neg_gain(x):
            a = x[0] + 1j * x[1]
            if np.abs(a) >= self.r_max:
                return 1e6
            return -self._gain(G, a)

        best_a, best_g = None, -np.inf
        for s0 in starts:
            res = minimize(neg_gain, x0=[s0.real, s0.imag],
                           method="Nelder-Mead",
                           options=dict(xatol=1e-12, fatol=1e-14,
                                        maxiter=400))
            a = res.x[0] + 1j * res.x[1]
            g = -res.fun
            if g > best_g:
                best_a, best_g = a, g
        # Monotonicity guarantee: never do worse than the fallback starts.
        for s0 in extra_starts:
            g = self._gain(G, s0)
            if g > best_g:
                best_a, best_g = s0, g
        return best_a

    # -- reduced-remainder recursion --------------------------------------

    def _consume_pole(self, G: np.ndarray, a: complex) -> Tuple[np.ndarray, np.ndarray]:
        """One AFD reduction step: extract coefficient, divide out b_a."""
        M = np.sqrt(1.0 - np.abs(a) ** 2) * self._interior_value(G, a)
        e = kernel_e(self._z, a)
        b = blaschke_b(self._z, a)
        shape = (len(self._z),) + (1,) * (G.ndim - 1)
        G_next = (G - e.reshape(shape) * M) / b.reshape(shape)
        return M, G_next

    def _reduced_residual(self, pole_list: Sequence[complex]) -> np.ndarray:
        """Project F onto TM(pole_list); return B^{-1}(F - projection)."""
        G = self._F.copy()
        for a in pole_list:
            _, G = self._consume_pole(G, a)
        return G

    def _project(self) -> float:
        """Recompute matrix coefficients for the current poles; return error."""
        G = self._F.copy()
        self.coeffs = []
        for a in self.poles:
            M, G = self._consume_pole(G, a)
            self.coeffs.append(M)
        approx = self.evaluate(self._z)
        err = np.linalg.norm((self._F - approx).ravel()) / \
            np.linalg.norm(self._F.ravel())
        return float(err)

    # -- public API -------------------------------------------------------

    def fit(self, z_samples: np.ndarray, F_samples: np.ndarray) -> "MimoAFDVF":
        """Fit from samples F_samples[i] = F(z_samples[i]) on a uniform
        circle grid.  F_samples has shape (L, p, q) (or (L,) for scalar)."""
        self._z = np.asarray(z_samples)
        self._F = np.asarray(F_samples, dtype=complex)

        # Phase 1: greedy matrix AFD on the reduced remainder.  Fixed
        # basis functions (e.g. the constant term at a = 0) are consumed
        # first, then the remaining poles are chosen greedily.
        G = self._F.copy()
        self.poles = []
        for a in self.fixed_poles:
            self.poles.append(complex(a))
            _, G = self._consume_pole(G, a)
        for k in range(len(self.fixed_poles), self.n_poles):
            a = self._find_pole(G)
            self.poles.append(a)
            _, G = self._consume_pole(G, a)
            if self.verbose:
                print(f"  greedy pole {k + 1}/{self.n_poles}: "
                      f"a = {a:.6f}")
        err = self._project()
        self.error_history = [err]
        if self.verbose:
            print(f"  greedy relative error: {err:.3e}")

        # Phase 2: energy-monotone cyclic relocation (free poles only).
        # Theorem: each relocation cannot decrease the captured energy, so
        # the error is non-increasing in exact arithmetic.  Near the
        # accuracy floor, quadrature/optimizer noise can produce O(1e-9)
        # upticks; the snapshot/revert below removes even those.
        for cycle in range(self.refine_cycles):
            poles_prev = list(self.poles)
            for j in range(len(self.fixed_poles), self.n_poles):
                rest = self.poles[:j] + self.poles[j + 1:]
                G = self._reduced_residual(rest)
                self.poles[j] = self._find_pole(
                    G, extra_starts=[self.poles[j]])
            err_new = self._project()
            if err_new > err:
                self.poles = poles_prev
                self._project()
                break
            self.error_history.append(err_new)
            if self.verbose:
                print(f"  relocation sweep {cycle + 1}: "
                      f"relative error {err_new:.3e}")
            if err - err_new < self.rtol * max(err, EPS):
                break
            err = err_new
        return self

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        """Evaluate the model sum_k M_k B_k(z)."""
        z = np.asarray(z)
        tail_shape = self.coeffs[0].shape if self.coeffs else ()
        out = np.zeros(z.shape + tail_shape, dtype=complex)
        prefix = np.ones_like(z, dtype=complex)
        shape = z.shape + (1,) * len(tail_shape)
        for a, M in zip(self.poles, self.coeffs):
            B = (prefix * kernel_e(z, a)).reshape(shape)
            out += B * M
            prefix = prefix * blaschke_b(z, a)
        return out

    def lambda_poles(self) -> np.ndarray:
        """Identified poles mapped back to the prototype frequency plane.
        Fixed basis functions (the direct term) carry no physical pole and
        are excluded."""
        free = self.poles[len(self.fixed_poles):]
        return np.array([tm_param_to_lambda_pole(a) for a in free])


# ---------------------------------------------------------------------------
# Coupled-resonator (coupling matrix) filter models
# ---------------------------------------------------------------------------

def chebyshev_g_values(N: int, ripple_db: float) -> np.ndarray:
    """Element values g_1..g_{N+1} of the Chebyshev lowpass prototype."""
    beta = np.log(1.0 / np.tanh(ripple_db / 17.37))
    gamma = np.sinh(beta / (2.0 * N))
    a = np.array([np.sin((2 * k - 1) * np.pi / (2 * N)) for k in range(1, N + 1)])
    b = np.array([gamma ** 2 + np.sin(k * np.pi / N) ** 2 for k in range(1, N + 1)])
    g = np.empty(N + 1)
    g[0] = 2.0 * a[0] / gamma
    for k in range(1, N):
        g[k] = 4.0 * a[k - 1] * a[k] / (b[k - 1] * g[k - 1])
    g[N] = 1.0 if N % 2 == 1 else 1.0 / np.tanh(beta / 4.0) ** 2
    return g


def inline_chebyshev_coupling(N: int, ripple_db: float
                              ) -> Tuple[np.ndarray, float, float]:
    """Inline (all-pole) Chebyshev coupling matrix and port resistances."""
    g = chebyshev_g_values(N, ripple_db)
    M = np.zeros((N, N))
    for k in range(N - 1):
        M[k, k + 1] = M[k + 1, k] = 1.0 / np.sqrt(g[k] * g[k + 1])
    R1 = 1.0 / g[0]
    RN = 1.0 / (g[N - 1] * g[N])
    return M, R1, RN


def coupled_resonator_smatrix(M: np.ndarray, R1: float, RN: float,
                              lam: np.ndarray) -> np.ndarray:
    """2x2 S-matrix of the lowpass coupled-resonator model.

    A(lambda) = lambda*I - j*R + M,
    S11 = 1 + 2j R1 [A^-1]_{11},  S22 = 1 + 2j RN [A^-1]_{NN},
    S21 = S12 = -2j sqrt(R1 RN) [A^-1]_{N1}.
    """
    N = M.shape[0]
    R = np.zeros((N, N))
    R[0, 0], R[-1, -1] = R1, RN
    S = np.empty((len(lam), 2, 2), dtype=complex)
    I = np.eye(N)
    for i, l in enumerate(lam):
        Ainv = np.linalg.inv(l * I - 1j * R + M)
        s11 = 1.0 + 2j * R1 * Ainv[0, 0]
        s22 = 1.0 + 2j * RN * Ainv[-1, -1]
        s21 = -2j * np.sqrt(R1 * RN) * Ainv[-1, 0]
        S[i] = [[s11, s21], [s21, s22]]
    return S


def true_lambda_poles(M: np.ndarray, R1: float, RN: float) -> np.ndarray:
    """Poles of the response in the prototype frequency plane:
    roots of det(lambda*I - jR + M) = 0, i.e. eigenvalues of jR - M."""
    N = M.shape[0]
    R = np.zeros((N, N))
    R[0, 0], R[-1, -1] = R1, RN
    return np.linalg.eigvals(1j * R - M)


def sample_filter_on_circle(M: np.ndarray, R1: float, RN: float,
                            n_samples: int
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the filter S-matrix on a uniform circle grid via Cayley."""
    z, theta = uniform_circle_grid(n_samples)
    lam = lambda_from_theta(theta)
    S = coupled_resonator_smatrix(M, R1, RN, lam)
    return z, lam, S


# ---------------------------------------------------------------------------
# Baseline: entrywise scalar AFD-VF (independent pole sets per entry)
# ---------------------------------------------------------------------------

def fit_entrywise_scalar(z: np.ndarray, S: np.ndarray, n_poles: int,
                         refine_cycles: int) -> Tuple[np.ndarray, list]:
    """Fit S11, S21, S22 independently (S12 = S21 by reciprocity)."""
    entries = [(0, 0), (1, 0), (1, 1)]
    approx = np.empty_like(S)
    models = []
    for (i, j) in entries:
        m = MimoAFDVF(n_poles, refine_cycles=refine_cycles,
                      fixed_poles=[0.0], verbose=False)
        m.fit(z, S[:, i, j])
        models.append(m)
        approx[:, i, j] = m.evaluate(z)
    approx[:, 0, 1] = approx[:, 1, 0]
    return approx, models


# ---------------------------------------------------------------------------
# Metrics and plotting
# ---------------------------------------------------------------------------

def rel_error(S: np.ndarray, S_hat: np.ndarray) -> float:
    return float(np.linalg.norm((S - S_hat).ravel()) /
                 np.linalg.norm(S.ravel()))


def pole_match_errors(true_poles: np.ndarray,
                      found_poles: np.ndarray) -> np.ndarray:
    """Distance from each true pole to its nearest identified pole."""
    return np.array([np.min(np.abs(found_poles - p)) for p in true_poles])


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x) + EPS)


def _style_axis(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(labelsize=8)


def plot_experiment(name: str, filename: str, lam: np.ndarray,
                    S: np.ndarray, greedy: MimoAFDVF, hybrid: MimoAFDVF,
                    true_poles: np.ndarray, lam_view: float = 3.0):
    z_view = np.abs(lam) <= lam_view
    idx = np.argsort(lam[z_view])
    lv = lam[z_view][idx]
    S_v = S[z_view][idx]
    Sg = greedy.evaluate((1j * lv - 1.0) / (1j * lv + 1.0))
    Sh = hybrid.evaluate((1j * lv - 1.0) / (1j * lv + 1.0))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    # (a) responses in dB
    ax = axes[0, 0]
    ax.plot(lv, db(S_v[:, 0, 0]), color=C_TRUE, lw=1.6, label='true $S_{11}$')
    ax.plot(lv, db(S_v[:, 1, 0]), color=C_TRUE, lw=1.6, ls='--',
            label='true $S_{21}$')
    ax.plot(lv, db(Sh[:, 0, 0]), color=C_VERM, lw=1.0, ls=':',
            label='hybrid model')
    ax.plot(lv, db(Sh[:, 1, 0]), color=C_VERM, lw=1.0, ls=':')
    ax.set_xlabel(r'$\lambda$ (prototype frequency)', fontsize=9)
    ax.set_ylabel('magnitude (dB)', fontsize=9)
    ax.set_ylim(-80, 5)
    ax.set_title('(a) S-parameters, true vs. hybrid model', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style_axis(ax)

    # (b) pointwise Frobenius error
    ax = axes[0, 1]
    err_g = np.linalg.norm((S_v - Sg).reshape(len(lv), -1), axis=1)
    err_h = np.linalg.norm((S_v - Sh).reshape(len(lv), -1), axis=1)
    ax.semilogy(lv, err_g + EPS, color=C_BLUE, lw=1.2, label='greedy MIMO AFD')
    ax.semilogy(lv, err_h + EPS, color=C_VERM, lw=1.2, label='hybrid MIMO AFD-VF')
    ax.set_xlabel(r'$\lambda$', fontsize=9)
    ax.set_ylabel(r'$\|S(\lambda)-\hat S(\lambda)\|_F$', fontsize=9)
    ax.set_title('(b) pointwise error', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style_axis(ax)

    # (c) pole map in the lambda plane
    ax = axes[1, 0]
    pg = greedy.lambda_poles()
    ph = hybrid.lambda_poles()
    ax.scatter(true_poles.real, true_poles.imag, marker='x', s=70,
               color=C_TRUE, label='true poles', zorder=3, linewidths=1.6)
    ax.scatter(pg.real, pg.imag, marker='o', s=45, facecolors='none',
               edgecolors=C_BLUE, label='greedy', zorder=2, linewidths=1.2)
    ax.scatter(ph.real, ph.imag, marker='^', s=45, facecolors='none',
               edgecolors=C_VERM, label='hybrid', zorder=2, linewidths=1.2)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xlabel(r'Re $\lambda$', fontsize=9)
    ax.set_ylabel(r'Im $\lambda$', fontsize=9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, 2.0)
    ax.set_title('(c) pole locations (prototype plane)', fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    _style_axis(ax)

    # (d) monotone convergence of the relocation sweeps
    ax = axes[1, 1]
    hist = np.array(hybrid.error_history)
    ax.semilogy(np.arange(len(hist)), hist, color=C_VERM, lw=1.4,
                marker='o', ms=4, label='hybrid (relocation sweeps)')
    ax.axhline(greedy.error_history[0], color=C_BLUE, lw=1.2, ls='--',
               label='greedy (no refinement)')
    ax.set_xlabel('relocation sweep', fontsize=9)
    ax.set_ylabel('relative error', fontsize=9)
    ax.set_title('(d) monotone convergence of refinement', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style_axis(ax)

    fig.suptitle(name, fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  figure saved: {filename}")


def plot_noise_experiment(filename: str, noise_levels_db: Sequence[float],
                          errors: dict, floor: dict):
    """Grouped bars: relative error per method per noise level."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    methods = [('greedy MIMO AFD', C_BLUE),
               ('hybrid MIMO AFD-VF', C_VERM),
               ('entrywise scalar AFD-VF', C_GREEN)]
    x = np.arange(len(noise_levels_db))
    width = 0.26
    bottom = 1e-9
    for m, (label, color) in enumerate(methods):
        vals = np.maximum(np.array(errors[label]), 2 * bottom)
        bars = ax.bar(x + (m - 1) * width, vals, width * 0.92, color=color,
                      bottom=0, label=label, edgecolor='white', linewidth=1.0)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.1e}', (b.get_x() + b.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 2),
                        ha='center', fontsize=7, color='#333333')
    drew_floor = False
    for i, nl in enumerate(noise_levels_db):
        if np.isfinite(nl):
            ax.plot([i - 1.6 * width, i + 1.6 * width],
                    [floor[nl], floor[nl]], color=C_TRUE, lw=1.0, ls=':')
            drew_floor = True
    if drew_floor:
        ax.plot([], [], color=C_TRUE, lw=1.0, ls=':', label='noise floor')
    ax.set_yscale('log')
    ax.set_ylim(bottom, 4.0)
    ax.set_xticks(x)
    ax.set_xticklabels([('noiseless' if np.isinf(nl) else f'{nl:.0f} dB noise')
                        for nl in noise_levels_db], fontsize=9)
    ax.set_ylabel('relative error', fontsize=9)
    ax.set_title('Noise robustness (cross-coupled filter, $n=5$ poles)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper left',
              bbox_to_anchor=(0.0, -0.10))
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  figure saved: {filename}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def check_passivity(S: np.ndarray) -> float:
    """Max deviation of S^H S from the identity (lossless => 0)."""
    prods = np.einsum('lij,lik->ljk', np.conj(S), S)
    return float(np.max(np.abs(prods - np.eye(2))))


def run_filter_experiment(name: str, filename: str, M: np.ndarray,
                          R1: float, RN: float, n_poles: int,
                          n_samples: int = 600, noise_db: float = np.inf,
                          seed: int = 0):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    z, lam, S = sample_filter_on_circle(M, R1, RN, n_samples)
    unit_dev = check_passivity(S)
    print(f"  losslessness check |S^H S - I| = {unit_dev:.2e}")

    if np.isfinite(noise_db):
        rng = np.random.default_rng(seed)
        sigma = 10.0 ** (noise_db / 20.0)
        S = S + sigma * (rng.standard_normal(S.shape) +
                         1j * rng.standard_normal(S.shape)) / np.sqrt(2)

    tp = true_lambda_poles(M, R1, RN)
    print("  true prototype poles:",
          np.array2string(np.sort_complex(tp), precision=4))

    t0 = time.time()
    greedy = MimoAFDVF(n_poles, refine_cycles=0,
                       fixed_poles=[0.0], verbose=False).fit(z, S)
    t_g = time.time() - t0
    t0 = time.time()
    hybrid = MimoAFDVF(n_poles, refine_cycles=30,
                       fixed_poles=[0.0], verbose=False).fit(z, S)
    t_h = time.time() - t0

    e_g = rel_error(S, greedy.evaluate(z))
    e_h = rel_error(S, hybrid.evaluate(z))
    hist = np.array(hybrid.error_history)
    max_increase = float(np.max(np.diff(hist))) if len(hist) > 1 else 0.0

    pm_g = pole_match_errors(tp, greedy.lambda_poles())
    pm_h = pole_match_errors(tp, hybrid.lambda_poles())

    print(f"  greedy MIMO AFD    : rel. error {e_g:.3e}  ({t_g:.2f} s)")
    print(f"  hybrid MIMO AFD-VF : rel. error {e_h:.3e}  ({t_h:.2f} s)")
    print(f"  error reduction    : {(1 - e_h / e_g) * 100:.1f}%")
    print(f"  refinement monotone: max sweep-to-sweep increase "
          f"{max_increase:.2e}")
    print(f"  pole recovery (max |Δλ|): greedy {pm_g.max():.2e}, "
          f"hybrid {pm_h.max():.2e}")

    if filename:
        plot_experiment(name, filename, lam, S, greedy, hybrid, tp)

    return dict(z=z, lam=lam, S=S, greedy=greedy, hybrid=hybrid,
                e_g=e_g, e_h=e_h, t_g=t_g, t_h=t_h,
                pm_g=pm_g, pm_h=pm_h, true_poles=tp,
                max_increase=max_increase)


def main():
    print("=" * 70)
    print("MIMO HYBRID AFD-VF - MICROWAVE FILTER IDENTIFICATION EXPERIMENTS")
    print("=" * 70)

    results = {}

    # ---- Experiment M1: inline Chebyshev filter (all-pole) --------------
    N = 4
    M_in, R1_in, RN_in = inline_chebyshev_coupling(N, ripple_db=0.0432)
    results['M1'] = run_filter_experiment(
        "Experiment M1: inline Chebyshev filter (N=4, RL=20 dB)",
        "mimo_experiment1.png", M_in, R1_in, RN_in, n_poles=N + 1)

    # ---- Experiment M2: cross-coupled filter (transmission zeros) -------
    M_cc = M_in.copy()
    M_cc[0, 3] = M_cc[3, 0] = -0.20   # folded cross-coupling => 2 TZs
    results['M2'] = run_filter_experiment(
        "Experiment M2: cross-coupled filter (N=4, two transmission zeros)",
        "mimo_experiment2.png", M_cc, R1_in, RN_in, n_poles=N + 1)

    # ---- Experiment M3: noise robustness + scalar baseline --------------
    print(f"\n{'=' * 70}\nExperiment M3: noise robustness and "
          f"entrywise-scalar comparison\n{'=' * 70}")
    noise_levels = [np.inf, -60.0, -40.0]
    errors = {'greedy MIMO AFD': [], 'hybrid MIMO AFD-VF': [],
              'entrywise scalar AFD-VF': []}
    floors = {}
    for nl in noise_levels:
        z, lam, S_clean = sample_filter_on_circle(M_cc, R1_in, RN_in, 600)
        S = S_clean.copy()
        if np.isfinite(nl):
            rng = np.random.default_rng(1)
            sigma = 10.0 ** (nl / 20.0)
            S = S + sigma * (rng.standard_normal(S.shape) +
                             1j * rng.standard_normal(S.shape)) / np.sqrt(2)
        floors[nl] = (np.linalg.norm((S - S_clean).ravel()) /
                      np.linalg.norm(S_clean.ravel())) if np.isfinite(nl) else EPS

        g = MimoAFDVF(5, refine_cycles=0,
                      fixed_poles=[0.0], verbose=False).fit(z, S)
        h = MimoAFDVF(5, refine_cycles=30,
                      fixed_poles=[0.0], verbose=False).fit(z, S)
        S_sc, models_sc = fit_entrywise_scalar(z, S, 5, refine_cycles=30)
        # Errors are measured against the clean response.
        e_g = rel_error(S_clean, g.evaluate(z))
        e_h = rel_error(S_clean, h.evaluate(z))
        e_s = rel_error(S_clean, S_sc)
        errors['greedy MIMO AFD'].append(e_g)
        errors['hybrid MIMO AFD-VF'].append(e_h)
        errors['entrywise scalar AFD-VF'].append(e_s)
        tp = true_lambda_poles(M_cc, R1_in, RN_in)
        pm = pole_match_errors(tp, h.lambda_poles())
        lbl = 'noiseless' if np.isinf(nl) else f'{nl:.0f} dB'
        print(f"  [{lbl:>9s}] greedy {e_g:.3e} | hybrid {e_h:.3e} | "
              f"scalar {e_s:.3e} | hybrid pole err {pm.max():.2e}")

    print("\n  parameter count (complex numbers): "
          "MIMO common-pole: 5 poles + 5x(2x2 sym => 3) residues = 20; "
          "entrywise scalar: 3x(5 poles + 5 residues) = 30")
    plot_noise_experiment("mimo_experiment3.png", noise_levels, errors, floors)

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'Experiment':<42}{'greedy':>10}{'hybrid':>10}{'reduction':>10}")
    for key, label in [('M1', 'M1 inline Chebyshev (N=4)'),
                       ('M2', 'M2 cross-coupled, 2 TZs (N=4)')]:
        r = results[key]
        print(f"{label:<42}{r['e_g']:>10.2e}{r['e_h']:>10.2e}"
              f"{(1 - r['e_h'] / r['e_g']) * 100:>9.1f}%")
    print("\nAll experiments completed.")
    return results


if __name__ == "__main__":
    main()
