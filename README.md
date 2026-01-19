# ISOMORPH - Symplectic Particle Accelerator Dynamics Simulation

> **Status: Development & Testing Phase**
> This project is currently under active development. APIs and features may change without notice.

## Overview

**ISOMORPH** is a scientific software suite for simulating and analyzing **nonlinear beam dynamics** in particle accelerators using advanced mathematical techniques from Hamiltonian mechanics, Lie algebra, and normal form theory.

The project implements a complete computational pipeline for:
1. **Symplectic tracking** of particles through magnetic lattice elements
2. **Truncated Power Series Algebra (TPSA)** for automatic differentiation
3. **Lie algebraic methods** for Hamiltonian map factorization
4. **Baker-Campbell-Hausdorff (BCH)** formula for map compilation
5. **Normal form analysis** for extracting nonlinear dynamics invariants
6. **Resonance driving term (RDT)** computation
7. **Dynamic aperture** and chaos indicators via tune diffusion

---

## Scientific Background

### 1. Symplectic Geometry in Accelerator Physics

Particle motion in a circular accelerator is governed by Hamilton's equations. The 6D phase space coordinates are:

```
z = (x, px, y, py, z, delta)
```

where:
- `(x, px)`: horizontal position and momentum
- `(y, py)`: vertical position and momentum
- `(z, delta)`: longitudinal position and relative momentum deviation

The evolution is described by a **symplectic map** M satisfying:

```
M^T J M = J
```

where J is the standard symplectic matrix (block-diagonal with 2x2 blocks [[0,1],[-1,0]]).

### 2. Truncated Power Series Algebra (TPSA)

TPSA enables **exact computation of Taylor maps** to arbitrary order. Each phase space variable becomes a polynomial:

```
f(z) = sum_{|alpha| <= N} c_alpha * z^alpha
```

Operations (+, -, *, composition) propagate coefficients automatically, providing:
- **Transfer maps** through magnetic elements
- **Jacobians** at any phase space point
- **Higher-order aberrations** and chromaticity

### 3. Lie Algebraic Methods

The one-turn map is factorized using **Lie operators**:

```
M = exp(:F_1:) exp(:F_2:) ... exp(:F_n:)
```

where `:F:` denotes the **Lie operator** associated with Hamiltonian generator F:

```
:F: g = {g, F} = Poisson bracket
```

The exponential is computed via the series:

```
exp(:F:) g = g + {g,F} + (1/2!){​{g,F},F} + ...
```

### 4. Baker-Campbell-Hausdorff (BCH) Formula

Multiple Lie factors can be combined into a single generator using BCH:

```
exp(:A:) exp(:B:) = exp(:C:)
```

where C is computed via nested commutators:

```
C = A + B + (1/2)[A,B] + (1/12)[A,[A,B]] + (1/12)[B,[B,A]] - ...
```

This enables **map compilation** - converting element-by-element tracking into a single polynomial transformation.

### 5. Normal Form Theory

Given a one-turn map M, normal form theory seeks a **near-identity transformation** A such that:

```
A^{-1} M A = R + (resonant terms only)
```

where R is a pure rotation (linear dynamics). The transformation removes all **non-resonant** terms, leaving:

- **Detuning coefficients**: Amplitude-dependent tune shifts
- **Resonance Driving Terms (RDTs)**: Coupling and nonlinear resonance strengths

The normal coordinates are action-angle variables `(J_i, phi_i)` where:

```
nu_i(J) = nu_i0 + sum_j (dnu_i/dJ_j) * J_j
```


## Module Descriptions

### Core TPSA & Tracking

| File | Description |
|------|-------------|
| `tpsa_tracker.py` | TPSA class implementation with 6D element maps (drift, quadrupole, sextupole, RF cavity) |
| `lie_tpsa_tracker.py` | Lie algebraic formulation: Poisson brackets, exp(:F:) operator, Lie generators |
| `bch_tpsa_tracker.py` | BCH formula implementation (4th order), map compilation via fold strategies |
| `symplectic_tracker.py` | Numerical tracking engine with multipole kicks, RF cavities, aperture limits |

### Normal Form Analysis

| File | Description |
|------|-------------|
| `s0_2_coupled_normal_form.py` | 4D coupled linear normal form, eigenvalue normalization, homological equation solver |
| `normal_form_tracker.py` | 6D normal form via numerical tracking + FFT tune estimation |
| `s0_3_preset_verification.py` | Integration test: preset lattice through full normal form pipeline |

### Specialized Tools

| File | Description |
|------|-------------|
| `unstable_preset.py` | Dynamic aperture scanner with NAFF-style tune diffusion analysis |
| `isomorph_wormhole.py` | Educational visualization of network routing with physics metaphors |

---

## Mathematical Elements Implemented

### Magnetic Elements (Thin-Lens Approximation)

1. **Drift**: Free propagation
   ```
   x' = x + L*px/(1+delta)
   y' = y + L*py/(1+delta)
   ```

2. **Quadrupole Kick**: Linear focusing
   ```
   px' = px - k1*x
   py' = py + k1*y
   ```

3. **Sextupole Kick**: Nonlinear chromatic correction
   ```
   px' = px - (k2/2)(x^2 - y^2)
   py' = py + k2*x*y
   ```

4. **Skew Quadrupole**: Coupling element
   ```
   px' = px - ks*y
   py' = py - ks*x
   ```

5. **RF Cavity**: Longitudinal focusing
   ```
   delta' = delta + V*sin(phi0 + k_rf*z)
   ```

---

## Usage Examples

### Run BCH Compilation Test
```bash
python bch_tpsa_tracker.py
```

### Run Normal Form Analysis
```bash
python normal_form_tracker.py
```

### Run Dynamic Aperture Scan
```bash
python unstable_preset.py
```

### Run Coupled Normal Form (4D Symbolic)
```bash
python s0_2_coupled_normal_form.py
```

---

## Output Diagnostics

The software computes and reports:

1. **Symplectic Error**: Deviation from M^T J M = J (should be < 1e-10 for exact maps)
2. **Tunes**: Fractional oscillation frequencies nu_x, nu_y, nu_s
3. **Detuning Coefficients**: dnu/dJ matrix (amplitude-dependent tune shifts)
4. **Resonance Driving Terms**: Non-zero coefficients at resonant frequencies
5. **Dynamic Aperture**: Maximum stable amplitude before particle loss
6. **Tune Diffusion**: Chaos indicator via NAFF algorithm

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- NumPy >= 1.21.0
- SymPy >= 1.10.0
- Matplotlib >= 3.5.0

---

## Theoretical References

1. **Dragt, A.J.** - "Lie Methods for Nonlinear Dynamics with Applications to Accelerator Physics"
2. **Forest, E.** - "Beam Dynamics: A New Attitude and Framework"
3. **Bengtsson, J.** - "The Sextupole Scheme for the Swiss Light Source"
4. **Laskar, J.** - "Frequency Analysis for Multi-dimensional Systems" (NAFF method)

---

## 7. Results Showcase (Chaotic Edge S0.11)
We successfully stabilized a chaotic edge regime using an octupole sculptor ($k_3 = -10^4$).

**High-Res Poincaré Plot (x=1.70mm):**
The visualization below shows the **last stable island chain** trapped just before the chaotic cliff. The particle survives 50,000 turns with zero diffusion, protected by the resonance island, while particles at 1.80mm are lost immediately.

![Chaotic Edge Islands](poincare_CHAOTIC_EDGE_k3m1e4_x1.70mm.png)

**Resonance Atlas (Tune Footprint):**
The frequency map analysis shows the extensive tune shift driven by the sextupoles and "sculpted" by the octupoles to maintain dynamic aperture.

![Resonance Atlas](resonance_atlas_s08.png)

---

## Contributing

This project is in active development. Contributions, bug reports, and feature requests are welcome via GitHub Issues.

---

## Resonance Forensics (S1.0 - S1.2)

We successfully "reverse engineered" the underlying chaotic structure of the "Chaotic Edge" preset ($k_3 = -10^4$) by peeling away layers of phase space distortion.

### 1. Raw Data (S1.0)
Initially, the Fourier fingerprint of the Poincaré section at $x=1.70$ mm showed a dominant $m=2$ signal. This was an artifact caused by the strong linear tilt ($\alpha_x$) and ellipticity of the beam.
![Raw](poincare_CHAOTIC_EDGE_k3m1e4_x1.70mm.png)

### 2. Linear Whitening (S1.1)
After normalizing the coordinates $(x, p_x) \to (X, P_X)$ using Twiss parameters, the $m=2$ signal vanished, revealing a dominant **m=3 (Triangle)** structure. This indicated that the geometric deformation from the strong sextupoles ($k_2=10$) was masking the resonance.
![Whitened](poincare_CHAOTIC_EDGE_k3m1e4_x1.70mm_white.png)

### 3. Symplectic Normal Form (S1.2)
Finally, by applying a full 4D Symplectic Normal Form transformation (eigen-decomposition of the one-turn matrix), we decoupled the $x-y$ motion. The true skeleton of the instability was revealed: the **4th Order Resonance ($m=4$)**, confirming that the island chain driving the chaos is indeed $4\nu_x = 1$.
![Symplectic](poincare_CHAOTIC_EDGE_k3m1e4_x1.70mm_symp.png)


### 8. Sticky Chaos (The Beach)
Our goal was to transform the "Cliff" (immediate loss) into a "Beach" (long-lived chaotic sticking) to enable safe particle extraction.

#### Step 1: Modulation & Validation (S1.3b ABR)
We introduced a quasi-periodic modulation $k_2(t) = k_{base}(1 + \epsilon \cos(\omega t))$ to break invariant tori.
- **Protocol**: Agile-But-Robust (ABR) scan (50k turns, 4 random phases).
- **Result**: Beach confirmed ($p_{beach}=10.2\%$, Width $\approx 56\mu m$).
- **Verdict**: Real effect, but narrow.

#### Step 2: Optimization Sweep (S1.3d)
We performed a parameter sweep on $\epsilon$ to maximize stickiness without creating stable islands ("Harbor").
- **Optimal $\epsilon$**: 0.10.
- **Result**: Beach fraction maximized (6.3% single-freq) with monotonically decreasing stability ($p_0$ dropped to 23%).

#### Step 3: Resonance Thickening (S1.4 Dual-Frequency)
To thicken the Cantori barrier, we added a second incommensurate frequency:
$$k_2(t) = k_{base} [1 + 0.10 \cos(\phi t) + 0.04 \cos(\sqrt{2} t)]$$

- **Physical Mechanism**: The secondary drive breaks the outer resonant layers that form the "semi-stable" harbor, converting them into sticky chaos.
- **Result**:
  - Beach Fraction: **11.8%** (Doubled vs Single Freq).
  - Wall Loss: Reduced from 69% to 64%.
  - Stability: Preserved at ~24%.

#### Step 4: Anti-Self-Deception (S1.4b Deep Confirm)
To ensure the "Beach" wasn't just a delayed loss (Wall) or misclassified stability, we ran a **200,000 turn** Deep Scan. We computed the **Transition Matrix** from 50k classification to 200k classification.

**Transition Matrix (50k $\to$ 200k):**
| Class (50k) | Fate @ 200k | Fraction | Interpretation |
| :--- | :--- | :--- | :--- |
| **Alive** | **Beach (Long)** | **6.3%** | **HIDDEN STICKY TAIL** |
| **Alive (>50k)** | **Harbor (Stable)** | **23.6%** | **TRUE CORE** |

**Conclusion**: The Dual-Frequency drive creates a massive, genuine sticky layer (Total Beach **13.2%**) verified by deep-time statistics and rigorous frequency sweeping (S1.4e).
![S1.4 Map](analysis_MUTATION_D_ABR.png)

#### Step 6: Truth Standardization (Phase 5)
A rigorous "Red Team" protocol (5 replicas, fixed domain) revealed that the 13% peak was an artifact of specific seeding/domain choices.
**Final Validated Performance**:
- **Total Beach**: **9.0% $\pm$ 0.5%** (Robust)
- **Harbor Stability**: **26%**
- **Physics**: The effect is real and stable, but parameter tuning in 4D has reached saturation. Next step: 6D Physics.

#### Statistical Physics of the Ring (Mixture Model)
The survival statistic $S(t)$ is best described by a **Mixture Cure Model**:
$$ S(t) = p_0 + (1-p_0) [\pi S_{sticky}(t) + (1-\pi) S_{fast}(t)] $$
- **$p_0$ (Cure Fraction)**: The Stable Harbor ($\approx 27\%$).
- **$S_{sticky}(t)$**: The heavy-tail transient (Power Law, slope $\approx -0.13$).
- **$S_{fast}(t)$**: The exponential wall loss.

This confirms we are manipulating the *proportions* ($\pi$) and *timescales* of the sticky transient, but the invariant core $p_0$ remains topological protected.

#### Step 7: 6D Physics (The CERN Step)
Moving to full 6D tracking ($Q_s \approx 0.005$) initially raised alarms (0% Stability), but rigorous Ablation Diagnostics traced this to a lattice mismatch. 
With the corrected physics, the **S1.4 Baseline** proved remarkably robust:
- **Uncorrected 6D**: Harbor 22.5% (Stable).
- **Chromatic Correction ($C=6.3$)**: Harbor **27.5%** (Matches 4D Perfect Baseline).

We have achieved **S1.5 Stabilization**: A dual-frequency chaotic drive that maintains a large, stable storage core (~27%) in full 6D phase space.

### Diagnostic Note: Symplecticity
The early finite-difference Jacobian check (`symplectic_tracker.py` toy test) reports an error of ~7e-3. This is a known limitation of the FD method conditioning and is superseded by the negligible error (~1e-12) and long-term boundedness verified in the production `TurboTracker` (S1.4/S1.5).

#### Classification Thresholds
- **Beach (Chaos)**: Particles lost in $t < 50k$ turns.
- **Sticky Halo**: Particles surviving $50k \le t < 200k$ turns (Transient).
- **Harbor (Core)**: Particles surviving $t \ge 200k$ turns (Topologically Protected).

#### Step 8: Controlled Extraction (Passo CERN)
To demonstrate "CERN on PC" capability, we implemented a **Slow Extraction Prototype** (`run_controlled_extraction_s1_6`).
- **Septum**: Particles beyond $4mm$ are extracted to a beamline.
- **Control Knob**: PID-controlled **Drive Ramp** ($\epsilon(t)$).
- **Prototype Result**: The system exhibits **Nuclear Stability**. It takes >30% Overdrive to initiate diffusion.


## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---


## Appendix A: Conceptual Extensions (Isomorph Wormhole)

The `isomorph_wormhole.py` module implements a **conceptual network routing model** using gravitational physics metaphors:
- **Kerr nodes**: Simulated massive/spinning objects creating "spacetime curvature".
- **Frame dragging**: Rotational effects from spinning nodes.
- **Quantum tunneling**: Dynamic topology changes when "stress" exceeds threshold.

*Note: This is a visualization/educational model, distinct from the symplectic physics engine.*

## Author

**Daniele Cangi**
GitHub: [https://github.com/Daniele-Cangi/Isomorph](https://github.com/Daniele-Cangi/Isomorph)
