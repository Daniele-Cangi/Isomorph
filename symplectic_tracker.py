import numpy as np
import math
from dataclasses import dataclass

# =========================
# 6D SYMPLECTIC CORE (S0)
# coords: [x, px, y, py, z, delta]
# =========================

J6 = np.array([
    [0, 1, 0, 0, 0, 0],
    [-1,0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0,-1,0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0,-1,0],
], dtype=float)

def symplectic_error(M: np.ndarray) -> float:
    return np.linalg.norm(M.T @ J6 @ M - J6)

def jacobian_fd(f, x0, eps=1e-8):
    """Finite-diff Jacobian of map f at x0 (6D)."""
    x0 = np.array(x0, dtype=float)
    J = np.zeros((6,6), dtype=float)
    fx = f(x0)
    for i in range(6):
        dx = np.zeros(6); dx[i] = eps
        fp = f(x0 + dx)
        fm = f(x0 - dx)
        J[:, i] = (fp - fm) / (2*eps)
    return J

class Element:
    def track(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, s: np.ndarray, turn: int = 0) -> np.ndarray:
        # Default alias for backward compatibility. 
        # Ignores 'turn' unless overridden.
        return self.track(s)

class Drift6D(Element):
    def __init__(self, L: float):
        self.L = float(L)

    def track(self, s):
        x, px, y, py, z, d = s
        inv = 1.0 / (1.0 + d)  # momentum scaling

        # Transverse drift
        x2 = x + self.L * px * inv
        y2 = y + self.L * py * inv

        # Longitudinal drift (paraxial path-length approx)
        # keeps coupling z<->(px,py,delta) visible; good enough for S0 R&D
        z2 = z + self.L * (0.5 * (px*px + py*py) * inv*inv + (inv - 1.0))

        return np.array([x2, px, y2, py, z2, d], dtype=float)

class MultipoleKick(Element):
    """
    Thin multipole kick using complex expansion:
      kick = sum_n knl[n]/n! * (x + i y)^n
      dpx = -Re(kick), dpy = +Im(kick)
    knl[0] ~ dipole kick, knl[1] quad, knl[2] sext, ...
    """
    def __init__(self, knl):
        self.knl = np.array(knl, dtype=float)

    def track(self, s):
        x, px, y, py, z, d = s
        zz = x + 1j*y
        kick = 0j
        # factorial grows, but we keep it short (R&D: add caching later)
        for n, kn in enumerate(self.knl):
            if kn == 0.0:
                continue
            kick += (kn / float(math.factorial(n))) * (zz ** n)

        dpx = -np.real(kick)
        dpy = +np.imag(kick)

        return np.array([x, px + dpx, y, py + dpy, z, d], dtype=float)

class RFCavityKick(Element):
    """
    Thin RF kick:
      delta += Vnorm * sin(phi0 + k_rf * z)
    where Vnorm is normalized (eV/p0c or simply a dimensionless knob for S0)
    """
    def __init__(self, Vnorm: float, k_rf: float, phi0: float):
        self.V = float(Vnorm)
        self.k = float(k_rf)
        self.phi0 = float(phi0)

    def track(self, s):
        x, px, y, py, z, d = s
        d2 = d + self.V * np.sin(self.phi0 + self.k * z)
        return np.array([x, px, y, py, z, d2], dtype=float)

class SkewQuadrupoleKick(Element):
    """
    Thin-lens skew quad (6D).
    Hamiltonian ~ ks * x * y  -> kicks:
      px' = px - ks * y
      py' = py - ks * x
    """
    def __init__(self, strength: float):
        self.ks = float(strength)

    def track(self, s):
        x, px, y, py, z, d = s
        return np.array([
            x,
            px - self.ks * y,
            y,
            py - self.ks * x,
            z,
            d
        ], dtype=float)

class QuadrupoleKick(Element):
    """Wrapper around MultipoleKick for Quad"""
    def __init__(self, strength: float):
        self.k = strength
        self._m = MultipoleKick([0, strength])
    def track(self, s): return self._m.track(s)

class SextupoleKick(Element):
    """Wrapper around MultipoleKick for Sext"""
    def __init__(self, strength: float):
        self.k = strength
        self._m = MultipoleKick([0, 0, strength])
    def track(self, s): return self._m.track(s)

class OctupoleKick(Element):
    """
    Thin-lens Octupole (6D).
    Potential ~ (k3/24)(x^4 - 6x^2 y^2 + y^4)
    Kicks:
    px' = px - (k3/6)(x^3 - 3x y^2)
    py' = py + (k3/6)(3x^2 y - y^3)
    """
    def __init__(self, strength: float):
        self.k3 = float(strength)

    def track(self, s):
        x, px, y, py, z, d = s
        k3 = self.k3
        dpx = -(k3/6.0) * (x**3 - 3*x*(y**2))
        dpy = +(k3/6.0) * (3*(x**2)*y - y**3)
        return np.array([
            x,
            px + dpx,
            y,
            py + dpy,
            z,
            d
        ], dtype=float)

# Alias for compatibility with preset
Drift = Drift6D

class WormholeCanonicalTranslate(Element):
    """
    'Wormhole' but CANONICAL:
    if inside entry radius, apply q-translation:
      (x,y,z) -> (x,y,z) + (dx,dy,dz)
    momenta unchanged => canonical map.
    """
    def __init__(self, entry, exit, radius):
        self.entry = np.array(entry, dtype=float)   # (x,y,z)
        self.exit  = np.array(exit, dtype=float)    # (x,y,z)
        self.radius = float(radius)

    def track(self, s):
        x, px, y, py, z, d = s
        pos = np.array([x,y,z], dtype=float)
        if np.linalg.norm(pos - self.entry) < self.radius:
            delta_q = self.exit - self.entry
            x, y, z = (pos + delta_q)
        return np.array([x, px, y, py, z, d], dtype=float)

class ModulatedSextupoleKick(Element):
    """
    Thin-lens sextupole kick with quasi-periodic modulation:
      k2(turn) = k2_base * (1 + eps*cos(omega*turn + phase))
    """
    def __init__(self, k2_base: float, eps: float=0.06, omega: float=None, phase: float=0.0):
        self.k2_base = float(k2_base)
        self.eps = float(eps)
        self.omega = float(omega if omega is not None else 2*np.pi*(np.sqrt(5.0)-1.0)/2.0)
        self.phase = float(phase)

    def apply(self, state, turn: int = 0):
        # Quasi-periodic modulation
        k2 = self.k2_base * (1.0 + self.eps * np.cos(self.omega * turn + self.phase))
        
        # 6D or 4D state? Wrapper assumes attributes or array
        # This simple tracker treats state as array usually, let's adapt standard apply
        has_attr = hasattr(state, 'x')
        x = state.x if has_attr else state[0]
        y = state.y if has_attr else state[2]
        
        dpx = -(k2 / 2.0) * (x**2 - y**2)
        dpy =  (k2)       * (x * y)
        
        if has_attr:
            state.px += dpx
            state.py += dpy
        else:
            state[1] += dpx
            state[3] += dpy
            
        return state

class SymplecticTracker6D:
    def __init__(self, elements):
        self.elements = elements
        # Optimization: Pre-check which elements accept 'turn'
        self._compiled_elements = []
        for elem in elements:
            func = elem.apply if hasattr(elem, 'apply') else elem.track
            # Check arguments
            import inspect
            sig = inspect.signature(func)
            accepts_turn = 'turn' in sig.parameters
            self._compiled_elements.append((func, accepts_turn))
            
    def one_turn(self, state_in, turn: int = 0):
        s = state_in.copy()
        for func, accepts_turn in self._compiled_elements:
            if accepts_turn:
                s = func(s, turn=turn)
            else:
                s = func(s)
        return s
    
    def track(self, start_state, n_turns):
        traj = np.zeros((n_turns, 6))
        s = start_state.copy()
        traj[0] = s
        for i in range(1, n_turns):
            s = self.one_turn(s, turn=i-1) # Turn index 0 for first turn
            traj[i] = s
        return traj, n_turns

    def verify_symplectic(self, s_ref, tol=1e-10):
        # Note: This method assumes one_turn does not depend on 'turn' for Jacobian calculation.
        # If elements like ModulatedSextupoleKick are present, the Jacobian will be for turn=0.
        def f(x): return self.one_turn(x, turn=0) # Pass turn=0 for Jacobian
        M = jacobian_fd(f, np.array(s_ref, dtype=float))
        return symplectic_error(M), M

# ====== SHOCK: Minimal ring ======
if __name__ == "__main__":
    # Build a toy ring: (FODO + sext + RF + wormhole)
    Ld = 1.0
    kq = 0.8      # quad integrated strength
    ks = 50.0     # sext integrated strength (aggressive on purpose)
    V  = 1e-3     # RF normalized kick
    krf = 2*np.pi # placeholder: tune later

    ring = [
        MultipoleKick([0, +kq, 0]), Drift6D(Ld),
        MultipoleKick([0, -kq, 0]), Drift6D(Ld),
        MultipoleKick([0, 0, ks]),  # sext kick
        RFCavityKick(Vnorm=V, k_rf=krf, phi0=0.0),
        WormholeCanonicalTranslate(entry=(0.002, 0.0, 0.0), exit=(0.0, 0.0, 0.05), radius=5e-4),
    ]

    tr = SymplecticTracker6D(ring)

    s0 = np.array([1e-3, 0.0, 1e-3, 0.0, 0.0, 1e-3])
    traj = tr.track(s0, n_turns=2000)

    err, M = tr.verify_symplectic(s0)
    print("Symplectic error (fd Jacobian):", err)
    print("Final state:", traj[-1])
