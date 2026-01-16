import numpy as np
from math import factorial

# =========================
# 6D canonical order: [x, px, y, py, z, delta]
# =========================
J6 = np.array([
    [0, 1, 0, 0, 0, 0],
    [-1,0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0,-1,0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0,-1,0],
], dtype=float)

def sympl_error(M: np.ndarray) -> float:
    return np.linalg.norm(M.T @ J6 @ M - J6)

# =========================
# TPSA core (Truncated Power Series Algebra)
# =========================
class TPSA:
    __slots__ = ("nvar", "order", "c")

    def __init__(self, nvar: int, order: int, coeffs=None):
        self.nvar = nvar
        self.order = order
        self.c = {} if coeffs is None else dict(coeffs)

    @staticmethod
    def const(nvar, order, value: float):
        t = TPSA(nvar, order)
        if value != 0.0:
            t.c[(0,)*nvar] = float(value)
        return t

    @staticmethod
    def var(nvar, order, i: int):
        t = TPSA(nvar, order)
        exp = [0]*nvar
        exp[i] = 1
        t.c[tuple(exp)] = 1.0
        return t

    def copy(self):
        return TPSA(self.nvar, self.order, self.c)

    def _truncate(self):
        kill = []
        for e in self.c:
            if sum(e) > self.order:
                kill.append(e)
        for e in kill:
            del self.c[e]
        kill = [e for e,v in self.c.items() if v == 0.0]
        for e in kill:
            del self.c[e]
        return self

    def __add__(self, other):
        if isinstance(other, (int,float)):
            other = TPSA.const(self.nvar, self.order, other)
        out = TPSA(self.nvar, self.order, self.c)
        for e,v in other.c.items():
            out.c[e] = out.c.get(e, 0.0) + v
        return out._truncate()

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int,float)):
            other = TPSA.const(self.nvar, self.order, other)
        out = TPSA(self.nvar, self.order, self.c)
        for e,v in other.c.items():
            out.c[e] = out.c.get(e, 0.0) - v
        return out._truncate()

    def __rsub__(self, other):
        return TPSA.const(self.nvar, self.order, other) - self

    def __neg__(self):
        return 0.0 - self

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            other = TPSA.const(self.nvar, self.order, other)
        out = TPSA(self.nvar, self.order)
        for e1,v1 in self.c.items():
            for e2,v2 in other.c.items():
                e = tuple(e1[i] + e2[i] for i in range(self.nvar))
                if sum(e) <= self.order:
                    out.c[e] = out.c.get(e, 0.0) + v1*v2
        return out._truncate()

    __rmul__ = __mul__

    def __pow__(self, p: int):
        assert p >= 0
        if p == 0:
            return TPSA.const(self.nvar, self.order, 1.0)
        out = self.copy()
        for _ in range(p-1):
            out = out * self
        return out

    def deriv(self, i: int):
        out = TPSA(self.nvar, self.order)
        for e,v in self.c.items():
            if e[i] == 0:
                continue
            ee = list(e)
            ee[i] -= 1
            out.c[tuple(ee)] = out.c.get(tuple(ee), 0.0) + v * e[i]
        return out._truncate()

    def eval(self, x: np.ndarray) -> float:
        s = 0.0
        for e,v in self.c.items():
            term = v
            for i,ei in enumerate(e):
                if ei:
                    term *= (x[i] ** ei)
            s += term
        return s

    def substitute(self, subs):
        out = TPSA(self.nvar, self.order)
        for e,v in self.c.items():
            term = TPSA.const(self.nvar, self.order, v)
            for i,ei in enumerate(e):
                if ei:
                    term = term * (subs[i] ** ei)
            out = out + term
        return out._truncate()

# =========================
# Series utilities (TPSA-safe)
# =========================
def inv1p(d: TPSA):
    # 1/(1+d) = sum_{k=0..order} (-d)^k
    out = TPSA.const(d.nvar, d.order, 0.0)
    one = TPSA.const(d.nvar, d.order, 1.0)
    term = one
    out = out + one
    for _k in range(1, d.order+1):
        term = term * (-d)
        out = out + term
    return out

def sin_tpsa(u: TPSA):
    out = TPSA.const(u.nvar, u.order, 0.0)
    for k in range(0, u.order+1):
        p = 2*k + 1
        if p > u.order:
            break
        out = out + ((-1)**k) * (u**p) * (1.0/factorial(p))
    return out

def cos_tpsa(u: TPSA):
    out = TPSA.const(u.nvar, u.order, 0.0)
    for k in range(0, u.order+1):
        p = 2*k
        if p > u.order:
            break
        out = out + ((-1)**k) * (u**p) * (1.0/factorial(p))
    return out

# =========================
# S0.2 — Poisson bracket + Lie exponential
# ad_F(g) := {g, F}
# =========================
PAIRS = [(0,1), (2,3), (4,5)]  # (q,p) index pairs

def poisson(g: TPSA, f: TPSA) -> TPSA:
    """
    {g,f} = Σ (dg/dq df/dp - dg/dp df/dq) over (x,px),(y,py),(z,δ)
    """
    out = TPSA.const(g.nvar, g.order, 0.0)
    for (q,p) in PAIRS:
        out = out + (g.deriv(q) * f.deriv(p) - g.deriv(p) * f.deriv(q))
    return out

def exp_lie_apply(F: TPSA, g: TPSA, L: float = 1.0, n_terms: int | None = None) -> TPSA:
    """
    exp(L :F:) g = Σ_{k=0..n_terms} (L^k/k!) ad_F^k(g)
    where ad_F(g) = {g,F}
    """
    if n_terms is None:
        n_terms = g.order  # default: safe upper bound

    out = g.copy()
    term = g.copy()
    for k in range(1, n_terms+1):
        term = poisson(term, F)              # ad_F(term)
        out = out + (L**k / factorial(k)) * term
    return out._truncate()

def map_from_generator(F: TPSA, vars6: list[TPSA], L: float = 1.0, n_terms=None) -> list[TPSA]:
    return [exp_lie_apply(F, v, L=L, n_terms=n_terms) for v in vars6]

def compose_map(mapA: list[TPSA], mapB: list[TPSA]) -> list[TPSA]:
    # mapA ∘ mapB
    return [f.substitute(mapB) for f in mapA]

def jacobian_at(mapF: list[TPSA], point: np.ndarray) -> np.ndarray:
    nvar = len(mapF)
    M = np.zeros((nvar, nvar), dtype=float)
    for i, Fi in enumerate(mapF):
        for j in range(nvar):
            M[i, j] = Fi.deriv(j).eval(point)
    return M

def verify_symplectic_random(mapF: list[TPSA], n: int = 16, amp: float = 1e-3, tol: float = 1e-9) -> tuple[bool, float]:
    worst = 0.0
    ok = True
    for _ in range(n):
        pt = np.random.uniform(-amp, amp, size=6)
        M = jacobian_at(mapF, pt)
        err = sympl_error(M)
        worst = max(worst, err)
        if err > tol:
            ok = False
    return ok, worst

# =========================
# Lie-generators for elements (6D)
# Convention: map = exp(L :F:) acting on canonical vars
# =========================
def gen_drift(vars6: list[TPSA]) -> TPSA:
    """
    Minimal 6D drift Hamiltonian polynomialized:
      F = 0.5*(px^2 + py^2) * 1/(1+δ)
    This guarantees x' = x + L*px/(1+δ), y similarly.
    z evolution will be canonical from this choice (non matching any ad-hoc z formula).
    """
    x, px, y, py, z, d = vars6
    inv = inv1p(d)
    return 0.5 * (px*px + py*py) * inv

def gen_quad_thin(vars6: list[TPSA], k1: float) -> TPSA:
    x, px, y, py, z, d = vars6
    # F = (k1/2)(x^2 - y^2)  => px += -∂F/∂x = -k1 x ; py += -∂F/∂y = +k1 y
    return 0.5 * k1 * (x*x - y*y)

def gen_sext_thin(vars6: list[TPSA], k2: float) -> TPSA:
    x, px, y, py, z, d = vars6
    # F = (k2/6)(x^3 - 3 x y^2)
    return (k2/6.0) * (x**3 - 3.0*x*(y**2))

def gen_rf_thin(vars6: list[TPSA], Vnorm: float, k_rf: float, phi0: float) -> TPSA:
    x, px, y, py, z, d = vars6
    # Want delta += V sin(k z + phi)
    # Choose F = (V/k) cos(k z + phi) so -∂F/∂z = V sin(...)
    u = (k_rf * z) + phi0
    return (Vnorm / k_rf) * cos_tpsa(u)

# =========================
# S0.2 SHOCK: Build one-turn Lie map in 6D (FODO + sext + RF)
# =========================
def shock_lie_6d(order=5, n_terms=None, seed=42):
    np.random.seed(seed)
    nvar = 6
    vars6 = [TPSA.var(nvar, order, i) for i in range(nvar)]

    # Lattice params
    Ld = 1.0
    kq = 0.8
    ks = 50.0
    V  = 1e-3
    krf = 2*np.pi
    phi0 = 0.0

    # Identity map
    Fmap = vars6

    # Elements as Lie maps
    # FODO:
    FqF = gen_quad_thin(vars6, +kq)
    FqD = gen_quad_thin(vars6, -kq)

    # Drift generator
    Fd = gen_drift(vars6)

    # Sext / RF
    Fs = gen_sext_thin(vars6, ks)
    Frf = gen_rf_thin(vars6, V, krf, phi0)

    # Compose maps: (note: generators defined on base vars; apply via exp_lie on current map by substitution)
    def apply_lie(Fgen: TPSA, L=1.0):
        nonlocal Fmap
        # express generator in current coordinates: Fgen(x) -> Fgen(Fmap)
        Fgen_cur = Fgen.substitute(Fmap)
        # apply exp(L :F:) to each coordinate function in the map
        Fmap = [exp_lie_apply(Fgen_cur, g, L=L, n_terms=n_terms) for g in Fmap]

    apply_lie(FqF, L=1.0)
    apply_lie(Fd,  L=Ld)
    apply_lie(FqD, L=1.0)
    apply_lie(Fd,  L=Ld)
    apply_lie(Fs,  L=1.0)
    apply_lie(Frf, L=1.0)

    # Checks
    origin = np.zeros(6)
    M0 = jacobian_at(Fmap, origin)
    err0 = sympl_error(M0)

    ok_rand, worst = verify_symplectic_random(Fmap, n=24, amp=1e-3, tol=1e-8)

    print("=== S0.2 LIE 6D SHOCK ===")
    print("TPSA order:", order, "| exp terms:", (n_terms if n_terms is not None else order))
    print("Symplectic error @ origin:", f"{err0:.3e}")
    print("Symplectic worst (random pts):", f"{worst:.3e}", "| PASS:", ok_rand)
    print("Mzδ block:\n", M0[4:6,4:6])
    return Fmap, M0, (err0, worst, ok_rand)

if __name__ == "__main__":
    shock_lie_6d(order=5, n_terms=None, seed=42)
