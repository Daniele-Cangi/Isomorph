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

# ============================================================
# S0.3 — BCH COMPILER (6D Lie algebra via Poisson commutator)
# Requires from S0.2:
# - class TPSA
# - poisson(g,f)  -> {g,f}
# - exp_lie_apply(F, g, L=1.0, n_terms=None)
# - jacobian_at(mapF, point)
# - sympl_error(M)
# ============================================================

def comm(A: TPSA, B: TPSA) -> TPSA:
    """Lie commutator for generators: [A,B] := {A,B}"""
    return poisson(A, B)

def bch_4(A: TPSA, B: TPSA) -> TPSA:
    """
    BCH up to 4th order in nested commutators:
      C = A + B
        + 1/2 [A,B]
        + 1/12 [A,[A,B]] + 1/12 [B,[B,A]]
        - 1/24 [B,[A,[A,B]]]
      + O(5)
    """
    AB   = comm(A, B)
    AAB  = comm(A, AB)
    BBA  = comm(B, comm(B, A))      # [B,[B,A]]
    BAAB = comm(B, comm(A, AB))     # [B,[A,[A,B]]]

    C = A + B
    C = C + 0.5   * AB
    C = C + (1.0/12.0) * AAB
    C = C + (1.0/12.0) * BBA
    C = C - (1.0/24.0) * BAAB
    return C

def apply_map_generator(F: TPSA, vars6: list[TPSA], n_terms=None) -> list[TPSA]:
    """Map from generator: z' = exp(:F:) z"""
    return [exp_lie_apply(F, v, L=1.0, n_terms=n_terms) for v in vars6]

def compose_maps(mapA: list[TPSA], mapB: list[TPSA]) -> list[TPSA]:
    """mapA ∘ mapB"""
    return [f.substitute(mapB) for f in mapA]

def map_via_sequence(generators: list[TPSA], vars6: list[TPSA], n_terms=None) -> list[TPSA]:
    """
    Build map = exp(:G_n:) ... exp(:G_2:) exp(:G_1:) vars
    (left-to-right application order, consistent with S0.2 apply_lie)
    """
    current = vars6
    for G in generators:
        current = [exp_lie_apply(G, g, L=1.0, n_terms=n_terms) for g in current]
    return current

def fold_bch_left(generators: list[TPSA]) -> TPSA:
    """C = BCH(...BCH(BCH(G1,G2),G3)...,Gn)"""
    C = generators[0]
    for G in generators[1:]:
        C = bch_4(C, G)
    return C

def fold_bch_tree(generators: list[TPSA]) -> TPSA:
    """
    Pairwise tree fold to reduce asymmetry error:
      [G1,G2]->H1, [G3,G4]->H2, ... then fold again.
    """
    level = list(generators)
    while len(level) > 1:
        nxt = []
        i = 0
        while i < len(level):
            if i+1 < len(level):
                nxt.append(bch_4(level[i], level[i+1]))
                i += 2
            else:
                nxt.append(level[i])
                i += 1
        level = nxt
    return level[0]

def map_diff_report(map1: list[TPSA], map2: list[TPSA],
                    n_pts=32, amp=1e-3, tol=1e-8, seed=123) -> dict:
    rng = np.random.default_rng(seed)
    worst = 0.0
    rms_acc = 0.0
    worst_sympl_1 = 0.0
    worst_sympl_2 = 0.0
    ok = True

    for _ in range(n_pts):
        pt = rng.uniform(-amp, amp, size=6)

        v1 = np.array([f.eval(pt) for f in map1])
        v2 = np.array([f.eval(pt) for f in map2])
        d = np.linalg.norm(v1 - v2)
        worst = max(worst, d)
        rms_acc += d*d
        if d > tol:
            ok = False

        M1 = jacobian_at(map1, pt)
        M2 = jacobian_at(map2, pt)
        e1 = sympl_error(M1)
        e2 = sympl_error(M2)
        worst_sympl_1 = max(worst_sympl_1, e1)
        worst_sympl_2 = max(worst_sympl_2, e2)

    rms = np.sqrt(rms_acc / n_pts)
    return {
        "ok": ok,
        "tol": tol,
        "n_pts": n_pts,
        "amp": amp,
        "worst_state_diff": worst,
        "rms_state_diff": rms,
        "worst_sympl_map1": worst_sympl_1,
        "worst_sympl_map2": worst_sympl_2,
    }

# ============================================================
# S0.3 SHOCK: compile a short lattice and prove equivalence
# ============================================================
def shock_bch_compile_6d(order=7, exp_terms=None, amp=1e-3):
    """
    Extreme: compile a mini-lattice into ONE generator via BCH,
    then verify map equivalence on random points.
    """
    nvar = 6
    vars6 = [TPSA.var(nvar, order, i) for i in range(nvar)]

    # --- Lattice knobs (you can mutate these hard) ---
    Ld  = 1.0
    kq  = 0.8
    ks  = 60.0
    V   = 2e-3
    krf = 2*np.pi
    phi = 0.0

    # Generators (from S0.2)
    # NOTE: exp(L:F:) == exp(:L*F:) since bracket is linear.
    Fd   = gen_drift(vars6)
    FqF  = gen_quad_thin(vars6, +kq)
    FqD  = gen_quad_thin(vars6, -kq)
    Fs   = gen_sext_thin(vars6, ks)
    Frf  = gen_rf_thin(vars6, V, krf, phi)

    # Build sequence: exp(:G1:) exp(:G2:) ...
    gens = [
        FqF,
        (Ld * Fd),
        FqD,
        (Ld * Fd),
        Fs,
        Frf,
    ]

    # Map by explicit sequence (ground truth)
    map_seq = map_via_sequence(gens, vars6, n_terms=exp_terms)

    # Compile to single generator via BCH
    C_left = fold_bch_left(gens)
    C_tree = fold_bch_tree(gens)

    map_left = apply_map_generator(C_left, vars6, n_terms=exp_terms)
    map_tree = apply_map_generator(C_tree, vars6, n_terms=exp_terms)

    # Reports
    rep_left = map_diff_report(map_seq, map_left, n_pts=48, amp=amp, tol=5e-7, seed=1)
    rep_tree = map_diff_report(map_seq, map_tree, n_pts=48, amp=amp, tol=5e-7, seed=2)

    print("=== S0.3 BCH COMPILER SHOCK (6D) ===")
    print(f"TPSA order={order} | exp_terms={(exp_terms if exp_terms is not None else order)} | amp={amp}")
    print("\n[LEFT-FOLD BCH]")
    for k,v in rep_left.items():
        print(f"  {k}: {v}")
    print("\n[TREE-FOLD BCH]")
    for k,v in rep_tree.items():
        print(f"  {k}: {v}")

    # Symplectic check at origin (quick sanity)
    Mseq0  = jacobian_at(map_seq,  np.zeros(6))
    Mleft0 = jacobian_at(map_left, np.zeros(6))
    Mtree0 = jacobian_at(map_tree, np.zeros(6))
    print("\nSymplectic error @ origin:")
    print("  seq :", f"{sympl_error(Mseq0):.3e}")
    print("  left:", f"{sympl_error(Mleft0):.3e}")
    print("  tree:", f"{sympl_error(Mtree0):.3e}")

    return {
        "gens": gens,
        "C_left": C_left,
        "C_tree": C_tree,
        "map_seq": map_seq,
        "map_left": map_left,
        "map_tree": map_tree,
        "rep_left": rep_left,
        "rep_tree": rep_tree,
    }

if __name__ == "__main__":
    shock_bch_compile_6d(order=7, exp_terms=None, amp=1e-3)
