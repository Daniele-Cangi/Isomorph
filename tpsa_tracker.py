import numpy as np
from math import factorial

# =========================
# 6D symplectic form J
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
        self.c = {} if coeffs is None else dict(coeffs)  # {multi_index_tuple: float}

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
        # remove terms with total degree > order
        kill = []
        for e in self.c:
            if sum(e) > self.order:
                kill.append(e)
        for e in kill:
            del self.c[e]
        # remove tiny zeros
        kill = [e for e,v in self.c.items() if v == 0.0]
        for e in kill:
            del self.c[e]
        return self

    def __add__(self, other):
        if isinstance(other, (int,float)):
            other = TPSA.const(self.nvar, self.order, other)
        out = TPSA(self.nvar, self.order)
        out.c = dict(self.c)
        for e,v in other.c.items():
            out.c[e] = out.c.get(e, 0.0) + v
        return out._truncate()

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int,float)):
            other = TPSA.const(self.nvar, self.order, other)
        out = TPSA(self.nvar, self.order)
        out.c = dict(self.c)
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
        # evaluate series at numeric point x (len nvar)
        s = 0.0
        for e,v in self.c.items():
            term = v
            for i,ei in enumerate(e):
                if ei:
                    term *= (x[i] ** ei)
            s += term
        return s

    def substitute(self, subs):
        """
        Substitute variables with TPSA objects.
        subs: list of TPSA length nvar, where subs[i] replaces variable i
        """
        out = TPSA(self.nvar, self.order)
        for e,v in self.c.items():
            term = TPSA.const(self.nvar, self.order, v)
            for i,ei in enumerate(e):
                if ei:
                    term = term * (subs[i] ** ei)
            out = out + term
        return out._truncate()

# =========================
# Series utilities: inv(1+d), sin, cos truncated
# =========================
def inv1p(d: TPSA):
    # 1/(1+d) = sum_{k=0..order} (-d)^k
    out = TPSA.const(d.nvar, d.order, 0.0)
    one = TPSA.const(d.nvar, d.order, 1.0)
    term = one
    for k in range(0, d.order+1):
        if k == 0:
            out = out + one
        else:
            term = term * (-d)
            out = out + term
    return out

def sin_tpsa(u: TPSA):
    # sin(u) = sum (-1)^k u^(2k+1)/(2k+1)!
    out = TPSA.const(u.nvar, u.order, 0.0)
    for k in range(0, u.order+1):
        p = 2*k + 1
        if p > u.order:
            break
        out = out + ((-1)**k) * (u**p) * (1.0/factorial(p))
    return out

def cos_tpsa(u: TPSA):
    # cos(u) = sum (-1)^k u^(2k)/(2k)!
    out = TPSA.const(u.nvar, u.order, 0.0)
    for k in range(0, u.order+1):
        p = 2*k
        if p > u.order:
            break
        out = out + ((-1)**k) * (u**p) * (1.0/factorial(p))
    return out

# =========================
# 6D element maps in TPSA (polynomialized)
# =========================
def drift6D_map(vars6, L: float):
    x, px, y, py, z, d = vars6
    inv = inv1p(d)  # series for 1/(1+d)

    x2 = x + (L * px) * inv
    y2 = y + (L * py) * inv

    # longitudinal slip (paraxial series-consistent)
    # z' = z + L*( 0.5*(px^2+py^2)*inv^2 + (inv - 1) )
    inv2 = inv * inv
    z2 = z + L * (0.5*(px*px + py*py)*inv2 + (inv - 1.0))

    return [x2, px, y2, py, z2, d]

def quad_thin_kick_map(vars6, k1: float):
    x, px, y, py, z, d = vars6
    # thin-lens quad: px' = px - k1*x ; py' = py + k1*y
    return [x, px - k1*x, y, py + k1*y, z, d]

def sext_thin_kick_map(vars6, k2: float):
    x, px, y, py, z, d = vars6
    # thin sext: px' = px - (k2/2)(x^2 - y^2) ; py' = py + k2*x*y
    return [x, px - 0.5*k2*(x*x - y*y), y, py + k2*x*y, z, d]

def rf_thin_kick_map(vars6, Vnorm: float, k_rf: float, phi0: float):
    x, px, y, py, z, d = vars6
    u = (k_rf * z) + phi0
    # delta' = delta + V * sin(u)  (series)
    d2 = d + Vnorm * sin_tpsa(u)
    return [x, px, y, py, z, d2]

def compose_map(mapA, mapB):
    """
    Compose polynomial maps:
      mapA ∘ mapB
    mapA: list[TPSA] outputs in terms of vars
    mapB: list[TPSA] outputs in terms of vars
    returns list[TPSA]
    """
    return [f.substitute(mapB) for f in mapA]

def linear_matrix_from_map(mapF, nvar=6):
    """
    Extract Jacobian at origin from TPSA map:
      M_ij = ∂F_i/∂q_j evaluated at 0
    Since TPSA stores coefficients, derivative at 0 is just coefficient of monomial with exp_j=1.
    """
    M = np.zeros((nvar, nvar), dtype=float)
    origin = np.zeros(nvar)
    for i, Fi in enumerate(mapF):
        for j in range(nvar):
            dF = Fi.deriv(j)
            M[i, j] = dF.eval(origin)
    return M

# =========================
# SHOCK TEST: 6D FODO + sext + RF (TPSA order N)
# =========================
def shock_tpsa_6d(order=4):
    nvar = 6
    vars6 = [TPSA.var(nvar, order, i) for i in range(nvar)]

    L = 1.0
    kq = 0.8
    ks = 50.0
    V  = 1e-3
    krf = 2*np.pi
    phi0 = 0.0

    # Build one-turn map by composition
    # Order of operations in tracking code: kick, drift, kick...
    # HERE, if we want F = M_n ∘ ... ∘ M_1, we must compose carefully.
    # Let F_curr be map from start to current.
    # F_next = M_element ∘ F_curr
    
    F = vars6  # identity
    
    # FODO:
    # 1. Quad (+kq)
    F = compose_map(quad_thin_kick_map(F, +kq), vars6) # Initialize first step? No, wait.
    # The functions specific "map(vars)" returns the map in terms of vars.
    # if F is the accumulated map from s=0 to s, then next step is map_element(F).
    # Wait, my compose_map definition: compose_map(mapA, mapB) is mapA(mapB).
    # If mapA is the element map (in terms of canonical vars), and mapB is the accumulated map (in terms of initial vars),
    # then mapA ∘ mapB gives the new accumulated map.
    
    # Let's rebuild the sequence as done in the shock function in the user prompt:
    # F = vars6 (Identity)
    # F = compose_map(quad_thin_kick_map(F, +kq), F) -- This looks wrong if F is passed as vars.
    # The snippet says: F = compose_map(quad_thin_kick_map(F, +kq), F) which is recursive?
    # No, look at the prompt code:
    # F = vars6
    # F = compose_map(quad_thin_kick_map(F, +kq), F)
    # This implies quad_thin_kick_map returns a map in terms of F's variables.
    
    # Let's trace `quad_thin_kick_map(vars6, k1)`. It returns [x, px-k1*x, ...].
    # If we pass F (which is a list of TPSA in terms of initial vars), `quad_thin_kick_map(F, k1)` returns
    # [F[0], F[1]-k1*F[0], ...]. This effectively computes Element(AccumulatedMap).
    # So F_new = Element(F_old).
    # `compose_map` is doing `f.substitute(mapB)`.
    # If we call `quad_thin_kick_map(F, k1)`, we are ALREADY substituting F into the element expressions.
    # So `compose_map` might be redundant or used differently?
    
    # The user's code:
    # F = compose_map(quad_thin_kick_map(F, +kq), F)  <-- This usage seems odd if quad.. takes F.
    # Let's check `quad_thin_kick_map`. It takes `vars6`.
    # If `vars6` are the TPSA objects, it returns new TPSA objects.
    # If I pass `F` (list of TPSA) to `quad_thin_kick_map`, it does arithmetic on TPSA objects.
    # So `quad_thin_kick_map(F, ...)` returns the composed map directly!
    # why did the user use `compose_map`?
    
    # Re-reading user code carefully:
    # def compose_map(mapA, mapB): return [f.substitute(mapB) for f in mapA]
    #
    # F = vars6
    # F = compose_map(quad_thin_kick_map(F, +kq), F)
    #
    # This implies `quad_thin_kick_map(F, +kq)` returns a map (let's call it M_elem evaluated at F).
    # That is ALREADY the composition Element ∘ F.
    # Then `compose_map(..., F)` would substitute F INTO that? That would be (Element ∘ F) ∘ F ? That seems wrong.
    #
    # OR, maybe `quad_thin_kick_map` is meant to be called with `vars6` (canonical identity vars), returning M_elem.
    # Then `F = compose_map(M_elem, F)` computes M_elem ∘ F.
    #
    # Let's look at the user calls in `shock_tpsa_6d`:
    # F = vars6
    # F = compose_map(quad_thin_kick_map(F, +kq), F)
    #
    # If `quad_thin_kick_map` is called with F, it returns Element(F).
    # Then `compose_map` takes that result and substitutes F again?
    # That would mean Element(F(F)). This is definitely wrong if F is the map from s=0.
    
    # HYPOTHESIS: The user code might have a bug in the line `F = compose_map(quad_thin_kick_map(F, ...), F)`
    # OR I am misunderstanding `quad_thin_kick_map`.
    # `quad_thin_kick_map` takes `vars6`. If `vars6` is `F`, it returns `x_f, px_f - k*x_f, ...`.
    # This IS the new map. We don't need `compose_map` if we simply pass `F` to the element functions.
    #
    # HOWEVER, the provided code IS the "First Truth". I should probably reproduce it EXACTLY first.
    # But wait, `compose_map` with `F` twice?
    # `[f.substitute(F) for f in quad_thin_kick_map(F, +kq)]`
    # if `quad` returns `F[0]`, then `substitute(F)` makes it `F[0](F)`.
    # If F is identity (vars6), F(F) is F.
    # So for the first step it doesn't matter.
    # But for the second step:
    # F1 = Element1(Identity)
    # F2 = compose_map(Element2(F1), F1) -> Element2(F1)(F1) -> Element2(F1 ∘ F1). This is completely wrong.
    #
    # CORRECTION:
    # The standard way to track TPSA is:
    # F = vars6
    # F = element_map(F)
    # ...
    #
    # The user's code snippet has:
    # F = compose_map(drift6D_map(F, L), F)
    # This looks like he is passing F to the map generator AND composing.
    # IF the user intends `drift6D_map` to ALWAYS be called with `vars6` (identity), then:
    # M_drift = drift6D_map(vars6, L)
    # F = compose_map(M_drift, F)
    # This makes perfect sense: NewMap = DriftMap ∘ CurrentMap.
    
    # BUT, the code says: `drift6D_map(F, L)`.
    # If he passes F, he is already composing!
    #
    # Maybe the user made a typo in the provided snippet?
    # "Sotto hai un file singolo che ti dà questa base. È volutamente “duro”, non ottimizzato: prima verità, poi performance."
    # The user says "Here is the code, it's 'rough/hard'".
    #
    # Let's inspect `shock_tpsa_6d` in the USER REQUEST again carefully.
    # `F = compose_map(quad_thin_kick_map(F, +kq), F)`
    #
    # Maybe `compose_map` is NOT what I think it is?
    # `def compose_map(mapA, mapB): return [f.substitute(mapB) for f in mapA]`
    #
    # If I copy the code EXACTLY, I might propagate a bug.
    # But as an agent, I should fix obvious bugs if I can, OR reproduce and fail.
    # However, `quad_thin_kick_map(F, kq)` creates a TPSA with coefficients from F.
    # If F has coefficients, `substitute` will square them?
    #
    # Let's look at `substitute`:
    # `term = term * (subs[i] ** ei)`
    # It replaces variables with TPSA objects.
    #
    # If `drift6D_map(F, L)` is called:
    # It computes arithmetic on TPSAs in F. The result `res` is a TPSA expressing the drift output in terms of the INITIAL variables (since F is in terms of initial vars).
    # So `res` IS the updated map.
    # Then `compose_map(res, F)` takes `res` and substitutes variables with `F`.
    # But `res` is ALREADY in terms of initial variables (0,1,2,3,4,5).
    # `F` expresses current vars in terms of initial vars.
    # `res` expresses next vars in terms of initial vars.
    # If we substitute `vars` in `res` with `F`... we are replacing initial vars with the current map expressions?
    # That would mean $F_{new}(x_0) = F_{calculated}(F(x_0))$.
    # That means $x_{new} = x_{calc}(x_1) = x_{calc}(x_0(x_0))$.
    # This interprets `res` as a map from CURRENT to NEXT, but written in terms of INITIAL.
    # This is contradictory.
    #
    # If `drift6D_map` is called with `F`, it treats `F`'s values as the inputs.
    # So `res` is Map(F(x0)).
    # This IS the composition.
    #
    # So `compose_map(res, F)` is applying F AGAIN?
    #
    # I strongly suspect the user meant:
    # `F = compose_map(quad_thin_kick_map(vars6, +kq), F)`
    # OR
    # `F = quad_thin_kick_map(F, +kq)` (without compose_map)
    #
    # Given the snippet looks like a copy-paste "Truth", I should probably check if `vars6` is available in `shock_tpsa_6d` scope.
    # Yes, `vars6` is defined.
    #
    # If I run the code AS IS, and it works, maybe I'm crazy.
    # If `F` is identity initially.
    # Step 1: `kick(F, k)` -> Returns `M1(x)`. `compose(M1, F)` -> `M1(x(x))` -> `M1(x)`. Correct.
    # Step 2: `drift(F, L)` -> `F` is now `M1`. `drift` computes `M2(M1(x))`. This is `M_total`.
    #         `compose(M_total, F)` -> `M_total(M1(x))`.
    #         This effectively effectively applies M1 TWICE?
    #         `M_total` is polynomial of `x`. `F` is `M1(x)`.
    #         `substitute` replaces `x` with `M1(x)`.
    #         So we get `M2(M1(M1(x)))`.
    #
    # This looks like a BUG in the user's snippet. Double composition.
    # The user said "prima verità" (truth first).
    # Maybe he wants me to run it and see it fail or produce garbage?
    # OR I should fix it.
    #
    # I think `F = quad_thin_kick_map(F, ...)` is the Pythonic way to track if functions support TPSA arithmetic.
    # And `compose_map` is for when you have `Map(x)` and want `Map(OtherMap(x))`.
    #
    # I will write the file EXACTLY as provided first.
    # BUT, I will construct logic to correct it if it fails or if the error is high.
    #
    # Actually, looking at the code again:
    # `vars6` is local in `shock_tpsa_6d`.
    # The user passes `F` to the functions.
    #
    # I will stick to the user's code for `tpsa_tracker.py` but I'll add a comment or just run it.
    # Wait, if `sympl_error` turns out huge, I know why.
    #
    # Let's paste the code.
    
    # Wait, I see `vars6 = ...` at the top of `shock_tpsa_6d`.
    # The user calls: `F = compose_map(quad_thin_kick_map(F, +kq), F)`
    #
    # I'll paste it. If it's wrong, I'll fix it in verification.
    
    pass
