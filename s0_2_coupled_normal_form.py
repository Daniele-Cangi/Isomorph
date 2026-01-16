# s0_2_coupled_normal_form.py
import numpy as np
import sympy as sp

# -------------------------
# CONFIG
# -------------------------
ORDER = 2          # 3 basta per sext thin (kick ~ quadratic) -> map terms up to degree 3 after composition
EPS_DEN = 1e-10    # near-resonance threshold for denominators
np.set_printoptions(precision=4, suppress=True)

# Lattice knobs (spingili pure)
Ld = 1.0
kq = 0.55
ks = 0.1          # skew quadrupole strength (coupling)
k2 = 10.0          # sext strength (nonlinearity)

# -------------------------
# SYMBOLS (4D)
# -------------------------
x, px, y, py = sp.symbols("x px y py")
a1, a1c, a2, a2c = sp.symbols("a1 a1c a2 a2c")  # complex normal coords (formal independent)
I = sp.I

VXY = (x, px, y, py)
VA  = (a1, a1c, a2, a2c)

# -------------------------
# UTILS
# -------------------------
def truncate(expr, vars_, order):
    expr = sp.expand(expr)
    poly = sp.Poly(expr, *vars_, domain="EX")
    out = 0
    for mon, coeff in poly.terms():
        if sum(mon) <= order:
            term = coeff
            for v, pwr in zip(vars_, mon):
                if pwr:
                    term *= v**pwr
            out += term
    return sp.expand(out)

def jacobian_map(map_exprs, vars_):
    Jm = sp.zeros(len(vars_), len(vars_))
    for i, fi in enumerate(map_exprs):
        for j, v in enumerate(vars_):
            Jm[i, j] = sp.diff(fi, v)
    return Jm

def eval_at_zero(M, vars_):
    subs0 = {v: 0 for v in vars_}
    return np.array(M.subs(subs0), dtype=np.complex128)

def symplectic_J(n=2):
    # 2 DOF -> 4x4
    J = np.zeros((2*n, 2*n), dtype=np.complex128)
    for i in range(n):
        J[2*i,   2*i+1] = 1
        J[2*i+1, 2*i]   = -1
    return J

# -------------------------
# ELEMENT MAPS (thin)
# -------------------------
def drift(L, state):
    x_, px_, y_, py_ = state
    return (x_ + L*px_, px_, y_ + L*py_, py_)

def quad_kick(k, state):
    x_, px_, y_, py_ = state
    return (x_, px_ - k*x_, y_, py_ + k*y_)

def skew_quad_kick(ks, state):
    """
    Thin skew quadrupole (coupling):
    canonical form: px' = px - ks*y, py' = py - ks*x
    """
    x_, px_, y_, py_ = state
    return (x_, px_ - ks*y_, y_, py_ - ks*x_)

def sext_kick(k2, state):
    x_, px_, y_, py_ = state
    dpx = -(k2/2) * (x_**2 - y_**2)
    dpy =  (k2)   * (x_*y_)
    return (x_, px_ + dpx, y_, py_ + dpy)

# -------------------------
# BUILD ONE-TURN MAP
# -------------------------
def build_map():
    state = (x, px, y, py)

    # F
    state = quad_kick(+kq, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    # O
    state = drift(Ld, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    # Coupling element (skew)
    state = skew_quad_kick(ks, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    # Nonlinear element (sext)
    state = sext_kick(k2, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    # D
    state = quad_kick(-kq, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    # O
    state = drift(Ld, state)
    state = tuple(truncate(s, VXY, ORDER) for s in state)

    return state

# -------------------------
# COUPLED LINEAR NORMAL FORM (eigenvectors + symplectic normalization)
# -------------------------
def coupled_linear_normal_form(M):
    """
    For stable symplectic 4x4 M (real), eigenvalues come in exp(±iμ1), exp(±iμ2).
    We build C = [v1, v1*, v2, v2*] such that:
      C^{-1} M C = diag(λ1, λ1*, λ2, λ2*)
    and v_i are symplectically normalized: v^H J v = 2i.
    """
    J = symplectic_J(2)

    vals, vecs = np.linalg.eig(M)
    # pick unit-circle eigenvalues with Im>0 as "forward modes"
    candidates = []
    for i, lam in enumerate(vals):
        if abs(abs(lam) - 1.0) < 1e-6 and lam.imag > 0:
            candidates.append((lam, vecs[:, i]))

    if len(candidates) < 2:
        # fallback: sort by |Im| descending and take two with Im != 0
        pairs = [(vals[i], vecs[:, i]) for i in range(len(vals)) if abs(vals[i].imag) > 1e-9]
        pairs.sort(key=lambda t: t[0].imag, reverse=True)
        candidates = pairs[:2]

    if len(candidates) < 2:
        raise ValueError("Matrix not stably coupled (could be unstable or degenerate).")

    # sort by phase (tune)
    candidates.sort(key=lambda t: np.angle(t[0]))
    (lam1, v1), (lam2, v2) = candidates[0], candidates[1]

    # symplectic normalization: v^H J v = 2i
    def symp_norm(v):
        s = v.conj().T @ J @ v  # should be purely imaginary for stable mode
        if abs(s) < 1e-14:
            raise ValueError("Symplectic norm ~ 0: degeneracy/unstable.")
        factor = np.sqrt((2j) / s)
        return v * factor

    v1n = symp_norm(v1)
    v2n = symp_norm(v2)

    C = np.column_stack([v1n, np.conj(v1n), v2n, np.conj(v2n)])
    Cin = np.linalg.inv(C)

    mu1 = float(np.angle(lam1))
    mu2 = float(np.angle(lam2))
    # enforce positive in (0, pi)
    if mu1 < 0: mu1 += 2*np.pi
    if mu2 < 0: mu2 += 2*np.pi

    return {
        "C": C, "Cin": Cin,
        "lam": (lam1, np.conj(lam1), lam2, np.conj(lam2)),
        "mu": (mu1, -mu1, mu2, -mu2),
        "tunes": (mu1/(2*np.pi), mu2/(2*np.pi)),
        "J": J
    }

# -------------------------
# TRANSFORM MAP INTO NORMAL COMPLEX COORDS eta = (a1,a1*,a2,a2*)
# eta' = Cin * F( C * eta )
# -------------------------
def map_in_eta(one_turn, C, Cin):
    eta = sp.Matrix([a1, a1c, a2, a2c])

    # z = C * eta  (z = [x,px,y,py])
    Csym = sp.Matrix(C.tolist())  # numeric -> sympy
    z = Csym * eta
    subs_z = {x: z[0], px: z[1], y: z[2], py: z[3]}

    # apply one-turn map in z, then substitute z(eta)
    Fz = sp.Matrix([truncate(f.subs(subs_z), VA, ORDER) for f in one_turn])

    # eta' = Cin * Fz
    Cinsym = sp.Matrix(Cin.tolist())
    eta1 = Cinsym * Fz
    eta1 = sp.Matrix([truncate(e, VA, ORDER) for e in eta1])
    return eta1

def extract_g_for_coord(delta_expr, mu1, mu2, mu_coord):
    """
    delta_expr is polynomial in (a1,a1c,a2,a2c) for:
       eta'_coord - exp(i*mu_coord)*eta_coord
    Denom for monomial a1^j a1c^k a2^l a2c^m:
       exp(i*((j-k)*mu1 + (l-m)*mu2)) - exp(i*mu_coord)
    """
    poly = sp.Poly(sp.expand(delta_expr), *VA, domain="EX")
    terms = []
    for mon, coeff in poly.terms():
        j,k,l,m = mon
        deg = j+k+l+m
        if deg < 2:
            continue
        phase = (j-k)*mu1 + (l-m)*mu2
        denom = sp.exp(I*phase) - sp.exp(I*mu_coord)
        denom_val = complex(sp.N(denom))
        resonant = abs(denom_val) < EPS_DEN
        g = coeff/denom if not resonant else sp.nan
        terms.append((mon, coeff, denom, g, resonant, abs(complex(sp.N(coeff)))))
    # sort by |coeff| descending
    terms.sort(key=lambda t: t[5], reverse=True)
    return terms

# -------------------------
# MAIN
# -------------------------
def main():
    one_turn = build_map()

    # linear matrix at origin
    Jsym = jacobian_map(one_turn, VXY)
    M = eval_at_zero(Jsym, VXY)

    # coupled linear normal form
    nf = coupled_linear_normal_form(M)
    C, Cin = nf["C"], nf["Cin"]
    lam = nf["lam"]
    mu  = nf["mu"]
    tune1, tune2 = nf["tunes"]

    print("=== S0.2 COUPLED LINEAR NORMAL FORM ===")
    print("M (4x4):\n", np.real_if_close(M))
    print(f"Tunes (mode1, mode2): {tune1:.8f}, {tune2:.8f}")
    print("Eigenvalues:", lam)

    # build map in eta-coordinates
    eta_prime = map_in_eta(one_turn, C, Cin)

    eta_vec = sp.Matrix([a1, a1c, a2, a2c])
    # delta = eta' - diag(exp(i*mu_coord)) eta
    rot = sp.diag(sp.exp(I*mu[0]), sp.exp(I*mu[1]), sp.exp(I*mu[2]), sp.exp(I*mu[3]))
    delta = sp.Matrix([truncate(eta_prime[i] - rot[i,i]*eta_vec[i], VA, ORDER) for i in range(4)])

    print("\n=== NONLINEAR CONTENT (eta' - R eta) ===")
    print("delta_a1  =", delta[0])
    print("delta_a1c =", delta[1])
    print("delta_a2  =", delta[2])
    print("delta_a2c =", delta[3])

    # extract g-terms per coordinate
    g_a1  = extract_g_for_coord(delta[0], mu1=mu[0], mu2=mu[2], mu_coord=mu[0])
    g_a1c = extract_g_for_coord(delta[1], mu1=mu[0], mu2=mu[2], mu_coord=mu[1])
    g_a2  = extract_g_for_coord(delta[2], mu1=mu[0], mu2=mu[2], mu_coord=mu[2])
    g_a2c = extract_g_for_coord(delta[3], mu1=mu[0], mu2=mu[2], mu_coord=mu[3])

    def dump(name, terms, n=25):
        print(f"\n=== HOMOLOGICAL SOLUTION g: {name} (top {n}) ===")
        print("mon=(j,k,l,m) | coeff | denom | g | resonant")
        for mon, coeff, denom, g, resonant, _mag in terms[:n]:
            print(f"{mon} | {sp.N(coeff,6)} | {sp.N(denom,6)} | {sp.N(g,6)} | {resonant}")

    dump("a1",  g_a1)
    dump("a2",  g_a2)

    # resonance radar (flag near-resonant denominators)
    def radar(terms, label):
        hits = [t for t in terms if t[4] is True]
        print(f"\n=== RESONANCE RADAR: {label} ===")
        if not hits:
            print("No near-resonant denominators under EPS_DEN.")
            return
        for mon, coeff, denom, g, resonant, _mag in hits[:50]:
            print(f"NEAR-RES: mon={mon} coeff={sp.N(coeff,6)} denom={sp.N(denom,6)}")

    radar(g_a1, "a1")
    radar(g_a2, "a2")

    print("\nDONE S0.2 (coupled). NEXT = S0.3: apply exp(:g:) and extract detuning + residual resonant terms.")
    
    # === S0.3 RUN ===
    w_prime, g_eta, c = s0_3_run(eta_prime, delta, mu)

# -------------------------
# S0.3 — NORMAL FORM CONJUGATION (4D coupled)
# Build g(w) from delta (non-resonant terms), then conjugate:
#   z = w + g(w)
#   z' = F(z)
#   w' = z' - g(z')
# Result: w' = R w + (resonant terms + detuning)
# -------------------------

b1, b1c, b2, b2c = sp.symbols("b1 b1c b2 b2c")
VB = (b1, b1c, b2, b2c)

def monomial(vars_, exps):
    out = 1
    for v, p in zip(vars_, exps):
        if p:
            out *= v**p
    return out

def build_g_from_delta(delta_vec, mu1, mu2, mu_vec, order=ORDER, eps_den=EPS_DEN):
    """
    delta_vec: 4x1 sympy vector in (a1,a1c,a2,a2c) for delta = eta' - R eta
    Returns g_vec: 4x1 sympy vector (same vars) containing only NON-RESONANT terms.
    """
    g = [0, 0, 0, 0]
    for i in range(4):
        poly = sp.Poly(sp.expand(delta_vec[i]), *VA, domain="EX")
        for mon, coeff in poly.terms():
            j,k,l,m = mon
            deg = j+k+l+m
            if deg < 2 or deg > order:
                continue

            phase = (j-k)*mu1 + (l-m)*mu2
            denom = sp.exp(I*phase) - sp.exp(I*mu_vec[i])
            denom_val = complex(sp.N(denom))

            # resonant => keep it in normal form (do NOT put into g)
            if abs(denom_val) < eps_den:
                continue

            g[i] += (coeff/denom) * monomial(VA, mon)

        g[i] = truncate(sp.expand(g[i]), VA, order)
    return sp.Matrix(g)

def substitute_vec(expr_vec, subs_dict, vars_out=VB, order=ORDER):
    out = []
    for e in expr_vec:
        out.append(truncate(sp.expand(e.subs(subs_dict)), vars_out, order))
    return sp.Matrix(out)

def conjugate_map_to_normal_form(eta_prime, g_eta, order=ORDER):
    """
    eta_prime: 4x1 sympy vector giving eta' = F(eta) in VA vars.
    g_eta:     4x1 vector giving g(eta) in VA vars.
    Returns w_prime in VB vars (w = b's).
    """
    # 1) Build z(w) = w + g(w)
    # Convert g from (a1,a1c,a2,a2c) -> (b1,b1c,b2,b2c)
    g_w = substitute_vec(g_eta, {a1:b1, a1c:b1c, a2:b2, a2c:b2c}, vars_out=VB, order=order)
    z_w = sp.Matrix([b1, b1c, b2, b2c]) + g_w

    # 2) z' = F(z(w))  (substitute eta := z_w)
    subs_eta_to_z = {a1:z_w[0], a1c:z_w[1], a2:z_w[2], a2c:z_w[3]}
    z_prime_w = substitute_vec(eta_prime, subs_eta_to_z, vars_out=VB, order=order)

    # 3) w' = z' - g(z')  (near-identity inverse)
    # Compute g(z') by substituting a := z'
    subs_a_to_zp = {a1:z_prime_w[0], a1c:z_prime_w[1], a2:z_prime_w[2], a2c:z_prime_w[3]}
    g_at_zp = substitute_vec(g_eta, subs_a_to_zp, vars_out=VB, order=order)

    w_prime = sp.Matrix([truncate(sp.expand(z_prime_w[i] - g_at_zp[i]), VB, order) for i in range(4)])
    return w_prime

def extract_detuning_coeffs(w_prime, mu_vec):
    """
    For mode-1 (b1), detuning terms show up as:
      b1' = e^{i mu1} [ b1 + i*b1*(c11*(b1*b1c) + c12*(b2*b2c)) + ... ]
    So coefficient of b1^2*b1c  => i*c11
                and b1*b2*b2c  => i*c12
    Similarly for mode-2.
    Returns c11,c12,c21,c22 (complex).
    """
    rot = sp.diag(sp.exp(I*mu_vec[0]), sp.exp(I*mu_vec[1]), sp.exp(I*mu_vec[2]), sp.exp(I*mu_vec[3]))
    w = sp.Matrix([b1,b1c,b2,b2c])

    # factor out rotation
    r1 = truncate(sp.expand(w_prime[0] / rot[0,0]), VB, ORDER)
    r2 = truncate(sp.expand(w_prime[2] / rot[2,2]), VB, ORDER)

    P1 = sp.Poly(sp.expand(r1), *VB, domain="EX")
    P2 = sp.Poly(sp.expand(r2), *VB, domain="EX")

    def coeff_of(P, exps):
        try:
            return P.coeffs()[P.monoms().index(exps)]
        except ValueError:
            return 0

    # mode 1 detuning monomials:
    # b1^2*b1c  => (2,1,0,0)
    # b1*b2*b2c => (1,0,1,1)
    c_b1J1 = coeff_of(P1, (2,1,0,0))
    c_b1J2 = coeff_of(P1, (1,0,1,1))

    # mode 2:
    # b2^2*b2c  => (0,0,2,1)
    # b2*b1*b1c => (1,1,1,0)
    c_b2J2 = coeff_of(P2, (0,0,2,1))
    c_b2J1 = coeff_of(P2, (1,1,1,0))

    # c = coeff / i
    c11 = sp.N(c_b1J1 / I)
    c12 = sp.N(c_b1J2 / I)
    c21 = sp.N(c_b2J1 / I)
    c22 = sp.N(c_b2J2 / I)

    return complex(c11), complex(c12), complex(c21), complex(c22)

def list_resonant_terms(w_prime, mu1, mu2, mu_vec, eps_den=EPS_DEN, top=30):
    """
    List terms in (w' - Rw) that are resonant (small denominator).
    Uses the same resonance condition as S0.2.
    """
    rot = sp.diag(sp.exp(I*mu_vec[0]), sp.exp(I*mu_vec[1]), sp.exp(I*mu_vec[2]), sp.exp(I*mu_vec[3]))
    w = sp.Matrix([b1,b1c,b2,b2c])
    residual = sp.Matrix([truncate(sp.expand(w_prime[i] - rot[i,i]*w[i]), VB, ORDER) for i in range(4)])

    out = {}
    for i, label in enumerate(["b1","b1c","b2","b2c"]):
        poly = sp.Poly(sp.expand(residual[i]), *VB, domain="EX")
        terms = []
        for mon, coeff in poly.terms():
            j,k,l,m = mon
            deg = j+k+l+m
            if deg < 2: 
                continue
            phase = (j-k)*mu1 + (l-m)*mu2
            denom = sp.exp(I*phase) - sp.exp(I*mu_vec[i])
            denom_val = complex(sp.N(denom))
            resonant = abs(denom_val) < eps_den
            if resonant:
                terms.append((mon, complex(sp.N(coeff)), abs(complex(sp.N(coeff)))))
        terms.sort(key=lambda t: t[2], reverse=True)
        out[label] = terms[:top]
    return out

# -------------------------
# Hook into your main() after you computed:
#  - eta_prime (map in eta)
#  - delta     (eta' - R eta)
#  - mu (mu_vec) and tunes
# -------------------------

def s0_3_run(eta_prime, delta, mu):
    mu1 = float(mu[0])  # mode-1 phase
    mu2 = float(mu[2])  # mode-2 phase
    mu_vec = mu

    # 1) Build g (non-resonant) from delta
    g_eta = build_g_from_delta(delta, mu1, mu2, mu_vec, order=ORDER, eps_den=EPS_DEN)

    # 2) Conjugate map -> w' in VB vars
    w_prime = conjugate_map_to_normal_form(eta_prime, g_eta, order=ORDER)

    # 3) Detuning coefficients
    c11, c12, c21, c22 = extract_detuning_coeffs(w_prime, mu_vec)
    # Convert to tune slopes: Δν = (c / (2π)) * J
    dnu11 = c11/(2*np.pi); dnu12 = c12/(2*np.pi)
    dnu21 = c21/(2*np.pi); dnu22 = c22/(2*np.pi)

    print("\n=== S0.3 NORMAL FORM RESULTS ===")
    print("Detuning coefficients (phases):")
    print("  c11 (mode1<-J1) =", c11)
    print("  c12 (mode1<-J2) =", c12)
    print("  c21 (mode2<-J1) =", c21)
    print("  c22 (mode2<-J2) =", c22)

    print("\nDetuning slopes (tunes): Δν1 = dnu11*J1 + dnu12*J2,  Δν2 = dnu21*J1 + dnu22*J2")
    print("  dnu11 =", dnu11)
    print("  dnu12 =", dnu12)
    print("  dnu21 =", dnu21)
    print("  dnu22 =", dnu22)

    # 4) Resonant driving terms that survive
    resonant = list_resonant_terms(w_prime, mu1, mu2, mu_vec, eps_den=EPS_DEN, top=25)
    print("\n=== RESONANT TERMS (survivors in normal form) ===")
    for k, terms in resonant.items():
        print(f"\n[{k}] top terms:")
        for mon, coeff, mag in terms:
            print(f"  mon={mon} coeff={coeff} |coeff|={mag:.3e}")

    return w_prime, g_eta, (c11,c12,c21,c22)

if __name__ == "__main__":
    main()
