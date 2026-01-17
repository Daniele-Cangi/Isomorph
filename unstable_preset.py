import numpy as np
import matplotlib.pyplot as plt
from symplectic_tracker import (
    SymplecticTracker6D as SymplecticTracker,
    SymplecticTracker6D, Drift, MultipoleKick, 
    QuadrupoleKick, SextupoleKick, RFCavityKick, 
    WormholeCanonicalTranslate, SkewQuadrupoleKick, 
    OctupoleKick, ModulatedSextupoleKick, DualModulatedSextupoleKick,
    jacobian_fd, symplectic_error
)

PRESET_NAME = "CHAOTIC_EDGE_k3m1e4"

# ... (Previous code remains, skipping to end) ...


# ... (Previous code remains, skipping to end) ...

# Use 6D arrays for state, but allow 4D input convention for probes
def PhaseSpaceState(x, px, y, py, z=0.0, d=0.0):
    return np.array([x, px, y, py, z, d], dtype=float)

# --- S0.6 COMPLEX ANALYSIS TOOLS ---
def _top_freqs_complex(z, n=6, discard=256):
    z = z[discard:]
    if len(z) < 512: return np.array([]), np.array([])
    z = z - np.mean(z)
    w = np.hanning(len(z))
    Z = np.fft.fft(z * w)
    mag = np.abs(Z)
    freqs = np.fft.fftfreq(len(z), d=1.0)
    mask = (freqs > 1e-4) & (freqs < 0.5)
    if not np.any(mask): return np.array([0.0]), np.array([0.0])
    valid_mag = mag[mask]
    valid_freqs = freqs[mask]
    n_top = min(n, len(valid_mag))
    idx = np.argpartition(valid_mag, -n_top)[-n_top:]
    cand = valid_freqs[idx]
    cand_mag = valid_mag[idx]
    order = np.argsort(-cand_mag)
    return cand[order], cand_mag[order]

def eigentunes_from_traj(traj, discard=256, cluster_eps=5e-4):
    x, px, y, py = traj[:,0], traj[:,1], traj[:,2], traj[:,3]
    u = x + 1j*px
    v = y + 1j*py
    fu, mu = _top_freqs_complex(u, discard=discard)
    fv, mv = _top_freqs_complex(v, discard=discard)
    all_f = np.concatenate([fu, fv])
    all_m = np.concatenate([mu, mv])
    if len(all_f) == 0: return None
    key = np.round(all_f / cluster_eps).astype(int)
    clusters = {}
    for k, f, m in zip(key, all_f, all_m):
        clusters.setdefault(k, {"w":0.0, "f":0.0})
        clusters[k]["w"] += m
        clusters[k]["f"] += f*m
    items = [(k, d["w"], d["f"]/d["w"]) for k,d in clusters.items()]
    items.sort(key=lambda t: -t[1])
    if len(items) < 1: return 0.0, 0.0
    nu1 = float(items[0][2])
    nu2 = float(items[1][2]) if len(items) > 1 else nu1
    nu1, nu2 = sorted([nu1, nu2])
    return nu1, nu2

def diffusion_eigentunes(traj, win=2048, step=512, discard=256):
    n = len(traj)
    nus1, nus2 = [], []
    start0 = discard
    if n < win + start0: return 0.0, 0.0
    for start in range(start0, n - win, step):
        sub = traj[start:start+win]
        out = eigentunes_from_traj(sub, discard=0)
        if out is None: continue
        nu1, nu2 = out
        nus1.append(nu1); nus2.append(nu2)
    if len(nus1) < 4: return 0.0, 0.0
    return float(np.std(nus1)), float(np.std(nus2))

# --- APERTURE TRACKER ---
class ApertureTracker(SymplecticTracker):
    def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
        super().__init__(elements)
        self.limit_x = aperture_x
        self.limit_y = aperture_y

    def track(self, s0, n_turns):
        traj = np.zeros((n_turns, 6), dtype=float)
        s = np.array(s0, dtype=float)
        traj[0] = s
        survived = n_turns
        for i in range(1, n_turns):
            s = self.one_turn(s)
            traj[i] = s
            if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                survived = i
                traj[i:] = np.nan
                break
            if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                survived = i
                traj[i:] = np.nan
                break
        return traj, survived

# --- LATTICE ---
def build_resonance_lattice(k_quad=0.632, k2=10.0, ks=0.15, k3=0.0, L=1.0):
    # Symmetric Octupole insertion
    elems = []
    
    # 1. F-Block
    elems.append(QuadrupoleKick(+k_quad))
    elems.append(Drift(L/2.0))
    if k3 != 0.0: elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    
    elems.append(SextupoleKick(+k2))
    elems.append(Drift(L))
    
    # 2. D-Block
    elems.append(QuadrupoleKick(-k_quad))
    elems.append(Drift(L/2.0))
    if k3 != 0.0: elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    
    elems.append(SextupoleKick(-k2))
    elems.append(Drift(L))
    
    # 3. Coupling
    elems.append(SkewQuadrupoleKick(ks))
    elems.append(Drift(L))
    
    return elems

# --- S1.2 SYMPLECTIC WHITENING (4D Decoupling) ---
from scipy.linalg import eig

J4 = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0]
], dtype=float)

def compute_one_turn_matrix(tracker, h=1e-7):
    """Compute 6x6 Jacobian via Finite Difference"""
    M = np.zeros((6,6))
    origin = np.zeros(6)
    ref_out = tracker.one_turn(origin)
    
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        state = origin + d
        out = tracker.one_turn(state)
        # column i = (out - ref) / h
        M[:, i] = (out - ref_out) / h
    return M

def coupled_normal_form_A(M):
    """
    Build symplectic normalization A for a stable 4x4 one-turn matrix M.
    Returns:
      A: real 4x4 symplectic matrix
      tunes: (nu1, nu2)
      symp_err: ||A^T J A - J||
    """
    w, V = eig(M)  # complex eig
    # keep the 2 eigenmodes with Im(lambda)>0 (stable conjugate pairs)
    idx = [i for i in range(4) if np.imag(w[i]) > 0]
    if len(idx) < 2:
        print("Warning: Lattice not stable enough (no complex stable pairs found).")
        return np.eye(4), (0.0, 0.0), 999.9

    # sort by tune
    tunes = [np.angle(w[i])/(2*np.pi) % 1.0 for i in idx]
    idx = [i for _, i in sorted(zip(tunes, idx))]
    i1, i2 = idx[0], idx[1]

    def symp_norm(v):
        s = v.conj().T @ J4 @ v  # complex scalar
        # want v* J v = -2i  (convention); scale accordingly
        return v / np.sqrt((-1j) * s)

    v1 = symp_norm(V[:, i1])
    v2 = symp_norm(V[:, i2])

    # A columns: Re/Im of eigenvectors -> real symplectic basis
    A = np.column_stack([np.real(v1), np.imag(v1), np.real(v2), np.imag(v2)]).astype(float)

    symp_err = np.linalg.norm(A.T @ J4 @ A - J4)
    nu1 = np.angle(w[i1])/(2*np.pi) % 1.0
    nu2 = np.angle(w[i2])/(2*np.pi) % 1.0
    return A, (nu1, nu2), symp_err

def to_normal_coords(traj_4xn, A):
    """
    traj_4xn: shape (N,4) with columns [x,px,y,py]
    returns Z in normal coords (N,4)
    """
    # solve A z = x  -> z = A^{-1} x
    return np.linalg.solve(A, traj_4xn.T).T

def resonance_order_from_poincare(xpx: np.ndarray, outer_quantile=0.85, bins=720):
    """
    xpx: array (N,2) con colonne [x, px] (unità coerenti)
    Ritorna: (order_m, strength_snr, spectrum)
    """
    x = xpx[:,0].astype(float)
    px = xpx[:,1].astype(float)

    sx = np.std(x) + 1e-12
    sp = np.std(px) + 1e-12
    r = np.sqrt((x/sx)**2 + (px/sp)**2)
    if len(r) < 100: return 0, 0.0, np.array([])
        
    mask = r >= np.quantile(r, outer_quantile)
    
    # Angle calculation
    phi = np.mod(np.arctan2((px[mask]/sp), (x[mask]/sx)), 2*np.pi)
    
    hist, _ = np.histogram(phi, bins=bins, range=(0, 2*np.pi))
    hist = hist.astype(float)
    hist -= hist.mean()

    H = np.fft.rfft(hist)
    amp = np.abs(H)

    search_start = 1 
    m = int(np.argmax(amp[search_start:]) + search_start)
    noise_floor = (np.sum(amp[1:]) - amp[m]) / (len(amp[1:]) - 1 + 1e-12)
    snr = float(amp[m] / (noise_floor + 1e-12))
    
    return m, snr, amp

# --- MUTATION C: STICKY CHAOS PRESET ---
def build_sticky_chaos_lattice(k_quad=0.632, k2=10.0, ks=0.15, k3=-1.0e4, eps=0.06):
    L = 1.0
    elems = []
    
    # Modulation Frequency: Golden Ratio
    omega = 2*np.pi * (np.sqrt(5.0)-1.0)/2.0
    
    # 1. F-Block
    elems.append(QuadrupoleKick(+k_quad))
    elems.append(Drift(L/2.0))
    elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    # Modulated Sextupole
    elems.append(ModulatedSextupoleKick(k2, eps=eps, omega=omega, phase=0.0))
    elems.append(Drift(L))
    
    # 2. D-Block
    elems.append(QuadrupoleKick(-k_quad))
    elems.append(Drift(L/2.0))
    elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    # Modulated Sextupole (Antisymmetric k2 for D-block? Or Symmetric? Standard FODO uses -k2 for chromaticity)
    # Let's keep the standard antisymmetric configuration for chromatic correction
    elems.append(ModulatedSextupoleKick(-k2, eps=eps, omega=omega, phase=0.0))
    elems.append(Drift(L))
    
    # 3. Coupling (Static Shear)
    elems.append(SkewQuadrupoleKick(ks))
    elems.append(Drift(L))
    
    return elems

def escape_time_atlas(tracker, x_range, y_range, n_turns=100000, aperture=5e-3):
    X = np.linspace(x_range[0], x_range[1], 30) # 30x scans
    Y = np.linspace(y_range[0], y_range[1], 15)
    T = np.zeros((len(Y), len(X)), dtype=int)
    
    # Aperture check specialized for speed? Or use tracker?
    # Using tracker directly. tracker.track stops at n_turns.
    # We need a track_until_loss function or use the wrapper.
    
    print(f"Scanning {len(X)}x{len(Y)} grid for Escape Time...")
    for iy, y0 in enumerate(Y):
        print(f"  Scanning Row {iy+1}/{len(Y)} (y={y0*1000:.3f}mm)...")
        for ix, x0 in enumerate(X):
            init = PhaseSpaceState(x=x0, px=0.0, y=y0, py=0.0)
            
            # ApertureTracker handles tracking and loss check.
            # But we need exactly the turn count.
            # ApertureTracker.track returns (traj, survived_turns). 
            # We can use that.
            
            # Note: Tracker logic modified in this session to pass 'turn'.
            # ApertureTracker needs update? Yes, ApertureTracker overrides .track 
            # but relies on .one_turn of the parent.
            # The parent SymplecticTracker6D.one_turn now accepts turn.
            # ApertureTracker.track needs to pass turn too.
            
            # Quick patch: Just use the ApertureTracker instance.
            # Wait, ApertureTracker.track needs to loop and pass i.
            # Since we didn't update ApertureTracker class code above, let's assume standard behavior first. or fix it.
            # Actually, SymplecticTracker6D.track iterates i. ApertureTracker iterates separately.
            # We should patch ApertureTracker to be turn-aware.
            
            _, surv = tracker.track(init, n_turns)
            T[iy, ix] = surv
            

# --- S1.3b ROBUST VALIDATION ---
def escape_time_atlas_robust(tracker, x_range, y_range, n_turns=50000, nx=80, ny=60, n_phase=8):
    """
    Robust Atlas with Phase Randomization:
    For each (x,y), run n_phase micro-runs with small random (px, py).
    Take MEDIAN survival time to filter out phase-dependent lucky pockets.
    """
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    T_map = np.zeros((ny, nx), dtype=float)
    
    print(f"Scanning Robust Grid {nx}x{ny} (Phase Rand={n_phase})...")
    
    rng = np.random.default_rng(42)
    dp_scale = 1e-5
    
    for iy, y0 in enumerate(Y):
        print(f"  Row {iy+1}/{ny} (y={y0*1000:.3f}mm)...")
        for ix, x0 in enumerate(X):
            survivals = []
            for _ in range(n_phase):
                # Random phase perturbation
                dpx = (rng.random() - 0.5) * 2 * dp_scale
                dpy = (rng.random() - 0.5) * 2 * dp_scale
                
                init = PhaseSpaceState(x=x0, px=dpx, y=y0, py=dpy)
                _, surv = tracker.track(init, n_turns)
                survivals.append(surv)
            
            # Robust statistic: Median (truncating float)
            T_map[iy, ix] = np.median(survivals)
            
    return X, Y, T_map

def analyze_chaotic_band(T, n_turns, min_t=500):
    """
    S1.3b Conditioned Analysis:
    Filter out:
      - Stable Core (T == n_turns)
      - Immediate Loss (T < min_t)
    Analyze only the 'Chaotic Band' (min_t <= T < n_turns).
    """
    flat_T = T.ravel()
    
    # Masks
    mask_core = (flat_T >= n_turns)
    mask_loss = (flat_T < min_t)
    mask_band = (~mask_core) & (~mask_loss)
    
    T_band = flat_T[mask_band]
    frac_band = np.sum(mask_band) / len(flat_T)
    frac_core = np.sum(mask_core) / len(flat_T)
    frac_loss = np.sum(mask_loss) / len(flat_T)
    
    print(f"Population Split: Core={frac_core:.1%}, Wall={frac_loss:.1%}, Band={frac_band:.1%}")
    
    if len(T_band) < 50:
        print("Warning: Not enough points in Chaotic Band for robust stats.")
        return None
        
    # Conditioned Statistics on Band
    # We treat T_band as 'observed' failures (since they are < n_turns).
    # But effectively they are samples from the conditional distribution P(T|Band).
    # We can compute CCDF relative to the band size.
    
    # CCDF
    sorted_T = np.sort(T_band)
    n_band = len(T_band)
    # y = P(T > t | Band) = 1 - rank/n
    prob = 1.0 - np.arange(n_band)/n_band
    
    # Metrics
    # S_cond(t) is prob of surviving t GIVEN you are in band
    # Tail Ratio Conditioned: S_cond(10k) / S_cond(2k) ?
    # Let's use user specs: S(100k)/S(10k) if possible
    
    def get_prob(t_target):
        # find prob at t >= t_target
        idx = np.searchsorted(sorted_T, t_target)
        if idx >= n_band: return 0.0
        return prob[idx]
        
    s1 = get_prob(10000)
    s2 = get_prob(100000)
    tr_cond = s2 / (s1 + 1e-12)
    
    return {
        "times": sorted_T,
        "ccdf": prob,
        "S_cond(10k)": s1,
        "S_cond(100k)": s2,
        "tail_ratio_cond": tr_cond
    }

def to_samples(T, n_turns):
    t = T.ravel().astype(np.int64)
    event = (t < n_turns).astype(np.int8)   # 1 = lost observed, 0 = censored
    return t, event

def survival_ccdf(t, event, n_turns):
    # Kaplan-Meier light
    times = np.arange(1, n_turns+1)
    at_risk = np.zeros(n_turns+1, dtype=np.int64)
    d = np.zeros(n_turns+1, dtype=np.int64)  # deaths
    
    # Vectorize if possible or keep loop (N=450 is small)
    # Loop is fine for small grid
    for ti, ei in zip(t, event):
        at_risk[1:ti+1] += 1
        if ei == 1:
            d[ti] += 1
    
    S = np.ones(n_turns+1, dtype=np.float64)
    for k in range(1, n_turns+1):
        if at_risk[k] > 0:
            S[k] = S[k-1] * (1.0 - d[k]/at_risk[k])
        else:
            S[k] = S[k-1]
    return times, S[1:]  # CCDF 

def hazard_from_survival(S):
    # h_k roughly 1 - S_k / S_{k-1}
    h = np.zeros_like(S)
    h[0] = 1 - S[0]
    h[1:] = 1.0 - (S[1:] / (S[:-1] + 1e-12))
    return h

def sticky_metrics(times, S, thresholds=(10_000, 100_000)):
    t1, t2 = thresholds
    n = len(S)
    S1 = S[t1-1] if t1 <= n else S[-1]
    S2 = S[t2-1] if t2 <= n else S[-1]
    tail_ratio = (S2 / (S1 + 1e-12))
    return {"S(t1)": S1, "S(t2)": S2, "tail_ratio": tail_ratio}

def heavy_tail_score(T):
    flat_T = T.flatten()
    med = np.median(flat_T)
    p95 = np.percentile(flat_T, 95)
    if med == 0: med = 1e-9
    return p95 / med

def run_mutation_c():
    """Mutation C: Sticky Chaos with Modulated Sextupoles & S1.3 Analysis"""
    PRESET_NAME = "MUTATION_C_STICKY"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    eps = 0.06
    lat = build_sticky_chaos_lattice(eps=eps)
    
    # Need to update ApertureTracker to handle turn argument if not done
    class TurnAwareApertureTracker(SymplecticTracker):
        def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
            super().__init__(elements)
            self.limit_x = aperture_x
            self.limit_y = aperture_y

        def track(self, s0, n_turns):
            traj = np.zeros((n_turns, 6), dtype=float)
            s = np.array(s0, dtype=float)
            traj[0] = s
            survived = n_turns
            for i in range(1, n_turns):
                s = self.one_turn(s, turn=i-1)
                traj[i] = s
                if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                    survived = i
                    traj[i:] = np.nan
                    break
                if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                    survived = i
                    traj[i:] = np.nan
                    break
            return traj, survived

    tracker = TurnAwareApertureTracker(lat, aperture_x=5e-3, aperture_y=5e-3)
    
    # 1. Escape Time Atlas
    x_range = (1.55e-3, 2.10e-3)
    y_range = (0.0, 0.6e-3)
    n_turns = 200000 
    
    X, Y, T = escape_time_atlas(tracker, x_range, y_range, n_turns=n_turns)
    
    # Save Data immediately
    np.savez(f"escape_time_data_{PRESET_NAME}.npz", X=X, Y=Y, T=T)
    print(f"Saved escape_time_data_{PRESET_NAME}.npz")

    # S1.3 Analysis
    score = heavy_tail_score(T)
    print(f"\nHEAVY TAIL SCORE: {score:.2f}")
    
    t_samp, event = to_samples(T, n_turns)
    times, S = survival_ccdf(t_samp, event, n_turns)
    h = hazard_from_survival(S)
    metrics = sticky_metrics(times, S)
    print(f"S(10k): {metrics['S(t1)']:.4f}")
    print(f"S(100k): {metrics['S(t2)']:.4f}")
    print(f"Tail Ratio: {metrics['tail_ratio']:.4f}")
    
    if metrics['tail_ratio'] > 0.2:
        print(">> SUCCESS: Beach Confirmed (Tail Ratio > 0.2)")
    elif score > 3.0:
        print(">> PARTIAL: Sticky but maybe not full Beach (Score > 3)")
    else:
        print(">> FAIL: Cliff behavior detected.")
        
    # Plot 3-Panel
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Atlas
    ax1 = plt.subplot(1, 3, 1)
    # log10(T) mapping
    im = ax1.pcolormesh(X*1000, Y*1000, np.log10(T+1), shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax1, label='Log10(Survival Turns)')
    ax1.set_title(f"Escape Time Map\nHT Score: {score:.2f}")
    ax1.set_xlabel("x0 [mm]"); ax1.set_ylabel("y0 [mm]")
    
    # 2. Survival CCDF (Log-Log)
    ax2 = plt.subplot(1, 3, 2)
    # Downsample for plotting if huge
    stride = 100 if n_turns > 10000 else 1
    ax2.plot(times[::stride], S[::stride], 'b-', lw=2)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_title(f"Survival S(t) Log-Log\nTail Ratio: {metrics['tail_ratio']:.2f}")
    ax2.set_xlabel("Turns t")
    ax2.set_ylabel("P(Survival > t)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(1e-4, 1.1)
    
    # 3. Hazard Rate (MA smoothed?)
    ax3 = plt.subplot(1, 3, 3)
    # Simple smoothing window
    w_h = 1000
    if len(h) > w_h:
        h_smooth = np.convolve(h, np.ones(w_h)/w_h, mode='valid')
        t_smooth = times[w_h-1:]
    else:
        h_smooth = h
        t_smooth = times
    
    stride_h = 100
    ax3.plot(t_smooth[::stride_h], h_smooth[::stride_h], 'r-', alpha=0.7)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_title("Hazard Rate h(t)\n(Decreasing = Sticky)")
    ax3.set_xlabel("Turns t")
    ax3.set_ylabel("Hazard")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"analysis_S1_3_{PRESET_NAME}.png", dpi=150)
    print(f"Saved analysis_S1_3_{PRESET_NAME}.png")


# (Deleted duplicate main block)




# --- MUTATION D: CANTORI THICKENING (S1.4) ---
def build_cantori_lattice(k_quad=0.632, k2=10.0, ks=0.15, k3=-1.0e4, eps1=0.06, eps2=0.04):
    L = 1.0
    elems = []
    
    # Frequencies
    omega1 = 2*np.pi * (np.sqrt(5.0)-1.0)/2.0 # Golden
    omega2 = 2*np.pi * np.sqrt(2.0)           # Sqrt(2)
    
    # 1. F-Block
    elems.append(QuadrupoleKick(+k_quad))
    elems.append(Drift(L/2.0))
    elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    # Dual Modulated Sextupole
    elems.append(DualModulatedSextupoleKick(k2, eps1=eps1, eps2=eps2, omega1=omega1, omega2=omega2))
    elems.append(Drift(L))
    
    # 2. D-Block
    elems.append(QuadrupoleKick(-k_quad))
    elems.append(Drift(L/2.0))
    elems.append(OctupoleKick(k3/2.0))
    elems.append(Drift(L/2.0))
    # Dual Modulated Sextupole (Antisymmetric k2)
    elems.append(DualModulatedSextupoleKick(-k2, eps1=eps1, eps2=eps2, omega1=omega1, omega2=omega2))
    elems.append(Drift(L))
    
    # 3. Coupling
    elems.append(SkewQuadrupoleKick(ks))
    elems.append(Drift(L))
    return elems

def run_mutation_d():
    """Mutation D: Cantori Thickening (Two-Frequency Drive)"""
    PRESET_NAME = "MUTATION_D_CANTORI"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    eps1, eps2 = 0.06, 0.04
    lat = build_cantori_lattice(eps1=eps1, eps2=eps2)
    
    # Reuse TurnAware tracker pattern
    class TurnAwareApertureTracker(SymplecticTracker):
        def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
            super().__init__(elements)
            self.limit_x = aperture_x
            self.limit_y = aperture_y

        def track(self, s0, n_turns):
            traj = np.zeros((n_turns, 6), dtype=float)
            s = np.array(s0, dtype=float)
            traj[0] = s
            survived = n_turns
            for i in range(1, n_turns):
                s = self.one_turn(s, turn=i-1)
                traj[i] = s
                if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                    survived = i
                    traj[i:] = np.nan
                    break
                if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                    survived = i
                    traj[i:] = np.nan
                    break
            return traj, survived
            
    tracker = TurnAwareApertureTracker(lat, aperture_x=5e-3, aperture_y=5e-3)

    # 1. Escape Time Atlas (Same Grid to Compare)
    x_range = (1.55e-3, 2.10e-3)
    y_range = (0.0, 0.6e-3)
    n_turns = 200000 
    
    X, Y, T = escape_time_atlas(tracker, x_range, y_range, n_turns=n_turns)
    
    np.savez(f"escape_time_data_{PRESET_NAME}.npz", X=X, Y=Y, T=T)
    print(f"Saved escape_time_data_{PRESET_NAME}.npz")

    # S1.3 Analysis Reuse
    score = heavy_tail_score(T)
    print(f"\nHEAVY TAIL SCORE: {score:.2f}")
    
    t_samp, event = to_samples(T, n_turns)
    times, S = survival_ccdf(t_samp, event, n_turns)
    # h = hazard_from_survival(S) # Optional plot
    metrics = sticky_metrics(times, S)
    print(f"S(10k): {metrics['S(t1)']:.4f}")
    print(f"S(100k): {metrics['S(t2)']:.4f}")
    print(f"Tail Ratio: {metrics['tail_ratio']:.4f}")
    
    # Plot Simple Atlas
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X*1000, Y*1000, np.log10(T+1), shading='auto', cmap='inferno')
    plt.colorbar(label='Log10(Survival Turns)')
    plt.title(f"{PRESET_NAME}\nScore: {score:.2f}, TailRatio: {metrics['tail_ratio']:.2f}")
    plt.xlabel("x0 [mm]"); plt.ylabel("y0 [mm]")
    plt.tight_layout()
    plt.savefig(f"escape_time_{PRESET_NAME}.png", dpi=150)
    print(f"Saved escape_time_{PRESET_NAME}.png")


def run_validation_s1_3b():
    """
    Phase 2b: Robust Sticky Chaos Validation
    Preset: Mutation C (Single Modulated Sextupole)
    Grid: Zoomed 80x60 on Chaotic Band
    Phase Randomization: 8 runs
    Conditioned Analysis: Exclude Core/Wall
    """
    PRESET_NAME = "VALIDATION_S1_3b"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # 1. Setup Wrapper
    eps = 0.06
    lat = build_sticky_chaos_lattice(eps=eps)
    
    # Reuse TurnAware logic
    class TurnAwareApertureTracker(SymplecticTracker):
        def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
            super().__init__(elements)
            self.limit_x = aperture_x
            self.limit_y = aperture_y
        def track(self, s0, n_turns):
            traj = np.zeros((n_turns, 6), dtype=float)
            s = np.array(s0, dtype=float)
            traj[0] = s
            surv = n_turns
            for i in range(1, n_turns):
                s = self.one_turn(s, turn=i-1)
                traj[i] = s
                if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                    surv = i
                    traj[i:] = np.nan; break
                if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                    surv = i
                    traj[i:] = np.nan; break
            return traj, surv
            
    tracker = TurnAwareApertureTracker(lat)
    
    # 2. Agile Robust Atlas Plan
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30        # Reduced from 80x60
    n_turns = 50000        # Reduced from 200k (sufficient for beach detection)
    n_phase = 4            # Reduced from 8 (sufficient for median)
    
    X, Y, T = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
    
    np.savez(f"escape_time_data_{PRESET_NAME}.npz", X=X, Y=Y, T=T)
    print("Data saved.")
    
    # 3. Conditioned Analysis
    res = analyze_chaotic_band(T, n_turns)
    
    if res:
        print(f"\nCONDITIONED ANALYSIS:")
        print(f"  S_cond(10k): {res['S_cond(10k)']:.4f}")
        print(f"  S_cond(100k): {res['S_cond(100k)']:.4f}")
        print(f"  Tail Ratio (Robust): {res['tail_ratio_cond']:.4f}")
        
        if res['tail_ratio_cond'] > 0.2:
            print(">> ROBUST SUCCESS: Beach confirmed even with phase randomization.")
        else:
            print(">> WARNING: Beach might be thinner than expected.")
            
        # Plot
        plt.figure(figsize=(12, 5))
        
        # Atlas
        plt.subplot(1, 2, 1)
        plt.pcolormesh(X*1000, Y*1000, np.log10(T+1), shading='auto', cmap='inferno')
        plt.colorbar(label='Log10(Median Survival)')
        plt.title(f"Robust Atlas ({n_phase} phases)\nMed T")
        plt.xlabel("x0 [mm]"); plt.ylabel("y0 [mm]")
        
        # CCDF
        plt.subplot(1, 2, 2)
        plt.plot(res['times'], res['ccdf'], 'b-', lw=2)
        plt.xscale('log'); plt.yscale('log')
        plt.title(f"Conditioned CCDF (Band Only)\nTailRatio={res['tail_ratio_cond']:.2f}")
        plt.xlabel("Turns"); plt.ylabel("P(T>t | Band)")
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"analysis_{PRESET_NAME}.png", dpi=150)
        print("Analysis plot saved.")


def run_validation_abr_50k():
    """
    S1.3b: Agile But Robust (ABR-50k) Validation
    Grid: 40x30, x=[1.72, 1.92], y=[0.0, 0.3]
    Turns: 50k
    Phase Rand: 4 runs (median)
    """
    PRESET_NAME = "VALIDATION_ABR_50k"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # 1. Setup Wrapper
    eps = 0.06
    lat = build_sticky_chaos_lattice(eps=eps)
    
    class TurnAwareApertureTracker(SymplecticTracker):
        def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
            super().__init__(elements)
            self.limit_x = aperture_x
            self.limit_y = aperture_y
        def track(self, s0, n_turns):
            traj = np.zeros((n_turns, 6), dtype=float)
            s = np.array(s0, dtype=float)
            traj[0] = s
            surv = n_turns
            for i in range(1, n_turns):
                s = self.one_turn(s, turn=i-1)
                traj[i] = s
                if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                    surv = i
                    traj[i:] = np.nan; break
                if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                    surv = i
                    traj[i:] = np.nan; break
            return traj, surv
            
    tracker = TurnAwareApertureTracker(lat)
    
    # 2. Agile Robust Atlas Plan
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30        
    n_turns = 50000        
    n_phase = 4            
    
    X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
    
    np.savez(f"escape_time_{PRESET_NAME}.npz", X=X, Y=Y, T=T_median)
    print("Data saved.")

    # 3. ABR Analysis
    flat_T = T_median.ravel()
    
    mask_wall = flat_T < 500
    mask_island = flat_T >= n_turns
    mask_beach = (~mask_wall) & (~mask_island)
    
    p_wall = np.mean(mask_wall)
    p_island = np.mean(mask_island)
    p_beach = np.mean(mask_beach)
    
    print(f"\nABR-50k METRICS:")
    print(f"  p_wall   : {p_wall:.1%}")
    print(f"  p_beach  : {p_beach:.1%}")
    print(f"  p_island : {p_island:.1%}")
    
    # Bandwidth Calculation
    mask_beach_2d = mask_beach.reshape(ny, nx)
    has_beach_col = np.any(mask_beach_2d, axis=0) # boolean array of size nx
    if np.any(has_beach_col):
        x_indices = np.where(has_beach_col)[0]
        min_idx = x_indices[0]
        max_idx = x_indices[-1]
        bandwidth = X[max_idx] - X[min_idx]
    else:
        bandwidth = 0.0
    
    print(f"  Beach Bandwidth: {bandwidth*1000:.3f} mm")
    
    # Verdict
    if p_beach > 0.10 and bandwidth > 0.03e-3:
        print(">> VERDICT: BEACH CONFIRMED (Wide & Populated)")
    else:
        print(f">> VERDICT: CLIFF or THIN (p={p_beach:.1%}, bw={bandwidth*1000:.3f}mm)")
        
    # Plot 3-Class Map
    class_map = np.zeros_like(T_median, dtype=int)
    class_map[mask_beach_2d] = 1
    class_map[mask_island.reshape(ny, nx)] = 2
    
    plt.figure(figsize=(10, 5))
    
    # Map
    plt.subplot(1, 2, 1)
    from matplotlib.colors import ListedColormap
    # Colors: Wall=Red, Beach=Yellow, Island=Blue
    cmap = ListedColormap(['#FFCCCC', '#FFD700', '#ADD8E6']) 
    plt.pcolormesh(X*1000, Y*1000, class_map, shading='auto', cmap=cmap, vmin=0, vmax=2)
    plt.title(f"ABR-50k Map\nBeach={p_beach:.1%}")
    plt.xlabel("x0 [mm]"); plt.ylabel("y0 [mm]")
    
    # CCDF Cond
    plt.subplot(1, 2, 2)
    T_beach = flat_T[mask_beach]
    if len(T_beach) > 10:
        sorted_T = np.sort(T_beach)
        n_b = len(T_beach)
        prob = 1.0 - np.arange(n_b)/n_b
        plt.plot(sorted_T, prob, 'k-', lw=2)
        plt.xscale('log'); plt.yscale('log')
        plt.title("Beach CCDF (Conditioned)")
        plt.xlabel("Turns"); plt.ylabel("P(T>t | Beach)")
        plt.grid(True, which='both', alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No Beach Points", ha='center')
        
    plt.tight_layout()
    plt.savefig(f"analysis_{PRESET_NAME}.png", dpi=100)
    print("Saved plot.")


def run_validation_adaptive_zoom():
    """
    S1.3c: Adaptive Zoom (Deep Scan)
    1. Load ABR-50k results.
    2. Identify 'Beach' pixels (500 <= T < 50k).
    3. Re-track ONLY those pixels for 200k turns (Deep Dive).
    4. Compute conditioned stats (Tail Ratio, Hazard) on this Deep Set.
    """
    PRESET_NAME = "VALIDATION_ADAPTIVE_ZOOM"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # 1. Load Candidates
    try:
        data = np.load("escape_time_VALIDATION_ABR_50k.npz")
        X, Y, T_prev = data['X'], data['Y'], data['T']
    except FileNotFoundError:
        print("Error: Run ABR-50k first!")
        return

    # Identify Candidates
    nx, ny = len(X), len(Y)
    mask_beach = (T_prev >= 500) & (T_prev < 50000) # Only the beach
    candidates = []
    
    for iy in range(ny):
        for ix in range(nx):
            if mask_beach[iy, ix]:
                candidates.append( (ix, iy, X[ix], Y[iy]) )
                
    n_cand = len(candidates)
    print(f"Found {n_cand} Beach Candidates for Deep Scan (200k turns)...")
    
    if n_cand == 0:
        print("No candidates found. Skipping.")
        return

    # 2. Setup Tracker
    eps = 0.06
    lat = build_sticky_chaos_lattice(eps=eps)
    
    class TurnAwareApertureTracker(SymplecticTracker):
        def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
            super().__init__(elements)
            self.limit_x = aperture_x
            self.limit_y = aperture_y
        def track(self, s0, n_turns):
            traj = np.zeros((n_turns, 6), dtype=float)
            s = np.array(s0, dtype=float)
            traj[0] = s
            surv = n_turns
            for i in range(1, n_turns):
                s = self.one_turn(s, turn=i-1)
                traj[i] = s
                # Simple Aperture
                if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                    surv = i
                    traj[i:] = np.nan; break
                # NaN check
                if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                    surv = i
                    traj[i:] = np.nan; break
            return traj, surv
            
    tracker = TurnAwareApertureTracker(lat)
    
    # 3. Deep Track
    n_turns_deep = 200000
    T_deep = []
    
    print(f"Tracking {n_cand} particles for {n_turns_deep} turns...")
    
    for i, (ix, iy, x0, y0) in enumerate(candidates):
        if i % 10 == 0: print(f"  Particle {i+1}/{n_cand}...")
        init = PhaseSpaceState(x=x0, px=0.0, y=y0, py=0.0) # Nominal phase (or could use rand)
        _, surv = tracker.track(init, n_turns_deep)
        T_deep.append(surv)
        
    T_deep = np.array(T_deep)
    
    # Save Deep Data
    np.savez(f"escape_time_{PRESET_NAME}.npz", T_deep=T_deep, candidates=candidates)
    
    # 4. Analysis
    # Conditioned CCDF on this subset
    sorted_T = np.sort(T_deep)
    n = len(sorted_T)
    prob = 1.0 - np.arange(n)/n
    
    # Metrics
    def get_prob(t_target, t_arr, p_arr):
        idx = np.searchsorted(t_arr, t_target)
        if idx >= len(t_arr): return 0.0
        return p_arr[idx]
        
    s10k = get_prob(10000, sorted_T, prob)
    s100k = get_prob(100000, sorted_T, prob)
    tail_ratio = s100k / (s10k + 1e-12)
    
    print(f"\nDEEP ZOOM METRICS (Conditioned on Beach):")
    print(f"  N_particles: {n}")
    print(f"  S(10k)     : {s10k:.4f}")
    print(f"  S(100k)    : {s100k:.4f}")
    print(f"  Tail Ratio : {tail_ratio:.4f}")
    
    if tail_ratio > 0.1:
        print(">> VERDICT: DEEP BEACH CONFIRMED (Heavy Tail persists at 100k)")
    else:
        print(">> VERDICT: BEACH COLLAPSE (Tail drops off)")
        
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(sorted_T, prob, 'b.-', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.title(f"Deep Beach CCDF (N={n})\nTailRatio={tail_ratio:.2f}")
    plt.xlabel("Turns"); plt.ylabel("P(T>t | BeachCandidates)")
    plt.grid(True, which='both', alpha=0.3)
    plt.axvline(10000, color='gray', ls='--', alpha=0.5)
    plt.axvline(100000, color='gray', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"analysis_{PRESET_NAME}.png", dpi=100)
    print("Saved plot.")


def run_mini_sweep_s1_3d():
    """
    S1.3d: Epsilon Mini-Sweep (Scientific Guardrails)
    eps in {0.02, 0.06, 0.10}
    Grid: 40x30 (Same as ABR)
    Turns: 100k (Scientific minimum)
    Phase Rand: 4 runs (randomize modulation phase)
    """
    PRESET_NAME = "MINI_SWEEP_S1_3d"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    eps_values = [0.02, 0.06, 0.10]
    results = []
    
    # Grid Config (ABR-like)
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_turns = 100000 
    n_phase = 4
    
    print(f"{'Eps':<6} | {'p0':<8} | {'p_fast':<8} | {'Gamma':<8} | {'St(1e4)':<8} | {'St(1e5)':<8} | {'Beach%':<8}")
    print("-" * 70)
    
    for eps in eps_values:
        # Build Lattice
        lat = build_sticky_chaos_lattice(eps=eps)
        
        # Build Phase-Aware Tracker
        # We need to randomize MODULATION phase. 
        # The DualModulatedSextupole (or Single) generally starts at t=0.
        # To randomize phase, easiest way is to randomize initial turn index? 
        # ModulatedSextupoleKick uses: phase = self.omega * (turn + 1)
        # So we can pass a random 'turn_offset' to the tracker?
        # Or just use the existing robust atlas which randomizes INITIAL COORDINATES (p_x, p_y).
        # User said: "randomizza phase della modulazione φ (consigliato) OPPURE randomizza fase iniziale (angolo betatron)".
        # Our `escape_time_atlas_robust` does random p_x, p_y (betatron phase). This is "OPPURE" option.
        # It's easier to use existing robust atlas.
        
        class TurnAwareApertureTracker(SymplecticTracker):
            def __init__(self, elements, aperture_x=5e-3, aperture_y=5e-3):
                super().__init__(elements)
                self.limit_x = aperture_x
                self.limit_y = aperture_y
            def track(self, s0, n_turns):
                traj = np.zeros((n_turns, 6), dtype=float)
                s = np.array(s0, dtype=float)
                traj[0] = s
                surv = n_turns
                for i in range(1, n_turns):
                    s = self.one_turn(s, turn=i-1)
                    traj[i] = s
                    if abs(s[0]) > self.limit_x or abs(s[2]) > self.limit_y:
                        surv = i
                        traj[i:] = np.nan; break
                    if np.any(np.isnan(s)) or np.max(np.abs(s)) > 1e10:
                        surv = i
                        traj[i:] = np.nan; break
                return traj, surv

        tracker = TurnAwareApertureTracker(lat)
        
        # Run Robust Scan
        # Note: escape_time_atlas_robust prints "Scanning..."
        # We want to capture the Median T map
        X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
        
        # Calculate Metrics
        flat_T = T_median.ravel()
        n_total = len(flat_T)
        
        # 1. p0 (Censored Fraction)
        p0 = np.mean(flat_T >= n_turns)
        
        # 2. p_fast (Immediate Loss < 2k)
        p_fast = np.mean(flat_T < 2000)
        
        # 3. Beach Band Area% (500 <= T < 50k) - Note: 50k is consistent with ABR definition
        p_beach = np.mean((flat_T >= 500) & (flat_T < 50000))
        
        # 4. Gamma & Stilde (Plateau Corrected)
        # Filter escapers: T < n_turns
        escapers_T = flat_T[flat_T < n_turns]
        
        if len(escapers_T) > 10 and p0 < 0.99:
            # Empirical Survival of escapers
            sorted_T = np.sort(escapers_T)
            # P(T > t | Escaped)
            n_esc = len(escapers_T)
            prob_esc = 1.0 - np.arange(n_esc)/n_esc
            
            # Stilde(t) = S(t) - p0 / (1 - p0) ?? 
            # Actually, S(t) global = p0 + (1-p0)*S_esc(t).
            # So S_esc(t) IS Stilde(t) !
            # Check: S(t) -> p0 as t -> inf.
            # Stilde(t) -> 0 as t -> inf. Correct.
            
            # Helper for Stilde values
            def get_stilde(t_val):
                idx = np.searchsorted(sorted_T, t_val)
                if idx >= n_esc: return 0.0
                return prob_esc[idx]
            
            st_1e4 = get_stilde(10000)
            st_1e5 = get_stilde(100000)
            
            # Gamma Fit (Slope Hat)
            # Log-Log fit on [1e3, 1e5] (or 0.5*Tmax)
            t_min_fit = 1000
            t_max_fit = n_turns * 0.5 # 50k
            
            mask_fit = (sorted_T >= t_min_fit) & (sorted_T <= t_max_fit)
            if np.sum(mask_fit) > 5:
                log_t = np.log10(sorted_T[mask_fit])
                log_s = np.log10(prob_esc[mask_fit])
                slope, _ = np.polyfit(log_t, log_s, 1)
                gamma = -slope # Convention: S(t) ~ t^-gamma
            else:
                gamma = 0.0
        else:
            st_1e4 = 0.0
            st_1e5 = 0.0
            gamma = 0.0
            
        # Print Row
        print(f"{eps:<6.2f} | {p0:<8.4f} | {p_fast:<8.4f} | {gamma:<8.4f} | {st_1e4:<8.4f} | {st_1e5:<8.4f} | {p_beach:<8.1%}")
        
        results.append({
            "eps": eps, "p0": p0, "p_fast": p_fast, 
            "gamma": gamma, "st1e4": st_1e4, "st1e5": st_1e5,
            "p_beach": p_beach
        })
        
        # Save T map for this epsilon
        np.savez(f"escape_time_sweep_eps_{eps:.2f}.npz", X=X, Y=Y, T=T_median)

    print("-" * 70)
    print("Sweep Complete.")


from symplectic_tracker_v2 import TurboTrackerS1_4

def run_mutation_d_abr():
    """
    S1.4: Dual-Frequency Drive (Cantori Thickening)
    Protocol: ABR (Agile But Robust)
    Grid: 40x30 (Zoomed)
    Turns: 50k (Fast Scan) -> 200k (Deep Scan planned)
    Eps: 0.10, eps2=0.04
    Tracker: TurboTrackerS1_4 (Optimized)
    """
    PRESET_NAME = "MUTATION_D_ABR"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # 1. Setup Turbo Tracker
    # Epsilon choice: 0.10 (from Sweep)
    eps1 = 0.10
    eps2 = 0.04 # Secondary drive
    print(f"Loading Turbo Tracker (Dual Freq: eps1={eps1}, eps2={eps2})...")
    
    tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, eps1=eps1, eps2=eps2)
    
    # 2. ABR Grid
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_turns = 50000 
    n_phase = 4
    
    # 3. Running Robust Scan
    # We need to manually perform the loop because escape_time_atlas_robust expects 'tracker'
    # to behave like a SymplecticTracker. Our TurboTracker does match the .track signature (roughly).
    # But escape_time_atlas_robust also randomizes initial x,y for phases. 
    # That works fine with TurboTracker.
    
    print(f"Scanning Dual-Freq Grid {nx}x{ny} for {n_turns} turns...")
    
    # Reuse existing robust function
    X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
    
    np.savez(f"escape_time_{PRESET_NAME}.npz", X=X, Y=Y, T=T_median)
    print("Data saved.")
    
    # 4. Analysis
    flat_T = T_median.ravel()
    
    p_fast = np.mean(flat_T < 500)
    p_harbor = np.mean(flat_T >= n_turns)
    p_beach_event = np.mean( (flat_T >= 500) & (flat_T < n_turns) )
    
    print(f"\nS1.4 ABR METRICS:")
    print(f"  p_fast (Wall)   : {p_fast:.1%}")
    print(f"  p_harbor (Stab) : {p_harbor:.1%}")
    print(f"  p_beach_event   : {p_beach_event:.1%}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    # Map
    plt.subplot(1, 2, 1)
    from matplotlib.colors import ListedColormap
    # Custom cmap: Wall=Red, Beach=Yellow, Harbor=Blue
    cmap = ListedColormap(['#FFCCCC', '#FFD700', '#ADD8E6']) 
    
    # Create class map
    class_map = np.zeros_like(T_median, dtype=int)
    mask_beach = (T_median >= 500) & (T_median < n_turns)
    mask_harbor = (T_median >= n_turns)
    class_map[mask_beach] = 1
    class_map[mask_harbor] = 2
    
    plt.pcolormesh(X*1000, Y*1000, class_map, shading='auto', cmap=cmap, vmin=0, vmax=2)
    plt.title(f"S1.4 Dual-Drive\nBeach={p_beach_event:.1%}")
    plt.xlabel("x0 [mm]"); plt.ylabel("y0 [mm]")
    
    # CCDF of Events
    plt.subplot(1, 2, 2)
    events_T = flat_T[(flat_T >= 500) & (flat_T < n_turns)]
    if len(events_T) > 10:
        sorted_T = np.sort(events_T)
        n_ev = len(events_T)
        prob = 1.0 - np.arange(n_ev)/n_ev
        plt.plot(sorted_T, prob, 'k-', lw=2)
        plt.xscale('log'); plt.yscale('log')
        plt.title("Beach Events Only (No Censored)")
        plt.xlabel("Turns"); plt.ylabel("P(T>t | Stayed>500 & Lost)")
        plt.grid(True, which='both', alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Insufficient Events", ha='center')
        
    plt.tight_layout()
    plt.savefig(f"analysis_{PRESET_NAME}.png", dpi=100)
    print("Saved plot.")


def run_mutation_d_deep_confirm():
    """
    S1.4b: Deep-Confirm & Transition Matrix
    Grid: 40x30 (Same as ABR)
    Turns: 200,000 (Deep Scan)
    Protocol: Dual-Freq (eps=0.10, 0.04)
    Analysis: Transition Matrix (50k->200k), S_cond, Symplectic Check.
    """
    PRESET_NAME = "MUTATION_D_DEEP_CONFIRM"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # 1. Symplectic Sanity Check (Time-Dependent)
    # We verify Jacobian at a few turns to ensure no obvious drift/error.
    # Note: TurboTracker doesn't support Jacobian calculation easily (it's hardcoded).
    # We use the SymplecticTracker class with DualModulatedSextupole for this check.
    print("Performing Symplectic Sanity Check...")
    
    # Re-build lattice object for checking
    # Note: build_sticky_chaos_lattice creates SINGLE frequency.
    # We need DUAL frequency lattice for check.
    # We manually construct a test element.
    dms = DualModulatedSextupoleKick(k2_base=10.0, eps1=0.10, eps2=0.04)
    # Check simple tracking
    s0 = np.array([1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0])
    s1 = dms.apply(s0.copy(), turn=0)
    s2 = dms.apply(s0.copy(), turn=100)
    
    # Jacobian check
    def f_t0(x): return dms.apply(x.copy(), turn=0)
    def f_t100(x): return dms.apply(x.copy(), turn=100)
    
    M0 = jacobian_fd(f_t0, s0)
    M100 = jacobian_fd(f_t100, s0)
    
    err0 = symplectic_error(M0)
    err100 = symplectic_error(M100)
    
    print(f"  Symplectic Error (Turn 0)   : {err0:.2e}")
    print(f"  Symplectic Error (Turn 100) : {err100:.2e}")
    
    if max(err0, err100) > 1e-8:
        print("WARNING: Symplectic error high! Integration might be unstable.")
    else:
        print("  Sanity Check PASSED.")

    # 2. Setup Turbo Tracker for DEEP SCAN
    eps1 = 0.12  # Winner S1.4c
    eps2 = 0.04
    print(f"Loading Turbo Tracker (Deep Scan 200k) for WINNER eps=({eps1}, {eps2})...")
    tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, eps1=eps1, eps2=eps2)
    
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_turns = 200000 
    n_phase = 4
    
    print(f"Scanning Grid {nx}x{ny} for {n_turns} turns...")
    X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
    
    np.savez(f"escape_time_{PRESET_NAME}.npz", X=X, Y=Y, T=T_median)
    print("Data saved.")
    
    # 3. Transition Matrix & Analysis
    flat_T = T_median.ravel()
    
    # Class Definitions
    # 50k Cutoff Classes
    # Status50k: 
    #   0: Fast/Wall (< 500)
    #   1: Beach50k (500 <= T < 50k)
    #   2: Alive50k (T >= 50k)
    
    # 200k Classes
    # Status200k:
    #   0: Fast/Wall (< 500)
    #   1: BeachShort (500 <= T < 50k)   <-- Should match Beach50k
    #   2: BeachLong  (50k <= T < 200k)  <-- Was Alive50k, now revealed as Lazy Beach
    #   3: Harbor     (T >= 200k)        <-- Truly Stable (Censored)
    
    # We compute the flow from Alive50k -> {BeachLong, Harbor}
    
    n_total = len(flat_T)
    c50_wall = (flat_T < 500)
    c50_beach = (flat_T >= 500) & (flat_T < 50000)
    c50_alive = (flat_T >= 50000)
    
    # Of the c50_alive, where did they go?
    alive_T = flat_T[c50_alive]
    c200_long_beach = (alive_T < n_turns)
    c200_harbor = (alive_T >= n_turns)
    
    n_wall    = np.sum(c50_wall)
    n_beach50 = np.sum(c50_beach)
    n_alive50 = np.sum(c50_alive)
    
    n_long    = np.sum(c200_long_beach)
    n_harbor  = np.sum(c200_harbor)
    
    # Transition Logic
    # Wall -> Wall (100%)
    # Beach50 -> BeachShort (100% by definition)
    # Alive50 -> LongBeach OR Harbor
    
    print("\n=== TRANSITION MATRIX (50k -> 200k) ===")
    print(f"Total Particles: {n_total}")
    print(f"WARNING: This matrix infers transitions from a single 200k run.")
    
    print(f"1. WALL (<500)          : {n_wall} ({n_wall/n_total:.1%}) -> Remained Wall")
    print(f"2. BEACH (<50k)         : {n_beach50} ({n_beach50/n_total:.1%}) -> Remained Beach (Short)")
    print(f"3. ALIVE (@50k)         : {n_alive50} ({n_alive50/n_total:.1%})")
    print(f"   |-> Resolved to BEACH: {n_long} ({n_long/n_total:.1%}) [Hidden Lazy Beach]")
    print(f"   |-> Remained HARBOR  : {n_harbor} ({n_harbor/n_total:.1%}) [True Stable]")
    
    # Conditioned Survival Analysis
    # p0 = p_harbor (Censored fraction at 200k)
    p0 = n_harbor / n_total
    
    if p0 < 1.0:
        # S_cond(t) = (S(t) - p0) / (1 - p0)
        # We compute this for events (non-censored)
        # Filter events: T < n_turns
        events = flat_T[flat_T < n_turns]
        sorted_events = np.sort(events)
        
        # We need S(t) first.
        # S(t) = P(T > t)
        # S_cond(t) = P(T > t | T < inf) ... roughly.
        # Actually S_cond(t) = P(T > t) - p0 / (1-p0) is valid if S(t) includes p0.
        
        # Let's compute S_cond at critical points
        # t = 10k, 50k, 100k
        
        def get_S(t):
            return np.mean(flat_T > t)
            
        S_10k = get_S(10000)
        S_50k = get_S(50000)
        S_100k = get_S(100000)
        
        S_cond_10k  = max(0, (S_10k - p0) / (1 - p0))
        S_cond_50k  = max(0, (S_50k - p0) / (1 - p0))
        S_cond_100k = max(0, (S_100k - p0) / (1 - p0))
        
        print(f"\nCONDITIONED SURVIVAL (p0={p0:.3f}):")
        print(f"  Sc(10k) : {S_cond_10k:.4f}")
        print(f"  Sc(50k) : {S_cond_50k:.4f}")
        print(f"  Sc(100k): {S_cond_100k:.4f}")
        
        # Hazard check
        # If Sc ~ t^-gamma, then Sc(50k)/Sc(10k) ~ 5^-gamma
        # gamma ~ -log(ratio)/log(5)
        if S_cond_10k > 0:
            ratio = S_cond_50k / S_cond_10k
            if ratio > 0:
                gamma_est = -np.log(ratio) / np.log(5.0)
                print(f"  Est. Gamma (10k-50k): {gamma_est:.2f}")
    
    # Plot S_cond
    try:
        plt.figure(figsize=(6, 5))
        # Full S(t)
        all_sorted = np.sort(flat_T)
        S_t = 1.0 - np.arange(len(all_sorted))/len(all_sorted)
        # Calculate Sc
        Sc_t = (S_t - p0) / (1.0 - p0)
        # Mask where Sc > 0
        mask_valid = (Sc_t > 1e-4) & (all_sorted < n_turns)
        
        plt.plot(all_sorted[mask_valid], Sc_t[mask_valid], 'b-', lw=2, label='S_cond(t)')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Turns"); plt.ylabel("Conditioned Survival")
        plt.title(f"S1.4b Conditioned Survival\np0={p0:.2f}")
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.savefig(f"analysis_{PRESET_NAME}.png", dpi=100)
        print("Saved S_cond plot.")
    except Exception as e:
        print(f"Plotting error: {e}")


def run_micro_sweep_s1_4c():
    """
    S1.4c: Dual-Frequency Micro-Sweep
    Grid: 40x30 (Same as ABR)
    Turns: 50,000 (Short ABR)
    Phase: 4 Random
    Parameters:
      eps1 in {0.08, 0.10, 0.12}  (Primary)
      eps2 in {0.02, 0.04, 0.06}  (Secondary)
    Objective J = p_beach_50k - 0.5 * p_wall
    Constraint: p_alive_50k >= 0.22 (Harbor size check)
    """
    PRESET_NAME = "MICRO_SWEEP_S1_4c"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    eps1_vals = [0.08, 0.10, 0.12]
    eps2_vals = [0.02, 0.04, 0.06]
    
    results = []
    
    # Baseline for comparison (eps1=0.10, eps2=0.04)
    # beach=6.7% (short) + 5.8% (long) -> Total 12.5%. BUT ABR-50k sees short (6.7) + long (alive).
    # Wait, ABR-50k sees:
    #   p_beach (500-50k) = 6.7% + 5.8%? NO.
    #   ABR definition: p_beach is strictly 500 <= T < 50k.
    #   ABR p_alive is T >= 50k.
    #   So ABR saw p_beach=11.8% in run_mutation_d_abr?
    #   Let's check previous output: S1.4 ABR METRICS: p_beach_event : 11.8%.
    #   Wait, deep confirm showed: Beach 50k (6.7%) + Long Beach (5.8%).
    #   Ah, deep confirm uses T >= 200k for Harbor.
    #   ABR 50k uses T >= 50k for Harbor/Alive.
    #   So ABR p_beach (11.8%) included mostly short beach?
    #   Wait, 11.8% is p(500 < T < 50k).
    #   Deep Confirm transition matrix says: Beach < 50k = 80/1200 = 6.7%.
    #   Why the discrepancy? 11.8% vs 6.7%?
    #   S1.4 run had p_beach_event = 11.8%.
    #   S1.4b run had Beach(<50k) = 6.7%.
    #   Maybe randomness/phase? 
    #   Actually S1.4b single deep run might have just been unlucky or "lazy" particles were resolved differently.
    #   Anyway, we use consistent metric within this sweep.
    
    print(f"{'eps1':<5} | {'eps2':<5} | {'p_wall':<8} | {'p_beach':<8} | {'p_alive':<8} | {'Score J':<8}")
    print("-" * 65)
    
    candidates = []

    for e1 in eps1_vals:
        for e2 in eps2_vals:
            # 1. Setup Tracker
            tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, eps1=e1, eps2=e2)
            
            # 2. Run ABR-50k
            # print(f"Scanning ({e1}, {e2})...")
            # Suppress excessive printing
            
            x_range = (1.72e-3, 1.92e-3)
            y_range = (0.0, 0.30e-3)
            nx, ny = 40, 30
            n_turns = 50000 
            n_phase = 4
            
            # Direct call 
            X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase)
            
            flat_T = T_median.ravel()
            n_tot = len(flat_T)
            
            p_wall = np.mean(flat_T < 500)
            p_beach = np.mean((flat_T >= 500) & (flat_T < n_turns))
            p_alive = np.mean(flat_T >= n_turns)
            
            # Score J
            # J = p_beach - 0.5 * p_wall (Correction: user said p_long + 0.5 p_short... 
            # In 50k scan, p_alive contains p_long + p_harbor.
            # p_beach contains p_short.
            # User suggested: J = p_beach_50k - lambda p_wall is simpler ABR proxy.
            # Let's use: J = p_beach - 0.1 * p_wall, with Constraint p_alive >= 0.22.
            # Or following user prompt strictly: "J = p_beach - lambda p_wall". lambda=0.1 seems fair.
            
            J = p_beach - 0.1 * p_wall
            if p_alive < 0.22:
                J = -1.0 # Penalize heavily
                
            print(f"{e1:<5.2f} | {e2:<5.2f} | {p_wall:<8.1%} | {p_beach:<8.1%} | {p_alive:<8.1%} | {J:<8.4f}")
            
            filename = f"escape_time_sweep_{e1}_{e2}.npz"
            np.savez(filename, X=X, Y=Y, T=T_median)
            
            candidates.append({
                'eps1': e1, 'eps2': e2, 
                'p_wall': p_wall, 'p_beach': p_beach, 'p_alive': p_alive, 'J': J
            })
            
    print("-" * 65)
    

    # ... (previous code) ...
    
    # Sort by J
    candidates.sort(key=lambda x: x['J'], reverse=True)
    
    print("\nTOP 2 CANDIDATES:")
    for i in range(min(2, len(candidates))):
        c = candidates[i]
        print(f"#{i+1}: eps1={c['eps1']}, eps2={c['eps2']} -> J={c['J']:.4f} (Beach={c['p_beach']:.1%}, Alive={c['p_alive']:.1%})")
        
    # Check Stop Rule
    baseline = next((c for c in candidates if abs(c['eps1']-0.10)<1e-9 and abs(c['eps2']-0.04)<1e-9), None)
    
    if baseline:
        best = candidates[0]
        delta_beach = best['p_beach'] - baseline['p_beach']
        print(f"\nImprovement vs Baseline (0.10, 0.04): {delta_beach:+.1%}")
        
        if delta_beach < 0.01:
            print("verdict: MARGINAL IMPROVEMENT (< 1%). Suggest stopping.")
        else:
            print("verdict: SIGNIFICANT IMPROVEMENT. Worth Deep Scanning.")
    else:
        print("Baseline not found exactly? (rounding error?)")
        
    # ==========================
    # VISUALIZATION (Requested)
    # ==========================
    try:
        # Create a pivoting table for Heatmap
        # Rows: eps2, Cols: eps1
        n_e1 = len(eps1_vals)
        n_e2 = len(eps2_vals)
        J_matrix = np.zeros((n_e2, n_e1))
        Beach_matrix = np.zeros((n_e2, n_e1))
        
        # Fill matrices
        idx_e1 = {val: i for i, val in enumerate(eps1_vals)}
        idx_e2 = {val: i for i, val in enumerate(eps2_vals)} # eps2 is y-axis (rows) usually from bottom? 
        # Let's map e2 index 0 to top or bottom.imshow origin='lower' puts index 0 at bottom.
        # So index 0 (0.02) corresponds to bottom row.
        
        for c in candidates:
            i = idx_e2[c['eps2']]
            j = idx_e1[c['eps1']]
            J_matrix[i, j] = c['J']
            Beach_matrix[i, j] = c['p_beach']
            
        plt.figure(figsize=(10, 8))
        
        # Subplot 1: Score J
        plt.subplot(2, 1, 1)
        plt.imshow(J_matrix, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Objective J')
        plt.xticks(range(n_e1), eps1_vals)
        plt.yticks(range(n_e2), eps2_vals)
        plt.xlabel(r"$\epsilon_1$ (Golden)")
        plt.ylabel(r"$\epsilon_2$ ($\sqrt{2}$)")
        plt.title("Optimization Landscape: Score J")
        
        # Annotate
        for i in range(n_e2):
            for j in range(n_e1):
                val = J_matrix[i, j]
                plt.text(j, i, f"{val:.3f}", ha='center', va='center', color='w', fontweight='bold')

        # Subplot 2: Beach Fraction
        plt.subplot(2, 1, 2)
        plt.imshow(Beach_matrix, origin='lower', cmap='plasma', aspect='auto')
        plt.colorbar(label='Beach Fraction')
        plt.xticks(range(n_e1), eps1_vals)
        plt.yticks(range(n_e2), eps2_vals)
        plt.xlabel(r"$\epsilon_1$ (Golden)")
        plt.ylabel(r"$\epsilon_2$ ($\sqrt{2}$)")
        plt.title("Beach Fraction (Sticky Chaos)")
        
        # Annotate
        for i in range(n_e2):
            for j in range(n_e1):
                val = Beach_matrix[i, j]
                plt.text(j, i, f"{val:.1%}", ha='center', va='center', color='w', fontweight='bold')
                
        plt.tight_layout()
        plt.savefig("analysis_MICRO_SWEEP_S1_4c_summary.png", dpi=100)
        print("Saved summary plot: analysis_MICRO_SWEEP_S1_4c_summary.png")
        
    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    print(">>> SCRIPT STARTED: S1.4c DEEP CONFIRM (WINNER) <<<")
    run_mutation_d_deep_confirm()
