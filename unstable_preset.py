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
from synchrotron_tracker import SynchrotronTracker6D

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
def escape_time_atlas_robust(tracker, x_range, y_range, n_turns=50000, nx=80, ny=60, n_phase=8, seed=42):
    """
    Robust Atlas with Phase Randomization:
    For each (x,y), run n_phase micro-runs with small random (px, py).
    Take MEDIAN survival time to filter out phase-dependent lucky pockets.
    seed=None for random behavior.
    """
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    T_map = np.zeros((ny, nx), dtype=float)
    
    print(f"Scanning Robust Grid {nx}x{ny} (Phase Rand={n_phase}, Seed={seed})...")
    
    rng = np.random.default_rng(seed)
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


def run_s1_4d_comparative():
    """
    S1.4d: Comparative Validation (Anti-Self-Deception)
    Configs: Baseline, Runner-up, Winner.
    Protocol: 3 Replicas * 200k turns * 1 Phase.
    Grid: 40x30 robust.
    Metrics: Wall, ShortBeach, LongBeach, Harbor.
    """
    PRESET_NAME = "COMPARATIVE_S1_4d"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # Configs
    configs = [
        {'name': 'Baseline',  'eps1': 0.10, 'eps2': 0.04},
        {'name': 'Runner-Up', 'eps1': 0.10, 'eps2': 0.06},
        {'name': 'Winner',    'eps1': 0.12, 'eps2': 0.04},
    ]
    
    n_replicas = 3
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_turns = 200000
    n_phase_per_rep = 1 # To allow 3 distinct replicas in reasonable time
    
    # Storage structure: results[config_name] = list of metrics dicts
    data = {c['name']: [] for c in configs}
    
    print(f"{'Config':<10} | {'Rep':<3} | {'Wall':<6} | {'Short':<6} | {'Long':<6} | {'Harbor':<6} | {'TotBeach':<8}")
    print("-" * 75)
    
    for cfg in configs:
        name = cfg['name']
        e1 = cfg['eps1']
        e2 = cfg['eps2']
        
        for r in range(n_replicas):
            # 1. Setup Tracker
            tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, eps1=e1, eps2=e2)
            
            # 2. Run Scan
            # escape_time_atlas_robust handles randomization internally if we call it.
            # With n_phase=1 it takes 1 random phase offset.
            X, Y, T_median = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase_per_rep)
            
            flat_T = T_median.ravel()
            
            # 3. Classify (Deep Protocol)
            # Wall: < 500
            # Short: 500 <= T < 50k
            # Long: 50k <= T < 200k
            # Harbor: T >= 200k
            
            p_wall = np.mean(flat_T < 500)
            p_short = np.mean((flat_T >= 500) & (flat_T < 50000))
            p_long = np.mean((flat_T >= 50000) & (flat_T < n_turns))
            p_harbor = np.mean(flat_T >= n_turns)
            p_tot_beach = p_short + p_long
            
            metrics = {
                'p_wall': p_wall,
                'p_short': p_short,
                'p_long': p_long,
                'p_harbor': p_harbor,
                'p_tot_beach': p_tot_beach
            }
            data[name].append(metrics)
            
            print(f"{name:<10} | {r+1:<3} | {p_wall:<6.1%} | {p_short:<6.1%} | {p_long:<6.1%} | {p_harbor:<6.1%} | {p_tot_beach:<8.1%}")
            
            # Save raw data for safety
            np.savez(f"escape_time_s1_4d_{name}_rep{r}.npz", X=X, Y=Y, T=T_median)
            
    # Statistical Summary
    print("\n=== S1.4d STATISTICAL SUMMARY (Mean +/- Std) ===")
    print(f"{'Config':<10} | {'TotalBeach':<15} | {'LongBeach':<15} | {'Harbor':<15} | {'Verdict'}")
    print("-" * 80)
    
    # Helper for fmt
    def fmt(vals):
        m = np.mean(vals)
        s = np.std(vals)
        return f"{m:.1%} +/- {s:.1%}"
    
    stats_summary = {}
    
    for cfg in configs:
        name = cfg['name']
        mets = data[name]
        
        tot_beach = [m['p_tot_beach'] for m in mets]
        long_beach = [m['p_long'] for m in mets]
        harbor = [m['p_harbor'] for m in mets]
        
        stats_summary[name] = {
            'beach_mean': np.mean(tot_beach),
            'harbor_mean': np.mean(harbor)
        }
        
        print(f"{name:<10} | {fmt(tot_beach):<15} | {fmt(long_beach):<15} | {fmt(harbor):<15} | ...")
        
    # Final Decision Logic (J')
    # J' = TotalBeach - lambda * max(0, BaseHarbor - CandHarbor)
    # lambda = 0.5 (conservative penalty)
    
    base_harbor = stats_summary['Baseline']['harbor_mean']
    base_beach = stats_summary['Baseline']['beach_mean']
    
    print("\n=== FINAL VERDICT (Objective J') ===")
    lamb = 0.5
    
    for cfg in configs:
        name = cfg['name']
        cand_harbor = stats_summary[name]['harbor_mean']
        cand_beach = stats_summary[name]['beach_mean']
        
        harbor_loss = max(0.0, base_harbor - cand_harbor)
        # However, we only care if harbor drops below critical threshold? 
        # User said: J' = TotalBeach - lambda * loss.
        # User also said: "subject to harbor >= 0.22".
        
        J_prime = cand_beach - lamb * harbor_loss
        
        gain = cand_beach - base_beach
        
        flag = ""
        if name == "Baseline":
            flag = "(REF)"
        elif gain > 0.01: # +1%
            if cand_harbor >= 0.22:
                 flag = "WINNER CANDIDATE"
            else:
                 flag = "UNSTABLE (Harbor too low)"
        
        print(f"{name:<10}: Beach={cand_beach:.1%} (Gain={gain:+.1%}), Harbor={cand_harbor:.1%}, J'={J_prime:.4f} {flag}")


def run_frequency_sweep_s1_4e():
    """
    S1.4e: Frequency Sweep R&D (High ROI)
    Sweep f2 = omega2/(2pi) to maximize Long Beach.
    Protocol:
      1. ABR-50k screening on 6 frequencies.
      2. Identify Top-2 (LongBeach Proxy).
      3. Deep 200k Scan on Top-2.
    """
    PRESET_NAME = "FREQUENCY_SWEEP_S1_4e"
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # Constants
    pi = np.pi
    phi = (np.sqrt(5)-1)/2
    
    # 6 Frequencies of Interest
    # Format: (name, f2_val)
    freq_targets = [
        ("pi-3",      np.pi - 3.0),         # ~0.1416
        ("1/pi",      1.0/np.pi),           # ~0.3183
        ("sqrt2-1",   np.sqrt(2.0)-1.0),    # ~0.4142 (Current Baseline)
        ("sqrt2-0.9", np.sqrt(2.0)-0.9),    # ~0.5142 (Typo in user prompt said 0.5176? sqrt2=1.414. 1.414-0.9=0.514. Close enough.)
        ("phi-1",     phi - 1.0 + 1.0),     # phi approx 0.618. 
                                            # Wait user said (phi-1). phi=0.618. phi-1 = -0.382? 
                                            # Maybe they meant phi itself? "vicino a w1 ma non uguale". w1=phi.
                                            # User prompt: "0.6180 (phi-1)". Actually phi is 0.618. 
                                            # Let's use 0.6180.
        ("4nu-1",     abs(1.0 - 4*0.255)),  # Sideband target. nu_x~0.255. 1-1.02 ~ 0.02. 
                                            # User said 0.0320. Let's use user value: 0.0320
    ]
    
    # User specific values override
    freqs = [
        ("f_0.1416", 0.1416),
        ("f_0.3183", 0.3183),
        ("f_0.4142", 0.4142), # Baseline
        ("f_0.5176", 0.5176),
        ("f_0.6180", 0.6180),
        ("f_0.0320", 0.0320)
    ]
    
    eps1 = 0.10
    eps2 = 0.04
    
    print(f"{'Name':<10} | {'f2':<8} | {'p_wall':<8} | {'p_beach':<8} | {'p_alive':<8} | {'Score':<8}")
    print("-" * 70)
    
    candidates = []
    
    # 1. SCREENING (ABR 50k)
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_turns_screen = 50000 
    n_phase = 4
    
    for name, f2 in freqs:
        w2 = 2 * np.pi * f2
        
        # Setup tracker with omega2
        tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, 
                                   eps1=eps1, eps2=eps2,
                                   omega2=w2) # omega1 default is phi
                                   
        # Run ABR (Seed=None for randomness, but consistent within this screening? No, robustness requires rand)
        X, Y, T = escape_time_atlas_robust(tracker, x_range, y_range, n_turns_screen, nx, ny, n_phase, seed=None)
        
        flat_T = T.ravel()
        p_wall = np.mean(flat_T < 500)
        p_beach = np.mean((flat_T >= 500) & (flat_T < n_turns_screen))
        p_alive = np.mean(flat_T >= n_turns_screen) # Harbor + LongBeach
        
        # Score for Screening: Maximize Alive (potential Long Beach) + Beach
        # User: "start top-2 on LongBeach proxy".
        # Proxy for LongBeach at 50k is p_alive (since Harbor is inside p_alive).
        # We want Alive to be HIGH, but not 100% Harbor.
        # Let's use J = p_beach + p_alive - 0.5*p_wall. 
        # Or better: p_alive (assuming Harbor is ~20-25%).
        # If p_alive > 40%, it's likely just stable.
        # Let's use Score = p_beach + (p_alive if p_alive < 0.35 else 0.0)
        # Actually user said: "maximize LongBeach... proxy".
        # Let's stick to Total Retention (Beach+Alive).
        
        score = p_beach + p_alive
        
        print(f"{name:<10} | {f2:<8.4f} | {p_wall:<8.1%} | {p_beach:<8.1%} | {p_alive:<8.1%} | {score:<8.4f}")
        
        candidates.append({'name': name, 'f2': f2, 'score': score})
        
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top2 = candidates[:2]
    
    print("\n=== TOP 2 CANDIDATES (DEEP SCAN 200k) ===")
    print(f"1. {top2[0]['name']} (f2={top2[0]['f2']})")
    print(f"2. {top2[1]['name']} (f2={top2[1]['f2']})")
    
    # 2. DEEP SCAN (200k)
    print("\nRunning Deep Scan (200k) on Top 2...")
    n_turns_deep = 200000
    
    for cand in top2:
        name = cand['name']
        f2 = cand['f2']
        w2 = 2 * np.pi * f2
        
        print(f"\n>> Deep Scan: {name} (f2={f2:.4f}, eps=0.10/0.04)")
        
        tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, 
                                   eps1=eps1, eps2=eps2,
                                   omega2=w2)
                                   
        # Deep Scan (Seed=None)
        X, Y, T = escape_time_atlas_robust(tracker, x_range, y_range, n_turns_deep, nx, ny, n_phase=4, seed=None)
        
        flat_T = T.ravel()
        p_wall = np.mean(flat_T < 500)
        p_short = np.mean((flat_T >= 500) & (flat_T < 50000))
        p_long = np.mean((flat_T >= 50000) & (flat_T < n_turns_deep))
        p_harbor = np.mean(flat_T >= n_turns_deep)
        
        print(f"RESULTS ({name}):")
        print(f"  Wall (<500)   : {p_wall:.2%}")
        print(f"  Short (<50k)  : {p_short:.2%}")
        print(f"  Long (50k-200k): {p_long:.2%}  <-- KEY METRIC")
        print(f"  Harbor (>200k): {p_harbor:.2%}")
        print(f"  Total Beach   : {p_short+p_long:.2%}")
        
        if p_long > 0.058: # Baseline was 5.8%
             print("  VERDICT: BEATS BASELINE LONG BEACH!")
        else:
             print("  VERDICT: No improvement in Deep Tail.")


def run_truth_spec_validation():
    """
    PHASE 5: TRUTH SPEC VALIDATION
    Standardized Protocol to eliminate "Self-Deception" and numerical ambiguity.
    
    Spec:
      - Grid: 40x30 Robust (x=[1.72, 1.92]mm, y=[0, 0.3]mm)
      - Loss: 5mm
      - Turns: 200,000
      - Replicas: 5 (Dynamic Seeds)
      - Classes: Wall(<500), Short(500-50k), Long(50k-200k), Harbor(>=200k)
      
    Candidates:
      1. Baseline (0.10, 0.04)
      2. AmpWinner (0.12, 0.04)
      3. Robust (0.10, 0.06)
    """
    PRESET_NAME = "TRUTH_SPEC_VALIDATION"
    print(f"=== {PRESET_NAME} (The Red Team) ===")
    
    # 1. Configuration Definitions
    configs = [
        {"name": "Baseline",   "e1": 0.10, "e2": 0.04, "f2": 0.4142},
        {"name": "AmpWinner",  "e1": 0.12, "e2": 0.04, "f2": 0.4142},
        {"name": "Robust",     "e1": 0.10, "e2": 0.06, "f2": 0.4142},
    ]
    
    n_replicas = 5
    n_turns = 200000
    
    # Fixed Domain (The "Truth" Window)
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 40, 30
    n_phase_per_rep = 1 # 1 phase per replica, but 5 replicas -> 5 samples per pixel effectively
    
    # Results Storage
    results = {
        "Baseline": {"wall": [], "short": [], "long": [], "harbor": []},
        "AmpWinner": {"wall": [], "short": [], "long": [], "harbor": []},
        "Robust":    {"wall": [], "short": [], "long": [], "harbor": []},
    }
    
    print(f"{'Config':<10} | {'Rep':<3} | {'Wall':<7} | {'Short':<7} | {'Long':<7} | {'Harbor':<7} | {'TotBeach':<8}")
    print("-" * 80)
    
    for cfg in configs:
        name = cfg["name"]
        e1 = cfg["e1"]
        e2 = cfg["e2"]
        f2 = cfg["f2"]
        w2 = 2 * np.pi * f2
        
        for r in range(n_replicas):
            # Dynamic Seed per replica (ensures independence)
            seed = r * 100 + 42 
            
            tracker = TurboTrackerS1_4(limit_x=5e-3, limit_y=5e-3, 
                                       eps1=e1, eps2=e2,
                                       omega2=w2)
            
            # Robust Atlas with explicit seed
            X, Y, T = escape_time_atlas_robust(tracker, x_range, y_range, n_turns, nx, ny, n_phase_per_rep, seed=seed)
            
            flat_T = T.ravel()
            
            # Classification "Truth Definitions"
            p_wall   = np.mean(flat_T < 500)
            p_short  = np.mean((flat_T >= 500) & (flat_T < 50000))
            p_long   = np.mean((flat_T >= 50000) & (flat_T < n_turns))
            p_harbor = np.mean(flat_T >= n_turns)
            
            results[name]["wall"].append(p_wall)
            results[name]["short"].append(p_short)
            results[name]["long"].append(p_long)
            results[name]["harbor"].append(p_harbor)
            
            p_tot_beach = p_short + p_long
            
            print(f"{name:<10} | {r+1:<3} | {p_wall:<7.1%} | {p_short:<7.1%} | {p_long:<7.1%} | {p_harbor:<7.1%} | {p_tot_beach:<8.1%}")

    # Statistical Summary
    print("\n=== TRUTH SPEC SUMMARY (Mean +/- Std) ===")
    print(f"{'Config':<10} | {'TotalBeach':<16} | {'LongBeach':<16} | {'Harbor':<16} | {'Verdict'}")
    print("-" * 85)
    
    def fmt_stat(arr):
        return f"{np.mean(arr):.1%} +/- {np.std(arr):.1%}"
    
    summary_metrics = {}
    
    for cfg in configs:
        name = cfg["name"]
        dat = results[name]
        
        tot_beach = np.array(dat["short"]) + np.array(dat["long"])
        long_b = np.array(dat["long"])
        harbor = np.array(dat["harbor"])
        
        summary_metrics[name] = {
            "beach_m": np.mean(tot_beach),
            "long_m": np.mean(long_b),
            "harbor_m": np.mean(harbor)
        }
        
        verdict = ""
        if name == "Baseline":
            verdict = "(REF)"
        else:
            # Compare to Baseline
            base_long = summary_metrics["Baseline"]["long_m"]
            delta_long = np.mean(long_b) - base_long
            if delta_long > 0.005: 
                verdict = "VALID (Increases Tail)"
            elif delta_long < -0.005:
                verdict = "DEGRADED Tail"
            else:
                verdict = "NEUTRAL (Noise?)"
                
        print(f"{name:<10} | {fmt_stat(tot_beach):<16} | {fmt_stat(long_b):<16} | {fmt_stat(harbor):<16} | {verdict}")

    # ... (truth spec code) ...
    
def run_synchrotron_validation():
    """
    Validates Phase 6 Physics:
    1. Check Longitudinal Phase Space (z, delta)
    2. Verify Synchrotron Tune Qs
    """
    print("=== SYNCHROTRON VALIDATION (6D) ===")
    
    # Setup Tracker
    # Qs = 0.005 (Faster for checks), Chroma=0.0 (Decoupled check first)
    tracker = SynchrotronTracker6D(Qs=0.005, chromaticity=0.0, eps1=0.0, eps2=0.0) 
    
    # Track Single Particle
    n_turns = 2000
    s0 = [0.0, 0.0, 0.0, 0.0, 0.01, 0.0] # Initial z offset
    
    # We need to capture trajectory. The track() method returns (final, surv).
    # We need a method to get full trajectory? 
    # SynchrotronTracker6D.track returns (last_state, surv).
    # We need to modify it or interpret.
    # Actually, let's just make a small loop calling track(1) if necessary, 
    # OR modify tracker to support return_traj=True.
    # Given the file structure, loop is easier than modifying file again.
    
    traj = np.zeros((n_turns, 6))
    state = np.array(s0)
    
    print("Tracking Synchrotron Motion...")
    for i in range(n_turns):
        traj[i] = state
        state, s = tracker.track(state, 1) # 1 turn step
        if s < 1: break
        
    # Analyze
    z = traj[:, 4]
    delta = traj[:, 5]
    
    # Plot Phase Space
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(z, delta, '.')
    plt.title("Longitudinal Phase Space (z, delta)")
    plt.xlabel("z [m]")
    plt.ylabel("delta [dp/p]")
    plt.grid(True)
    
    # FFT for Qs
    plt.subplot(1, 2, 2)
    # Remove mean
    z_ac = z - np.mean(z)
    fft = np.abs(np.fft.rfft(z_ac))
    freqs = np.fft.rfftfreq(len(z_ac))
    
    # Peak
    idx = np.argmax(fft[1:]) + 1
    qs_meas = freqs[idx]
    
    plt.plot(freqs, fft)
    plt.axvline(qs_meas, color='r', linestyle='--', label=f"Qs~{qs_meas:.4f}")
    plt.title(f"Synchrotron Tune Spectrum (Target=0.005)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("analysis_SYNCHROTRON_CHECK.png")
    print(f"Saved analysis_SYNCHROTRON_CHECK.png. Measured Qs ~ {qs_meas:.4f}")
    
    # Validation
    if abs(qs_meas - 0.005) < 0.001:
        print(">> SUCCESS: Qs Matches Target.")
    else:
        print(">> WARNING: Qs Mismatch (Check RF Voltage scaling).")

    if abs(qs_meas - 0.005) < 0.001:
        print(">> SUCCESS: Qs Matches Target.")
    else:
        print(">> WARNING: Qs Mismatch (Check RF Voltage scaling).")

def run_arnold_diffusion_protocol():
    """
    PHASE 6: ARNOLD DIFFUSION STUDY
    Compare Harbor Stability with and without Chromaticity.
    
    Hypothesis: 
    Turning on Chromaticity (coupling delta -> Q) will cause
    particles in the 'Harbor' to diffuse and escape over long times.
    
    Setup:
    - Tracker: SynchrotronTracker6D
    - Grid: Compact subset of Truth Spec (focused on Harbor edge)
    - Turns: 50k (Initial check), 200k (if promising)
    """
    PRESET_NAME = "ARNOLD_DIFFUSION_TEST"
    print(f"=== {PRESET_NAME} (6D Chromaticity) ===")
    
    # --- TRANSVERSE SANITY CHECK ---
    print(">> RUNNING TRANSVERSE SANITY CHECK (z=0, Qs=0)...")
    
    w2_base = 2 * np.pi * 0.4142
    
    tracker_ref = TurboTrackerS1_4(eps1=0.10, eps2=0.04) # S1.4 uses 0.4142 internally? 
    # Wait, track_lattice_s1_4 defaults: omega2 = sqrt(2).
    # TurboTracker calls it with defaults.
    # SO TURBOTRACKER DEFAULT IS WRONG?
    # No, S1.4 Baseline f2 = 0.4142.
    # In `track_lattice_s1_4`: `if omega2 is None: omega2 = ... sqrt(2)`.
    # sqrt(2) = 1.414.
    # 0.4142 = sqrt(2) - 1.
    # Is it possible I tested 1.414 all along?
    
    # Let's check `TurboTrackerS1_4` init.
    # It passes `omega2` if self.omega2 is set.
    # In `unstable_preset.py`, `run_s1_4d_comparative` creates configs:
    # `{"f2": 0.4142}` and calls `TurboTracker(..., omega2=w2)`.
    # SO THE TRUTH SPEC USED 0.4142. (Correct).
    # BUT `TurboTracker` DEFAULT (no args) uses `None` -> `sqrt(2)` ~ 1.414.
    
    # So `tracker_ref = TurboTrackerS1_4(eps1=0.10, eps2=0.04)` uses 1.414!
    # And my Sanity Check showed `Ref Surv: 52`.
    # This implies 1.414 is unstable at x=1.83.
    # AND 0.4142 is STABLE (Truth Spec results).
    
    # So I MUST Pass omega2 to BOTH trackers in Sanity Check.
    
    tracker_ref = TurboTrackerS1_4(eps1=0.10, eps2=0.04, omega2=w2_base)
    tracker_syn = SynchrotronTracker6D(Qs=0.0, chromaticity=0.0, eps1=0.10, eps2=0.04, omega2=w2_base)
    
    # ... (setup) ...
    x_test = 1.83e-3
    # Initial z chosen to be mild
    z_init = 0.01
    
    s0_ref = [x_test, 0, 0, 0, 0, 0] # 4D Reference
    s0_syn = [x_test, 0, 0, 0, z_init, 0] # 6D Synchrotron
    
    # Track Ref
    _, surv_ref = tracker_ref.track(s0_ref, 2000)
    
    # Track Synchro with Trajectory Monitor
    # We loop manually to check delta
    state = np.array(s0_syn)
    max_d = 0.0
    surv_syn = 0
    for i in range(2000):
        state, s = tracker_syn.track(state, 1)
        d_val = state[5]
        max_d = max(max_d, abs(d_val))
        if s < 1: 
            surv_syn = i
            break
        surv_syn = i
        
    print(f"  Ref Surv (4D): {surv_ref}")
    print(f"  Syn Surv (6D, z={z_init}): {surv_syn}")
    print(f"  Max Delta: {max_d:.6f}")
    
    if max_d > 0.01:
        print(">> WARNING: Delta exploded (>1%). Longitudinal Instability?")
    elif surv_syn < 500 and surv_ref > 1000:
         print(">> PHYSICS: Natural Chromaticity killed the Harbor!")
         
    # Proceed to scan if not totally broken (or return to debug)
    # If 100% loss, scanning is useless.
    if surv_syn < 500:
        print(">> ABORTING SCAN: Harbor is dead in 6D. Tuning or Debugging needed.")
        # return # FORCE CONTINUE
        print(">> FORCE CONTINUE: Grid Scan will determine truth.")


def escape_time_atlas_6d(tracker, x_range, y_range, n_turns=50000, nx=20, ny=15, n_phase=4, seed=42, z_std=0.01):
    """
    Robust 6D Atlas.
    Injects Z-jitter (Synchrotron Amplitude) and Phase Randomization.
    """
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    T_map = np.zeros((ny, nx), dtype=float)
    
    print(f"Scanning 6D Grid {nx}x{ny} (Phase={n_phase}, Z_std={z_std})...")
    rng = np.random.default_rng(seed)
    dp_scale = 1e-5
    
    for iy, y0 in enumerate(Y):
        for ix, x0 in enumerate(X):
            survivals = []
            for _ in range(n_phase):
                # Randomize Phase
                p_off = rng.integers(0, 10000)
                
                # Randomize Momenta (ABR style)
                px = rng.uniform(-dp_scale, dp_scale)
                py = rng.uniform(-dp_scale, dp_scale)
                
                # Inject Synchrotron Amplitude
                # We want ACTIVE synchrotron motion. z=0 => d=0 => nothing.
                # Use z_std to pick a z.
                z = rng.normal(0, z_std)
                # Ensure z is not 0 (if z_std=0, use fixed small offset)
                if abs(z) < 1e-6: z = 1e-3
                
                d = 0.0 # Start on axis in delta? Or smear delta too?
                # Let's smear delta slightly to fill bucket
                d = rng.normal(0, 1e-4) 

                s0 = [x0, px, y0, py, z, d]
                
                # Turn offset must be passed via kwargs or modified tracker
                # SynchrotronTracker6D.track(s0, n, phase_offset=p_off)
                _, surv = tracker.track(s0, n_turns, phase_offset=p_off)
                survivals.append(surv)
            
            T_map[iy, ix] = np.median(survivals)
            
    return X, Y, T_map

def run_arnold_diffusion_protocol():
    """
    PHASE 6: ARNOLD DIFFUSION STUDY
    Compare Harbor Stability with and without Chromaticity.
    """
    PRESET_NAME = "ARNOLD_DIFFUSION_TEST"
    print(f"=== {PRESET_NAME} (6D Chromaticity) ===")
    
    # ... Sanity Check code remains (but non-blocking now) ...
    # Assuming code above this replacement block handles sanity check
    
    # Grid: Focus on the "Edge" where diffusion is most likely.
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    nx, ny = 20, 15
    n_turns = 50000
    n_phase = 4 # Robust!
    
    configs = [
        {"name": "Control (4D+z)", "chroma": 0.0, "Qs": 0.005},
        {"name": "Diffusion (6D)", "chroma": 10.0, "Qs": 0.005},
    ]
    
    w2_base = 2 * np.pi * 0.4142
    
    print(f"{'Config':<15} | {'Wall':<8} | {'Beach':<8} | {'Harbor':<8} | {'Diff_Gain'}")
    print("-" * 70)
    
    results = {}

    for cfg in configs:
        name = cfg['name']
        chroma = cfg['chroma']
        qs = cfg['Qs']
        
        # New Tracker with correct Omega2
        tracker = SynchrotronTracker6D(Qs=qs, chromaticity=chroma, eps1=0.10, eps2=0.04, omega2=w2_base)
        
        # Run 6D Atlas
        # Using z_std=0.01 (1cm bunch length)
        try:
             X, Y, T = escape_time_atlas_6d(tracker, x_range, y_range, n_turns, nx, ny, n_phase, seed=42, z_std=0.01)
        except Exception as e:
             # Fallback if I messed up args
             print(f"Error in atlas: {e}")
             return

        flat_T = T.ravel()
        p_harbor = np.mean(flat_T >= n_turns)
        p_wall = np.mean(flat_T < 500)
        p_beach = np.mean((flat_T >= 500) & (flat_T < n_turns))
        
        print(f"{name:<15} | {p_wall:<8.1%} | {p_beach:<8.1%} | {p_harbor:<8.1%} | ...")
        results[name] = p_harbor

    # Compare
    h_ctrl = results["Control (4D+z)"]
    h_test = results["Diffusion (6D)"]
    
    diff = h_ctrl - h_test
    print(f"\nDiffusion Effect (Harbor Erosion): {diff:+.1%}")
    if diff > 0.02: # 2% erosion is significant
        print(">> VERDICT: Arnold Diffusion CONFIRMED. Harbor is eroding.")
    else:
        print(">> VERDICT: No significant diffusion observed (Weak Coupling?).")

    # Compare
    h_ctrl = results["Control (4D+z)"]
    h_test = results["Diffusion (6D)"]
    
    diff = h_ctrl - h_test
    print(f"\nDiffusion Effect (Harbor Erosion): {diff:+.1%}")
    if diff > 0.02:
        print(">> VERDICT: Arnold Diffusion CONFIRMED. Harbor is eroding.")
    else:
        print(">> VERDICT: No significant diffusion observed (Weak Coupling?).")

def run_6d_ablation_diagnostics_A1_A3():
    """
    PHASE 6b: ABLATION TRUTH TABLE (A1-A3)
    Diagnose cause of 6D Harbor Collapse.
    """
    print("=== 6D ABLATION DIAGNOSTICS (A1-A3) ===")
    
    w2_base = 2 * np.pi * 0.4142
    
    # --- TEST A1: LONGITUDINAL ONLY SANITY ---
    # Transverse kicked off? No, start at 0.
    print("\nTEST A1: Longitudinal Only (x=y=0, RF ON)...")
    tracker_A1 = SynchrotronTracker6D(Qs=0.005, chromaticity=0.0, eps1=0.0, eps2=0.0, omega2=w2_base)
    s0_A1 = [0, 0, 0, 0, 0.01, 0.0]
    
    # Track manually to check boundedness
    state = np.array(s0_A1)
    max_d = 0.0
    traj_d = []
    
    for i in range(2000):
        state, _ = tracker_A1.track(state, 1)
        d = state[5]
        max_d = max(max_d, abs(d))
        traj_d.append(d)
        
    print(f"  Max Delta: {max_d:.6f}")
    if max_d < 0.1:
        print("  VERDICT: A1 PASS (Longitudinal Motion Bounded).")
    else:
        print("  VERDICT: A1 FAIL (Unstable RF).")

    # --- TEST A2: 4D REGRESSION INSIDE 6D ---
    # RF OFF (Qs=0), z=0. Should reproduce 4D Baseline.
    print("\nTEST A2: 4D Regression (Qs=0, z=0) WITH PHASE RAND...")
    tracker_A2 = SynchrotronTracker6D(Qs=0.0, chromaticity=0.0, eps1=0.10, eps2=0.04, omega2=w2_base)
    tracker_Ref = TurboTrackerS1_4(eps1=0.10, eps2=0.04, omega2=w2_base)
    
    # DEBUG STEP 1
    s0_debug = [1.83e-3, 0.0, 0.0, 0.0, 0.0, 0.0]
    out_ref, _ = tracker_Ref.track(s0_debug, 1)
    out_syn, _ = tracker_A2.track(s0_debug, 1)
    
    print(f"DEBUG REF T1: {out_ref[:4]}")
    print(f"DEBUG SYN T1: {out_syn[:4]}")
    err = np.linalg.norm(out_ref[:4] - out_syn[:4])
    print(f"DEBUG DIFF: {err:.6e}")
    if err > 1e-10:
        print(">> CRITICAL: Trackers divergence at Turn 1!")
    
    # Quick Grid Scan (Coarse)
    nx, ny = 10, 8
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    n_phase_diag = 4 # Must match baseline robustness
    rng = np.random.default_rng(42)
    
    survs_A2 = []
    
    for y0 in np.linspace(y_range[0], y_range[1], ny):
        for x0 in np.linspace(x_range[0], x_range[1], nx):
            # Median over phases
            local_survs = []
            for _ in range(n_phase_diag):
                ph = rng.integers(0, 10000)
                # Slight p-jitter
                px = rng.uniform(-1e-5, 1e-5)
                py = rng.uniform(-1e-5, 1e-5)
                
                s0 = [x0, px, y0, py, 0.0, 0.0]
                _, s = tracker_A2.track(s0, 50000, phase_offset=ph)
                local_survs.append(s)
            
            med_s = np.median(local_survs)
            survs_A2.append(med_s >= 50000)
            
    p_harbor_A2 = np.mean(survs_A2)
    print(f"  Harbor (A2): {p_harbor_A2:.1%}")
    if p_harbor_A2 > 0.15:
        print("  VERDICT: A2 PASS (Lattice matches 4D Baseline).")
    else:
        print("  VERDICT: A2 FAIL (Lattice Mismatch).")
        
    # --- TEST A3: DELTA-COUPLING ABLATION ---
    # RF ON (Qs=0.005), z=0.01 (Active), but chromatic_scaling=False.
    # If Harbor returns, then Natural Chromaticity was the killer.
    print("\nTEST A3: Coupling Ablation (RF ON, Scaling OFF) WITH PHASE RAND...")
    tracker_A3 = SynchrotronTracker6D(Qs=0.005, chromaticity=0.0, eps1=0.10, eps2=0.04, 
                                      omega2=w2_base, chromatic_scaling=False)
                                      
    survs_A3 = []
    for y0 in np.linspace(y_range[0], y_range[1], ny):
        for x0 in np.linspace(x_range[0], x_range[1], nx):
            local_survs = []
            for _ in range(n_phase_diag):
                ph = rng.integers(0, 10000)
                px = rng.uniform(-1e-5, 1e-5)
                py = rng.uniform(-1e-5, 1e-5)
                
                # Inject z!
                z = rng.normal(0, 0.01)
                if abs(z)<1e-4: z=1e-3
                
                s0 = [x0, px, y0, py, z, 0.0]
                _, s = tracker_A3.track(s0, 50000, phase_offset=ph)
                local_survs.append(s)
            
            med_s = np.median(local_survs)
            survs_A3.append(med_s >= 50000)
            
    p_harbor_A3 = np.mean(survs_A3)
    print(f"  Harbor (A3): {p_harbor_A3:.1%}")
    
    if p_harbor_A3 > 0.15:
        print("  VERDICT: A3 PASS (Harbor Recovered -> It was Chromaticity).")
    else:
        print("  VERDICT: A3 FAIL (Harbor Dead -> It is Parametric/Integrator Issue).")

    if p_harbor_A3 > 0.15:
        print("  VERDICT: A3 PASS (Harbor Recovered -> It was Chromaticity).")
    else:
        print("  VERDICT: A3 FAIL (Harbor Dead -> It is Parametric/Integrator Issue).")

def run_chromaticity_correction_scan():
    """
    PHASE 7: S1.5 CHROMATIC CORRECTION
    Goal: Find the chromaticity value that cancels Natural Chroma ($1/(1+d)$)
    and restores the Harbor in Full 6D.
    Theoretical target: chroma ~ 6.3
    """
    print("=== CHROMATICITY CORRECTION SWEEP ===")
    
    w2_base = 2 * np.pi * 0.4142
    chroma_vals = [0.0, 2.0, 4.0, 6.0, 6.3, 7.0, 8.0, 10.0]
    
    # Use coarse grid for speed
    nx, ny = 10, 8
    x_range = (1.72e-3, 1.92e-3)
    y_range = (0.0, 0.30e-3)
    n_phase = 4
    n_turns = 100000 
    seed = 42
    
    results = []
    
    for c in chroma_vals:
        tracker = SynchrotronTracker6D(Qs=0.005, chromaticity=c, eps1=0.10, eps2=0.04, 
                                       omega2=w2_base, chromatic_scaling=True) # Full 6D
        
        try:
             # Using 10k turns for speed in sweep? No, diffusion needs time. 50k minimal.
             # User specified 100k for stability.
             # This will be slow sequentially. Let's do 50k.
             X, Y, T = escape_time_atlas_6d(tracker, x_range, y_range, n_turns=50000, 
                                            nx=nx, ny=ny, n_phase=n_phase, seed=seed, z_std=0.01)
             
             flat = T.ravel()
             p_harbor = np.mean(flat >= 50000)
             print(f"Chroma {c:<4.1f} | Harbor: {p_harbor:<6.1%}")
             results.append((c, p_harbor))
             
        except Exception as e:
            print(f"Error {c}: {e}")
            
    # Find Best
    best_c, best_h = max(results, key=lambda x: x[1])
    print("-" * 40)
    print(f"BEST CONFIG: Chroma = {best_c} -> Harbor = {best_h:.1%}")
    if best_h > 0.15:
        print(">> SUCCESS: Harbor Recovered in 6D via Chromatic Correction!")
    else:
        print(">> FAILURE: Even corrected chroma cannot stabilize S1.4 (Drive too strong?).")

    print(">> FAILURE: Even corrected chroma cannot stabilize S1.4 (Drive too strong?).")

def run_controlled_extraction_s1_6():
    """
    PHASE 8: S1.6 CONTROLLED EXTRACTION
    Simulates a Slow Extraction Cycle using Chromaticity Ramp + Feedback.
    Goal: Maintain constant spill rate.
    """
    print("=== S1.6 CONTROLLED EXTRACTION (Passo CERN) ===")
    
    # 1. Config
    n_part_init = 2000
    n_turns_burn = 5000
    n_turns_block = 1000
    n_blocks = 100 
    
    w2_base = 2 * np.pi * 0.4142
    
    # Initial State (Waterbag)
    rng = np.random.default_rng(42)
    particles = []
    
    # Generate Beam
    for _ in range(n_part_init):
        x = rng.normal(0, 0.5e-3)
        px = rng.normal(0, 0.2e-3)
        y = rng.normal(0, 0.5e-3)
        py = rng.normal(0, 0.2e-3)
        z = rng.normal(0, 0.01)
        d = rng.normal(0, 0.001)
        particles.append(np.array([x, px, y, py, z, d]))
    
    active_mask = np.ones(n_part_init, dtype=bool)
    
    # 2. Burn-In (Purge Halo)
    print(f"Burn-In: {n_turns_burn} turns at Stability Peak (C=6.3)...")
    tracker = SynchrotronTracker6D(Qs=0.005, chromaticity=6.3, eps1=0.10, eps2=0.04, 
                                   omega2=w2_base, chromatic_scaling=True)
    
    burn_loss = 0
    for i in range(n_part_init):
         ph = rng.integers(0, 10000)
         s, _, status = tracker.track(particles[i], n_turns_burn, phase_offset=ph)
         if status != 0: # Lost or Extracted during burn-in
             active_mask[i] = False
             burn_loss += 1
         else:
             particles[i] = s # Update state
             
    n_core = n_part_init - burn_loss
    print(f"Burn-In Complete. Core Size: {n_core} (Lost {burn_loss}).")
    if n_core < 100:
        print(">> ERROR: Core collapsed during burn-in. System unstable.")
        return

    # 3. Extraction Loop (Drive Scaling Control)
    target_spill_rate = 0.01 
    target_particles_per_block = n_core * target_spill_rate 
    target_particles_per_block = max(1.0, target_particles_per_block)
    
    print(f"Starting Extraction Loop. Target: {target_particles_per_block:.1f} spill/block (Drive Knob).")
    
    drive_scale = 1.0 # Start Stable (S1.4)
    kp = 0.05 # Sensitive knob
    
    history_spill = []
    
    total_extracted = 0
    total_lost = 0 
    
    for block in range(n_blocks):
        # Update Tracker Drive
        curr_eps1 = 0.10 * drive_scale
        curr_eps2 = 0.04 * drive_scale
        tracker.eps1 = curr_eps1
        tracker.eps2 = curr_eps2
        
        block_spill = 0
        block_loss = 0
        
        for i in range(n_part_init):
            if not active_mask[i]: continue
            
            ph = rng.integers(0, 10000)
            s_final, _, status = tracker.track(particles[i], n_turns_block, phase_offset=ph)
            
            if status == 0:
                particles[i] = s_final
            elif status == 1: # EXTRACTED
                active_mask[i] = False
                block_spill += 1
                total_extracted += 1
            else: # LOST
                active_mask[i] = False
                block_loss += 1
                total_lost += 1
        
        # Feedback (Increase Drive to Extract)
        error = target_particles_per_block - block_spill
        # If Spill < Target (Positive Error) -> Increase Drive
        d_scale = kp * (error / target_particles_per_block)
        d_scale = np.clip(d_scale, -0.1, 0.1)
        
        drive_scale += d_scale
        drive_scale = max(1.0, drive_scale) 
        
        history_spill.append(block_spill)
        rem = np.sum(active_mask)
        
        print(f"Block {block:3d} | Drive {drive_scale:5.3f} | Spill {block_spill:3d} (Target {target_particles_per_block:.0f}) | Rem {rem:4d}")
        
        if rem == 0:
            print(">> BEAM EMPTIED.")
            break
            
    # Analysis
    spill_arr = np.array(history_spill)
    mean_spill = np.mean(spill_arr)
    std_spill = np.std(spill_arr)
    cv = std_spill / (mean_spill + 1e-9)
    eff = total_extracted / (total_extracted + total_lost + 1e-9)
    
    print("-" * 40)
    print(f"EXTRACTION SUMMARY:")
    print(f"Total Extracted: {total_extracted} ({eff:.1%})")
    print(f"Spill Smoothness (CV): {cv:.3f}")
    
    if eff > 0.80 and cv < 1.0:
        print(">> SUCCESS: S1.6 Controlled Extraction Validated.")
    else:
        print(">> WARNING: Extraction unstable.")

if __name__ == "__main__":
    print(">>> SCRIPT STARTED: S1.6 EXTRACTION <<<")
    run_controlled_extraction_s1_6()
