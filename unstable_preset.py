import numpy as np
import matplotlib.pyplot as plt
from symplectic_tracker import (
    SymplecticTracker6D as SymplecticTracker,
    QuadrupoleKick, Drift, SextupoleKick, SkewQuadrupoleKick, OctupoleKick,
    ModulatedSextupoleKick
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
            
    return X, Y, T

# --- S1.3 HEAVY-TAIL ANALYSIS ---
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

if __name__ == "__main__":
    #run_preset() # Old preset
    run_mutation_c() # New Mutation



if __name__ == "__main__":
    # Select which experiment to run
    # run_preset() # S0.10 - S1.2
    run_mutation_c() # Mutation C
