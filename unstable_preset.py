import numpy as np
import matplotlib.pyplot as plt
from symplectic_tracker import (
    SymplecticTracker6D as SymplecticTracker,
    QuadrupoleKick, Drift, SextupoleKick, SkewQuadrupoleKick, OctupoleKick
)

PRESET_NAME = "CHAOTIC_EDGE_k3m1e4"

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

def run_poincare_high_res(tracker, x0_list, n_turns=50000, stride=10, discard=1000):
    print(f"\n=== S0.11 HIGH-RES POINCARE ({n_turns} turns) ===")
    
    for x0 in x0_list:
        print(f"Tracking x0 = {x0*1000:.2f} mm...")
        init = PhaseSpaceState(x=x0, px=0.0, y=0.0, py=0.0)
        traj, turns = tracker.track(init, n_turns)
        
        # Prepare data for plotting
        if turns > discard:
            # Slice and decimating
            valid_traj = traj[discard:turns:stride]
            x_pts = valid_traj[:,0]*1000
            px_pts = valid_traj[:,1]*1000
            y_pts = valid_traj[:,2]*1000
            py_pts = valid_traj[:,3]*1000
            
            # Metrics
            nu1, nu2 = eigentunes_from_traj(traj[:min(turns, 4096)]) if turns > 256 else (0,0)
            
            plt.figure(figsize=(12, 5))
            
            # Plane 1: X-PX
            plt.subplot(1, 2, 1)
            plt.scatter(x_pts, px_pts, s=0.1, c='black', alpha=0.5)
            plt.title(f"X-PX (x0={x0*1000:.2f}mm)\nTunes: {nu1:.4f}, {nu2:.4f}")
            plt.xlabel("x [mm]")
            plt.ylabel("px [mrad]")
            plt.grid(True, alpha=0.3)
            
            # Plane 2: Y-PY
            plt.subplot(1, 2, 2)
            plt.scatter(y_pts, py_pts, s=0.1, c='blue', alpha=0.5)
            plt.title(f"Y-PY (Coupling View)\nSurvival: {turns/n_turns*100:.1f}%")
            plt.xlabel("y [mm]")
            plt.ylabel("py [mrad]")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fname = f"poincare_{PRESET_NAME}_x{x0*1000:.2f}mm.png"
            plt.savefig(fname, dpi=150)
            print(f"  Saved {fname}")
        else:
            print(f"  Lost immediately (turns={turns})")

def run_preset():
    print(f"=== {PRESET_NAME} SETUP ===")
    
    # S0.10 Config
    k3_magic = -1.0e4
    lat = build_resonance_lattice(k3=k3_magic)
    tracker = ApertureTracker(lat, aperture_x=5e-3, aperture_y=5e-3)
    
    # S0.11 Execution (Tri-state check)
    amplitudes = [0.8e-3, 1.6e-3, 1.70e-3]
    
    # Run High-Res
    run_poincare_high_res(tracker, amplitudes, n_turns=50000, stride=10, discard=1000)
    
    # Quick Atlas Scan for Context (Optional, but good for output log)
    print("\n--- Quick Boundary Check (1000 turns) ---")
    x_scan = [1.6e-3, 1.65e-3, 1.7e-3, 1.75e-3, 1.8e-3]
    for x in x_scan:
        init = PhaseSpaceState(x=x, px=0.0, y=0.0, py=0.0)
        traj, turns = tracker.track(init, 2000)
        nu1, nu2 = eigentunes_from_traj(traj) if turns > 512 else (0,0)
        D1, D2 = diffusion_eigentunes(traj) if turns > 2048 else (0,0)
        status = "ALIVE" if turns == 2000 else f"LOST@{turns}"
        print(f"x={x*1000:.2f}mm | {status:<8} | nu1={nu1:.4f} | D={D1:.2e}")

if __name__ == "__main__":
    run_preset()
