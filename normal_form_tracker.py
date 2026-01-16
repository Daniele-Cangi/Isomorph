import numpy as np
import sys

# Import S0.2/S0.3 stack
from bch_tpsa_tracker import TPSA, jacobian_at, sympl_error, shock_bch_compile_6d

# Assumes from S0.2/S0.3:
# - TPSA objects in map_seq: list[TPSA] length 6
# - jacobian_at(map6: list[TPSA], pt6: np.ndarray) -> 6x6
# - sympl_error(M: np.ndarray) -> float

J6 = np.array([
    [0, 1, 0, 0, 0, 0],
    [-1,0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0,-1,0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0,-1,0],
], dtype=float)

# -------------------------
# 1) Linear Normal Form (6D)
# -------------------------
def linear_normal_form(M, J=J6, tol_unit=1e-6):
    """
    Symplectic eigenbasis construction:
      - pick eigenvalues on unit circle with Im>0 => 3 modes
      - normalize eigenvectors u s.t. u^H J u = 2i
      - build real A = [Re(u1), Im(u1), Re(u2), Im(u2), Re(u3), Im(u3)]
    Returns: A, Ainv, tunes (nu array len=3), Mnorm = Ainv M A
    """
    w, V = np.linalg.eig(M.astype(complex))

    # select stable modes on unit circle with positive imag
    idx = []
    for i, lam in enumerate(w):
        if abs(abs(lam) - 1.0) < tol_unit and np.imag(lam) > 0:
            idx.append(i)

    if len(idx) < 3:
        print("Eigenvalues:", w)
        print("Abs(w):", np.abs(w))
        raise RuntimeError(
            f"Not enough stable unit-circle modes: found {len(idx)}. "
            "Your one-turn map is unstable/coupled in a way that needs handling."
        )

    # sort by phase advance
    mus = [(np.angle(w[i]) % (2*np.pi), i) for i in idx]
    mus.sort(key=lambda t: t[0])
    mus = mus[:3]

    cols = []
    nus  = []
    for mu, i in mus:
        v = V[:, i]
        s = v.conj().T @ J @ v  # should be ~ purely imaginary
        # normalize to u^H J u = 2i
        scale = np.sqrt((2j) / s)
        u = scale * v
        cols.append(np.real(u))
        cols.append(np.imag(u))
        nus.append(mu / (2*np.pi))

    A = np.column_stack(cols).astype(float)
    Ainv = np.linalg.inv(A)

    # sanity: A should be symplectic (approx)
    symA = np.linalg.norm(A.T @ J @ A - J)
    Mnorm = Ainv @ M @ A

    return A, Ainv, np.array(nus), Mnorm, symA

# -------------------------
# 2) Iterate TPSA one-turn map numerically
# -------------------------
def iterate_map_tpsa(map6, x0, n_turns):
    """
    map6: list[TPSA] length 6, each has .eval(pt6) -> float
    """
    x = np.array(x0, dtype=float)
    traj = np.zeros((n_turns, 6), dtype=float)
    for t in range(n_turns):
        traj[t] = x
        x = np.array([f.eval(x) for f in map6], dtype=float)
    return traj

# -------------------------
# 3) Tune estimator (FFT + quadratic peak)
# -------------------------
def estimate_tune_complex(h):
    """
    h: complex turn-by-turn signal (e.g., q + i p in normalized coords)
    returns tune in [0, 0.5]
    """
    h = np.asarray(h, dtype=complex)
    h = h - np.mean(h)

    N = len(h)
    H = np.fft.rfft(h)
    mag = np.abs(H)
    mag[0] = 0.0

    k = int(np.argmax(mag))
    if k <= 0 or k >= len(mag)-1:
        return k / N

    # quadratic interpolation around k on log-magnitude
    y0 = np.log(mag[k-1] + 1e-30)
    y1 = np.log(mag[k]   + 1e-30)
    y2 = np.log(mag[k+1] + 1e-30)
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-30:
        delta = 0.0
    else:
        delta = 0.5*(y0 - y2)/denom

    k_refined = k + delta
    tune = k_refined / N
    # constrain to [0,0.5] for safety
    return float(np.clip(tune, 0.0, 0.5))

# -------------------------
# 4) Nonlinear "Normal Form" proxies:
#    detuning vs actions via regression in normalized coords
# -------------------------
def actions_from_norm(w):
    """
    w: (N,6) in normalized coordinates blocks (q,p)
    Actions J = (q^2 + p^2)/2 for each mode
    """
    J1 = 0.5*(w[:,0]**2 + w[:,1]**2)
    J2 = 0.5*(w[:,2]**2 + w[:,3]**2)
    J3 = 0.5*(w[:,4]**2 + w[:,5]**2)
    return J1, J2, J3

def detuning_fit(map6, A, Ainv, base_state, grid_Jx, grid_Jy, n_turns=4096, burn=256):
    """
    Build initial conditions in normalized coords with (Jx, Jy),
    map back to physical coords via A, track, estimate tunes, fit:
      nu_x = nu_x0 + axx*Jx + axy*Jy
      nu_y = nu_y0 + ayx*Jx + ayy*Jy
    """
    rows = []
    targets = []

    for Jx in grid_Jx:
        for Jy in grid_Jy:
            # normalized initial (phases = 0)
            w0 = np.zeros(6, dtype=float)
            w0[0] = np.sqrt(2*Jx)
            w0[2] = np.sqrt(2*Jy)
            # keep longitudinal from base_state in normalized coords
            wb = Ainv @ np.array(base_state, dtype=float)
            w0[4] = wb[4]
            w0[5] = wb[5]

            x0 = A @ w0

            traj = iterate_map_tpsa(map6, x0, n_turns)
            w = (Ainv @ traj.T).T

            hx = w[burn:,0] + 1j*w[burn:,1]
            hy = w[burn:,2] + 1j*w[burn:,3]
            # hz if you want: w[4]+i w[5]

            nux = estimate_tune_complex(hx)
            nuy = estimate_tune_complex(hy)

            rows.append([1.0, Jx, Jy])
            targets.append([nux, nuy])

    X = np.array(rows, dtype=float)
    Y = np.array(targets, dtype=float)  # shape (K,2)

    # least squares: beta_x, beta_y
    beta_x, *_ = np.linalg.lstsq(X, Y[:,0], rcond=None)
    beta_y, *_ = np.linalg.lstsq(X, Y[:,1], rcond=None)

    # unpack
    model = {
        "nu_x0": beta_x[0], "dnu_x_dJx": beta_x[1], "dnu_x_dJy": beta_x[2],
        "nu_y0": beta_y[0], "dnu_y_dJx": beta_y[1], "dnu_y_dJy": beta_y[2],
    }
    return model

# -------------------------
# 5) Resonance spectrum (detect combination lines)
# -------------------------
def spectrum_lines(h, n_keep=16):
    """
    returns top spectral lines (freq, amp) from complex signal h
    """
    h = np.asarray(h, dtype=complex)
    h = h - np.mean(h)
    N = len(h)
    H = np.fft.rfft(h)
    mag = np.abs(H)
    mag[0] = 0.0
    idx = np.argsort(mag)[::-1][:n_keep]
    lines = [(float(i/N), float(mag[i])) for i in idx]
    lines.sort(key=lambda t: t[0])
    return lines

# -------------------------
# 6) SHOCK RUN: Normal Form report from map_seq
# -------------------------
def shock_normal_form(map_seq, order_label="S0.4", n_turns=4096):
    """
    - Computes M0 at origin
    - Extracts linear normal form
    - Tracks one reference particle, prints tunes
    - Fits detuning on a small amplitude grid
    """
    pt0 = np.zeros(6, dtype=float)
    M0 = jacobian_at(map_seq, pt0)
    eM = sympl_error(M0)
    
    # Debug Stability
    print(f"DEBUG: M0 Trace (x,y,z): {M0[0,0]+M0[1,1]}, {M0[2,2]+M0[3,3]}, {M0[4,4]+M0[5,5]}")
    w_eig, _ = np.linalg.eig(M0)
    print("DEBUG: Eigenvalues:", w_eig)
    print("DEBUG: Abs(w):", np.abs(w_eig))

    # Relax tolerance because BCH map has ~0.5% symplectic error
    A, Ainv, nus, Mnorm, symA = linear_normal_form(M0, tol_unit=1e-1)

    print(f"=== {order_label} NORMAL FORM (6D) ===")
    print(f"Symplectic error one-turn M0: {eM:.3e}")
    print(f"Symplectic error of A:        {symA:.3e}")
    print("Linear tunes (from eigenphase):", nus)

    # Track a reference particle (small amp)
    x0 = np.array([1e-3, 0.0, 1e-3, 0.0, 0.0, 1e-3], dtype=float)
    traj = iterate_map_tpsa(map_seq, x0, n_turns)
    w = (Ainv @ traj.T).T

    hx = w[:,0] + 1j*w[:,1]
    hy = w[:,2] + 1j*w[:,3]
    hz = w[:,4] + 1j*w[:,5]

    nux = estimate_tune_complex(hx[256:])
    nuy = estimate_tune_complex(hy[256:])
    nus_est = estimate_tune_complex(hz[256:])

    print("Tunes (FFT estimate):")
    print("  nu_x:", nux)
    print("  nu_y:", nuy)
    print("  nu_s:", nus_est)

    # Detuning fit grid (aggressive but small)
    grid_Jx = [0.5*(a*a) for a in [5e-4, 1e-3, 1.5e-3]]
    grid_Jy = [0.5*(a*a) for a in [5e-4, 1e-3, 1.5e-3]]

    det = detuning_fit(
        map_seq, A, Ainv,
        base_state=x0,
        grid_Jx=grid_Jx,
        grid_Jy=grid_Jy,
        n_turns=4096,
        burn=256
    )
    print("Detuning model (numerical normal form proxy):")
    for k,v in det.items():
        print(f"  {k}: {v}")

    # Resonance spectrum quick peek
    lines_x = spectrum_lines(hx[256:], n_keep=12)
    lines_y = spectrum_lines(hy[256:], n_keep=12)
    print("Top spectral lines (x):", lines_x)
    print("Top spectral lines (y):", lines_y)

    return {
        "M0": M0,
        "A": A, "Ainv": Ainv,
        "nus_linear": nus,
        "nus_est": (nux, nuy, nus_est),
        "detuning": det,
        "traj": traj,
        "wtraj": w,
        "Mnorm": Mnorm
    }

if __name__ == "__main__":
    # Get the map from S0.3 (BCH compiled sequence)
    print("Generating Lattice Map via S0.3 BCH stack...")
    # Use explicit sequence map for best accuracy to test Normal Form Engine 
    # (though BCH compiled map works too with O(4) errors)
    try:
        shock_res = shock_bch_compile_6d(order=4, exp_terms=4) 
        
        # Run S0.4 Normal Form Analysis
        nf_res = shock_normal_form(shock_res["map_seq"], n_turns=128)
    except Exception as e:
        print(f"\nCRITICAL FAILURE IN NORMAL FORM: {e}")
        import traceback
        traceback.print_exc()
