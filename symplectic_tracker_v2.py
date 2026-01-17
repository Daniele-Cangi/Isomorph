
import numpy as np
import math

# =========================
# TURBO TRACKER V2.1 (Fixed Lattice)
# =========================

def track_lattice_s1_4(start_state, n_turns, limit_x=5e-3, limit_y=5e-3, 
                       k_quad=0.632, k3=-1.0e4, ks=0.15,
                       k2_base=10.0, eps1=0.10, eps2=0.04, 
                       omega1=None, omega2=None, phase_offset=0.0):
    
    # Pre-compute constants
    if omega1 is None: omega1 = 2*np.pi * (np.sqrt(5.0)-1.0)/2.0
    if omega2 is None: omega2 = 2*np.pi * np.sqrt(2.00)
    
    L = 1.0 
    L_half = 0.5
    
    # Unpack state
    x = start_state[0]
    px = start_state[1]
    y = start_state[2]
    py = start_state[3]
    z = start_state[4]
    d = start_state[5]
    
    kq_apply = k_quad
    k3_half_fact = (k3 / 2.0) / 6.0 # Octupole factor: (k3/2)/6
    ks_apply = ks
    
    # Pre-calc drift factors? 
    # Drift is L * px / (1+d). 
    # Since d is constant (no RF), inv is constant.
    inv = 1.0 / (1.0 + d)
    
    surv = n_turns
    
    for turn in range(n_turns):
        eff_turn = turn + phase_offset
        cos1 = math.cos(omega1 * eff_turn)
        cos2 = math.cos(omega2 * eff_turn)
        
        # Combined Modulation Factor M = (1 + eps1*c1 + eps2*c2)
        mod_factor = 1.0 + eps1 * cos1 + eps2 * cos2
        
        # === F-BLOCK ===
        # 1. Quad (+k)
        px -= kq_apply * x
        py += kq_apply * y
        
        # 2. Drift (L/2)
        x += L_half * px * inv
        y += L_half * py * inv
        
        # 3. Octupole (k3/2)
        # dpx = -fact * (x^3 - 3xy^2)
        # dpy = +fact * (3x^2y - y^3)
        x2 = x*x; y2 = y*y
        px -= k3_half_fact * (x*x2 - 3.0*x*y2)
        py += k3_half_fact * (3.0*x2*y - y*y2)
        
        # 4. Drift (L/2)
        x += L_half * px * inv
        y += L_half * py * inv
        
        # 5. Sextupole (+k2 * mod)
        # dpx = -(k2/2)(x^2 - y^2)
        # dpy = k2 * x * y
        k2_curr = k2_base * mod_factor
        x2 = x*x; y2 = y*y
        px -= 0.5 * k2_curr * (x2 - y2)
        py += k2_curr * x * y
        
        # 6. Drift (L)
        x += L * px * inv
        y += L * py * inv
        
        # === D-BLOCK ===
        # 7. Quad (-k)
        px -= (-kq_apply) * x
        py += (-kq_apply) * y
        
        # 8. Drift (L/2)
        x += L_half * px * inv
        y += L_half * py * inv
        
        # 9. Octupole (k3/2)
        x2 = x*x; y2 = y*y
        px -= k3_half_fact * (x*x2 - 3.0*x*y2)
        py += k3_half_fact * (3.0*x2*y - y*y2)
        
        # 10. Drift (L/2)
        x += L_half * px * inv
        y += L_half * py * inv
        
        # 11. Sextupole (-k2 * mod) - Note: -k2 base!
        k2_curr = -k2_base * mod_factor
        x2 = x*x; y2 = y*y
        px -= 0.5 * k2_curr * (x2 - y2)
        py += k2_curr * x * y
        
        # 12. Drift (L)
        x += L * px * inv
        y += L * py * inv
        
        # === COUPLING ===
        # 13. Skew Quad (ks)
        # px' = px - ks * y
        # py' = py - ks * x
        px_new = px - ks_apply * y
        py_new = py - ks_apply * x
        px = px_new
        py = py_new
        
        # 14. Drift (L)
        x += L * px * inv
        y += L * py * inv
        
        # === CHECKS ===
        # Aperture
        if abs(x) > limit_x or abs(y) > limit_y:
            surv = turn
            break
            
        # Nan
        if abs(x) > 1e10:
            surv = turn
            break
            
    return surv

class TurboTrackerS1_4:
    def __init__(self, limit_x=5e-3, limit_y=5e-3, eps1=0.10, eps2=0.04):
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.eps1 = eps1
        self.eps2 = eps2
        self.turn_offset = 0 # Can be set for phase randomization
        
    def track(self, s0, n_turns):
        # s0 is PhaseSpaceState or array.
        if hasattr(s0, 'x'):
            arr = np.array([s0.x, s0.px, s0.y, s0.py, s0.z, s0.delta])
        else:
            arr = np.array(s0)
            
        surv = track_lattice_s1_4(arr, n_turns, 
                                  limit_x=self.limit_x, limit_y=self.limit_y,
                                  eps1=self.eps1, eps2=self.eps2,
                                  phase_offset=self.turn_offset)
                                  
        return None, surv # Return None for traj to save RAM
