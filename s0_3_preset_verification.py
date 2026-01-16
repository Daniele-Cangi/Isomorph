import s0_2_coupled_normal_form as s0_2
import numpy as np
import sympy as sp

# PRESET CONFIG (Match unstable_preset.py)
s0_2.Ld = 1.0
s0_2.kq = 0.632   # MATCH PRESET (Target nu ~ 0.255)
s0_2.ks = 0.15    # MATCH PRESET (Higher coupling)
s0_2.k2 = 10.0    # MATCH PRESET
s0_2.ORDER = 3    # Keep 3 for RDTs

# Monkeypatch build_map to match unstable_preset lattice
# Sequence: Qf(+k), Drift, Sx(+k2), Drift, Qd(-k), Drift, Sx(-k2), Drift, Skew(ks), Drift
def build_preset_map():
    state = (s0_2.x, s0_2.px, s0_2.y, s0_2.py)
    
    # 1. Qf
    state = s0_2.quad_kick(+s0_2.kq, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)
    
    # 2. Drift
    state = s0_2.drift(s0_2.Ld, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)
    
    # 3. Sx (+)
    state = s0_2.sext_kick(+s0_2.k2, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 4. Drift
    state = s0_2.drift(s0_2.Ld, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 5. Qd
    state = s0_2.quad_kick(-s0_2.kq, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 6. Drift
    state = s0_2.drift(s0_2.Ld, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 7. Sx (-)
    state = s0_2.sext_kick(-s0_2.k2, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 8. Drift
    state = s0_2.drift(s0_2.Ld, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 9. Skew
    state = s0_2.skew_quad_kick(s0_2.ks, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)

    # 10. Drift
    state = s0_2.drift(s0_2.Ld, state)
    state = tuple(s0_2.truncate(s, s0_2.VXY, s0_2.ORDER) for s in state)
    
    return state

s0_2.build_map = build_preset_map

if __name__ == "__main__":
    print(f"=== S0.3 PRESET VERIFICATION (k2={s0_2.k2}, ks={s0_2.ks}) ===")
    
    # 1. Build Map (Uses updated globals)
    one_turn = s0_2.build_map()
    Jsym = s0_2.jacobian_map(one_turn, s0_2.VXY)
    M = s0_2.eval_at_zero(Jsym, s0_2.VXY)
    
    # 2. Linear Normal Form
    try:
        nf = s0_2.coupled_linear_normal_form(M)
        C, Cin = nf["C"], nf["Cin"]
        mu = nf["mu"]
        print(f"Tunes: {nf['tunes'][0]:.4f}, {nf['tunes'][1]:.4f}")
        
        # 3. Map in Normal Coords (eta)
        print("Transforming map to normal coordinates...")
        eta_prime = s0_2.map_in_eta(one_turn, C, Cin)
        
        # 4. Construct Delta (eta' - R eta)
        eta = sp.Matrix(s0_2.VA)
        rot = sp.diag(sp.exp(s0_2.I*mu[0]), sp.exp(s0_2.I*mu[1]), sp.exp(s0_2.I*mu[2]), sp.exp(s0_2.I*mu[3]))
        delta = sp.Matrix([s0_2.truncate(eta_prime[i] - rot[i,i]*eta[i], s0_2.VA, s0_2.ORDER) for i in range(4)])

        # 5. S0.3 Run (Congugation + Detuning + RDTs)
        s0_2.s0_3_run(eta_prime, delta, mu)
        
    except Exception as e:
        print(f"Normal Form Failed: {e}")
