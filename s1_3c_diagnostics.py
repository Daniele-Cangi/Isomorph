
import numpy as np
import matplotlib.pyplot as plt

def analyze_diagnostics():
    try:
        data = np.load("escape_time_VALIDATION_ADAPTIVE_ZOOM.npz")
        T = data['T_deep']
    except FileNotFoundError:
        print("Data not found.")
        return

    n = len(T)
    n_turns = 200000
    
    # 1. Censored Fraction
    n_censored = np.sum(T >= n_turns)
    censored_frac = n_censored / n
    
    # 2. Survival Function
    times = np.sort(T)
    prob = 1.0 - np.arange(n)/n
    
    def get_val(t_target):
        idx = np.searchsorted(times, t_target)
        if idx >= n: return 0.0
        return prob[idx]
        
    s_1e3 = get_val(1000)
    s_1e4 = get_val(10000)
    s_1e5 = get_val(100000)
    
    # 3. Slope Hat (Linear fit on log-log between 1e3 and 1e5)
    # Filter points in range
    mask = (times >= 1000) & (times <= 100000)
    if np.sum(mask) > 5:
        log_t = np.log10(times[mask])
        log_s = np.log10(prob[mask])
        
        # Polyfit degree 1
        slope, intercept = np.polyfit(log_t, log_s, 1)
    else:
        slope = 0.0
        
    print(f"DIAGNOSTICS_RESULT:")
    print(f"S(1e4): {s_1e4:.4f}")
    print(f"S(1e5): {s_1e5:.4f}")
    print(f"Censored_Frac: {censored_frac:.4f}")
    print(f"Max_Turns: {n_turns}")
    print(f"Slope_Hat: {slope:.4f}")
    print(f"Beach_Candidates_Def: 500 <= T < 50000 (from ABR-50k)")

if __name__ == "__main__":
    analyze_diagnostics()
