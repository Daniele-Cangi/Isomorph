import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Arrow, FancyArrowPatch

# --- CONFIGURAZIONE FISICA & NETWORKING ---
np.random.seed(42) # Deterministic Physics for Reproducibility

C = 1.0           
G = 1.0           
RS_SCALE = 0.15   

# Parametri Simulazione
DT = 0.05
STEPS = 3000
# Tuning Thrust: 0.075 => v_ss ~ 0.37 (Sweet spot: accumulo stress significativo senza saturare C)
ENGINE_THRUST = 0.075 
GRAV_SOFTENING = 0.6 
DRAG_SCALE = 8.0     

# --- PARAMETRI TOPOLOGIA DINAMICA (ER=EPR) ---
SPACETIME_TEAR_THRESHOLD = 7.5  # Soglia di stress accumulato per triggerare il wormhole
WORMHOLE_TTL = 150              # Durata del tunnel in step (Lifecycle)
# Upfront Cost: % di Velocità persa istantaneamente all'apertura del tunnel
EXOTIC_MATTER_COST = 0.50       # 50% Energy Dump richiesto per strappare la topologia
TUNNEL_RELIEF_VALUE = 5.0       # Quanto stress viene "scaricato" aprendo il tunnel

class Wormhole:
    def __init__(self, exit_pos):
        self.entry = np.array([0.0, 0.0]) # Placeholder, set dynamically
        self.exit = np.array(exit_pos)
        self.radius = 1.2 # Raggio di cattura (Bocca)
        self.active = False
        self.ttl = 0
        self.transported_count = 0

    def open_at(self, pos):
        """Strappa lo spaziotempo alle coordinate pos."""
        self.entry = np.array(pos)
        self.active = True
        self.ttl = WORMHOLE_TTL

    def update(self):
        if not self.active: return
        self.ttl -= 1
        if self.ttl <= 0:
            self.collapse()

    def collapse(self):
        self.active = False
        self.ttl = 0

    def check_traversal(self, packet):
        if not self.active: return False
        
        # Distanza dall'entrata
        dist = np.linalg.norm(np.array([packet.x, packet.y]) - self.entry)
        
        # Se siamo dentro la bocca del wormhole (Event Horizon artificiale)
        if dist < self.radius:
            # TELETRASPORTO (Tunneling)
            # Offset random minimo all'uscita per evitare sovrapposizioni esatte
            packet.x = self.exit[0] + np.random.uniform(-0.1, 0.1)
            packet.y = self.exit[1] + np.random.uniform(-0.1, 0.1)
            packet.has_traversed_wormhole = True
            
            # Policy: One-shot tunnel. Collassa dopo l'uso per conservazione energia.
            self.transported_count += 1
            self.collapse() 
            return True
        return False

class KerrNode:
    def __init__(self, x, y, mass, spin):
        self.x = x
        self.y = y
        self.mass = mass
        self.spin = spin
        self.rs = (2 * G * self.mass * RS_SCALE) / (C**2)
        self.a = self.spin / (self.mass + 1e-9) 

    def get_forces(self, px, py):
        dx = self.x - px
        dy = self.y - py
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        # Consistent Physics: Softened Gravity
        # F = G*M / (r^2 + eps^2)
        f_grav_mag = (G * self.mass) / (r2 + GRAV_SOFTENING**2)
        
        if r < 1e-3: dir_x, dir_y = 0, 0
        else: dir_x, dir_y = dx / r, dy / r
        
        ax_grav = f_grav_mag * dir_x
        ay_grav = f_grav_mag * dir_y

        # Frame Dragging (Decay 1/r^3)
        denom = r**3 + 1.0 
        drag_mag = (self.rs * self.a * C * DRAG_SCALE) / denom
        
        ax_drag = -dir_y * drag_mag
        ay_drag = dir_x * drag_mag
        
        return (ax_grav, ay_grav), (ax_drag, ay_drag)

class Packet:
    def __init__(self, start_x, start_y, target_x, target_y, mode="NEWTONIAN"):
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.mode = mode 
        self.target_x = target_x
        self.target_y = target_y
        self.history = [(start_x, start_y)]
        self.status = "IN_FLIGHT"
        
        # Telemetria
        self.steps_taken = 0
        self.path_length = 0.0
        self.min_rs_dist = float('inf')
        self.congestion_integral = 0.0
        self.has_traversed_wormhole = False
        self.wormhole_activation_step = -1
        self.wormhole_entry_pos = None # Stores precise entry coordinate

    def update(self, nodes, wormhole=None):
        if self.status != "IN_FLIGHT": return
        self.steps_taken += 1

        # --- 0. Wormhole Interaction (Passive Entry) ---
        # Caso in cui il tunnel è già aperto (da altri o persistente)
        if wormhole and wormhole.active:
            if wormhole.check_traversal(self):
                self.history.append((self.x, self.y)) # Mark jump
                return

        # --- 1. Motore ---
        dx_target = self.target_x - self.x
        dy_target = self.target_y - self.y
        dist_target = np.sqrt(dx_target**2 + dy_target**2)
        
        if dist_target < 0.5:
            self.status = "DELIVERED"
            return

        ax_engine = (dx_target / dist_target) * ENGINE_THRUST
        ay_engine = (dy_target / dist_target) * ENGINE_THRUST
        
        self.vx += ax_engine * DT
        self.vy += ay_engine * DT

        # --- 2. Fisica Ambientale ---
        total_congestion_step = 0.0
        
        for node in nodes:
            dist_sq = (self.x - node.x)**2 + (self.y - node.y)**2
            dist_node = np.sqrt(dist_sq)
            
            margin = dist_node - node.rs
            if margin < self.min_rs_dist: self.min_rs_dist = margin

            if dist_node < node.rs:
                self.status = "TIMEOUT"
                return

            # Consistent Metric: Usa lo stesso softening della forza
            total_congestion_step += node.mass / (dist_sq + GRAV_SOFTENING**2)

            (gx, gy), (dx, dy) = node.get_forces(self.x, self.y)
            self.vx += gx * DT
            self.vy += gy * DT

            if self.mode == "RELATIVISTIC" or self.mode == "QUANTUM_TUNNEL":
                self.vx += dx * DT
                self.vy += dy * DT

        self.congestion_integral += total_congestion_step * DT

        # --- 3. Dynamic Topology Trigger (The "Tear") ---
        # Solo il pacchetto QUANTUM ha l'hardware per aprire il tunnel
        if self.mode == "QUANTUM_TUNNEL" and wormhole and not wormhole.active and not self.has_traversed_wormhole:
            if self.congestion_integral > SPACETIME_TEAR_THRESHOLD:
                # TRIGGER: Apri wormhole QUI e ORA
                wormhole.open_at((self.x, self.y))
                self.wormhole_entry_pos = (self.x, self.y) # Store exact coordinate
                self.wormhole_activation_step = self.steps_taken
                
                # COSTO UPFRONT: Drena energia cinetica per aprire il buco
                # Penalità immediata alla velocità corrente
                self.vx *= (1.0 - EXOTIC_MATTER_COST)
                self.vy *= (1.0 - EXOTIC_MATTER_COST)

                # ATTRAVERSAMENTO IMMEDIATO: Non aspettare il prossimo tick
                if wormhole.check_traversal(self):
                    self.history.append((self.x, self.y)) # Mark jump immediately
                
                # Relief
                self.congestion_integral = max(0.0, self.congestion_integral - TUNNEL_RELIEF_VALUE)

        # --- 4. Attrito & Speed Cap ---
        self.vx *= (1.0 - 0.2 * DT)
        self.vy *= (1.0 - 0.2 * DT)
        v_sq = self.vx**2 + self.vy**2
        if v_sq > C**2:
            scale = C / np.sqrt(v_sq)
            self.vx *= scale
            self.vy *= scale

        self.x += self.vx * DT
        self.y += self.vy * DT
        
        # Calcolo path length
        prev_pos = self.history[-1]
        step_dist = np.sqrt((self.x - prev_pos[0])**2 + (self.y - prev_pos[1])**2)
        # Non contare il teletrasporto come distanza percorsa fisicamente
        if step_dist < 5.0: 
            self.path_length += step_dist

        self.history.append((self.x, self.y))

def simulate_routing():
    # Setup Nodi: Massicci per indurre stress
    nodes = [
        KerrNode(x=-3.0, y=-1.0, mass=4.2, spin=0.0),   
        KerrNode(x=3.0, y=1.0, mass=3.8, spin=12.0),    
    ]
    
    start_x, start_y = 0.0, -9.0
    target_x, target_y = 0.0, 9.0
    
    # Il wormhole target è fisso (Safe Zone), l'entry è dinamica
    wormhole = Wormhole(exit_pos=(0, 6))
    
    p_kerr = Packet(start_x, start_y, target_x, target_y, mode="RELATIVISTIC")
    p_quantum = Packet(start_x, start_y, target_x, target_y, mode="QUANTUM_TUNNEL")
    
    # Esecuzione
    for _ in range(STEPS):
        if wormhole.active: wormhole.update()
        p_kerr.update(nodes) 
        p_quantum.update(nodes, wormhole) 

    # --- VISUALIZZAZIONE ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#050505')

    # Campo Background
    grid_size = 30
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    TotalX, TotalY = np.zeros_like(X), np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            for node in nodes:
                (gx, gy), (dx, dy) = node.get_forces(X[i,j], Y[i,j])
                TotalX[i,j] += gx + dx
                TotalY[i,j] += gy + dy
    ax.streamplot(X, Y, TotalX, TotalY, color='#223344', density=0.7, linewidth=0.5, arrowsize=0.5)

    # Nodi
    for node in nodes:
        ax.add_artist(Circle((node.x, node.y), node.rs, color='#440000', alpha=0.9, zorder=10))
        if node.spin != 0:
            ax.add_artist(Circle((node.x, node.y), node.rs * 2.0, color='cyan', alpha=0.08, zorder=5))

    # Wormhole Visualization
    wh_color = '#00ffcc'
    
    # Se il wormhole è stato attivato (controlliamo se abbiamo salvato la posizione di ingresso)
    if p_quantum.wormhole_entry_pos is not None:
        entry_pos = p_quantum.wormhole_entry_pos

        ax.add_artist(Circle(entry_pos, wormhole.radius, color=wh_color, fill=False, linestyle='--', linewidth=2))
        ax.text(entry_pos[0], entry_pos[1]-1.5, f"TEAR POINT\n(Stress > {SPACETIME_TEAR_THRESHOLD})", color=wh_color, fontsize=8, ha='center')
        
        # Exit Point
        ax.add_artist(Circle(wormhole.exit, wormhole.radius, color=wh_color, fill=False, linestyle='-', linewidth=2))
        ax.text(wormhole.exit[0], wormhole.exit[1]+1.5, "EXIT POINT", color=wh_color, fontsize=8, ha='center')

        # Connection Line
        throat = FancyArrowPatch(posA=entry_pos, posB=wormhole.exit, 
                                 connectionstyle="arc3,rad=-0.4", 
                                 color=wh_color, arrowstyle='simple', alpha=0.3, linewidth=5, zorder=2)
        ax.add_artist(throat)

    # Traiettorie
    # Kerr
    hist_k = np.array(p_kerr.history)
    if len(hist_k) > 0:
        ax.plot(hist_k[:,0], hist_k[:,1], color='#ffcc00', linewidth=2, label='Kerr (L3 Routing)', alpha=0.8)

    # Quantum
    hist_q = np.array(p_quantum.history)
    if len(hist_q) > 0:
        # Split line on jump using stored activation knowledge if available
        # Fallback to distance split for robust visualization
        dists = np.sqrt(np.sum(np.diff(hist_q, axis=0)**2, axis=1))
        jumps = np.where(dists > 5.0)[0]
        
        if len(jumps) > 0:
            idx = jumps[0]
            # Pre-tunnel
            ax.plot(hist_q[:idx+1,0], hist_q[:idx+1,1], color='#ff00ff', linewidth=2.5, label='Quantum (L4 Overlay)', alpha=1.0)
            # Post-tunnel
            ax.plot(hist_q[idx+1:,0], hist_q[idx+1:,1], color='#ff00ff', linewidth=2.5, alpha=0.6) # Fade after tunnel implies stability
            ax.plot(hist_q[idx+1,0], hist_q[idx+1,1], 'w*', markersize=12, markeredgecolor='#ff00ff')
        else:
            ax.plot(hist_q[:,0], hist_q[:,1], color='#ff00ff', linewidth=2.5, label='Quantum (No Activation)', alpha=1.0)

    ax.plot(target_x, target_y, 'g+', markersize=15, markeredgewidth=2)

    # Stats Panel
    def stats(p, name):
        warp = "YES" if p.has_traversed_wormhole else "NO"
        return (f"{name}\n"
                f"Status: {p.status}\n"
                f"Time: {p.steps_taken*DT:.1f}s\n"
                f"Stress: {p.congestion_integral:.1f}\n"
                f"Tunnel: {warp}")

    ax.text(-9.5, -9.5, stats(p_kerr, "KERR AGENT"), color='#ffcc00', fontsize=9, bbox=dict(facecolor='#222', alpha=0.8))
    ax.text(4.5, -9.5, stats(p_quantum, "QUANTUM AGENT"), color='#ff00ff', fontsize=9, bbox=dict(facecolor='#222', alpha=0.8))

    # Configuration Info
    conf_txt = f"THRESHOLD: {SPACETIME_TEAR_THRESHOLD} | UPFRONT COST: {EXOTIC_MATTER_COST*100}% Velocity Dump"
    ax.text(0, 9.5, conf_txt, color='gray', fontsize=8, ha='center', bbox=dict(facecolor='black'))

    ax.set_title("PROJECT ISOMORPH: Dynamic Topology & Upfront Cost Tunneling", color='white', pad=15)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_routing()
