import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
dt = 0.01  # Time step (s)
total_time = 5.0  # Total simulation time (s)
time_vector = np.arange(0, total_time, dt)
n_points = len(time_vector)

# --- Reference Trajectory (Polynomial + Sinusoid) ---
traj_duration = 3.0

a_poly = -20.0 / 9.0
b_poly = 10.0
c_poly = 0.0
d_poly = 10.0

A_sin = 8.0
f_sin = 1.5
omega_sin = 2 * np.pi * f_sin
decay_sin = 1.0

def p_base_func(t_rel):
    return a_poly * t_rel**3 + b_poly * t_rel**2 + c_poly * t_rel + d_poly

def v_base_func(t_rel):
    return 3 * a_poly * t_rel**2 + 2 * b_poly * t_rel + c_poly

def p_ref_func(t_rel):
    t_clipped = np.clip(t_rel, 0, traj_duration)
    base_pos = p_base_func(t_clipped)
    sin_component = A_sin * np.sin(omega_sin * t_clipped) * np.exp(-decay_sin * t_clipped)
    if isinstance(t_clipped, np.ndarray):
        final_pos = np.copy(base_pos)
        active_mask = (t_clipped > 0) & (t_clipped < traj_duration)
        final_pos[active_mask] += sin_component[active_mask]
        zero_time_indices = np.where(t_clipped == 0)[0]
        if len(zero_time_indices) > 0:
            final_pos[zero_time_indices] = d_poly
        return final_pos
    else: # scalar
        if 0 < t_clipped < traj_duration: return base_pos + sin_component
        elif t_clipped == 0: return d_poly
        else: return p_base_func(t_clipped)

def v_ref_func(t_rel):
    t_clipped = np.clip(t_rel, 0, traj_duration)
    base_vel = v_base_func(t_clipped)
    sin_vel_component = A_sin * np.exp(-decay_sin * t_clipped) * \
                        (omega_sin * np.cos(omega_sin * t_clipped) - decay_sin * np.sin(omega_sin * t_clipped))
    if isinstance(t_clipped, np.ndarray):
        vel = np.zeros_like(t_clipped)
        base_mask = (t_clipped >= 0) & (t_clipped <= traj_duration)
        vel[base_mask] = base_vel[base_mask]
        active_mask = (t_clipped > 0) & (t_clipped < traj_duration)
        vel[active_mask] += sin_vel_component[active_mask]
        
        zero_time_indices = np.where(t_clipped == 0)[0]
        if len(zero_time_indices) > 0 and c_poly == 0:
            vel[zero_time_indices] = A_sin * omega_sin 
        
        vel[t_clipped >= traj_duration] = v_base_func(traj_duration) 
        vel[t_clipped < 0] = 0.0
        return vel
    else: # scalar
        if 0 < t_clipped < traj_duration: return base_vel + sin_vel_component
        elif t_clipped == 0: return v_base_func(0) + A_sin * omega_sin
        else: return v_base_func(t_clipped)

ref_time_active = np.arange(0, traj_duration + dt, dt)
ref_pos_active = p_ref_func(ref_time_active)
ref_vel_active = v_ref_func(ref_time_active)

ref_pos = np.zeros_like(time_vector)
ref_vel = np.zeros_like(time_vector)
len_active_ref = len(ref_pos_active)
ref_pos[:min(len_active_ref, n_points)] = ref_pos_active[:min(len_active_ref, n_points)]
ref_vel[:min(len_active_ref, n_points)] = ref_vel_active[:min(len_active_ref, n_points)]
if len_active_ref < n_points:
    ref_pos[len_active_ref:] = ref_pos_active[-1]
    ref_vel[len_active_ref:] = 0.0

# --- Initial Robot State ---
actual_pos_start = -30.0
actual_vel_start = 0.0

# --- Shared Speed Limit Parameter ---
V_TOTAL_MAX = 100.0 # deg/s

# --- Approach 1 (New): No Interpolation (Brutal Jump with Speed Limit) ---
pos_1 = np.zeros(n_points)
vel_1 = np.zeros(n_points)
pos_1[0] = actual_pos_start
for i in range(1, n_points):
    current_ref_target_pos = ref_pos[i]
    delta_pos_desired = current_ref_target_pos - pos_1[i-1]
    max_delta_pos_step = V_TOTAL_MAX * dt # This is 1.0 deg for V_TOTAL_MAX=100, dt=0.01
    actual_delta_pos = np.clip(delta_pos_desired, -max_delta_pos_step, max_delta_pos_step)
    pos_1[i] = pos_1[i-1] + actual_delta_pos
    vel_1[i] = actual_delta_pos / dt

# --- Approach 2 (New): Linear Interpolation then Follow ---
t_interp = 0.5
interp_points = int(t_interp / dt)
pos_2 = np.zeros(n_points)
vel_2 = np.zeros(n_points)
pos_2[0] = actual_pos_start
target_interp_pos = ref_pos_active[0]

for i in range(1, n_points):
    if i < interp_points:
        pos_2[i] = actual_pos_start + (target_interp_pos - actual_pos_start) * (i * dt / t_interp)
        vel_2[i] = (pos_2[i] - pos_2[i-1]) / dt
    else:
        t_ref_relative = (i * dt) - t_interp
        pos_2[i] = p_ref_func(t_ref_relative)
        vel_2[i] = v_ref_func(t_ref_relative)
if interp_points > 0 and n_points > 1 : vel_2[0] = vel_2[1]


# --- Approach 3: "Our Approach" (Proportional Control + Feedforward Velocity with Budgeting) ---
Kp_3 = 5.0
V_controller_max_3 = 50.0
V_feedforward_max_3 = 50.0

pos_3 = np.zeros(n_points)
vel_3 = np.zeros(n_points)
pos_3[0] = actual_pos_start

debug_vel_controller_limited = np.zeros(n_points)
debug_vel_feedforward_limited = np.zeros(n_points)
debug_vel_total_cmd_A3 = np.zeros(n_points)

for i in range(1, n_points):
    current_ref_target_pos = ref_pos[i]
    current_ref_target_vel = ref_vel[i]

    error_pos = current_ref_target_pos - pos_3[i-1]
    vel_cmd_controller = Kp_3 * error_pos
    vel_controller_limited = np.clip(vel_cmd_controller, -V_controller_max_3, V_controller_max_3)
    debug_vel_controller_limited[i] = vel_controller_limited

    vel_cmd_feedforward = current_ref_target_vel
    vel_feedforward_limited = np.clip(vel_cmd_feedforward, -V_feedforward_max_3, V_feedforward_max_3)
    debug_vel_feedforward_limited[i] = vel_feedforward_limited

    vel_total_cmd = vel_controller_limited + vel_feedforward_limited
    debug_vel_total_cmd_A3[i] = vel_total_cmd

    actual_final_velocity = np.clip(vel_total_cmd, -V_TOTAL_MAX, V_TOTAL_MAX)
    vel_3[i] = actual_final_velocity
    pos_3[i] = pos_3[i-1] + vel_3[i] * dt


# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot Positions
ref_label_line1 = rf'$p_{{base}}(t) + {A_sin:.1f} \sin({omega_sin/(2*np.pi):.1f} \cdot 2\pi t) e^{{-{decay_sin:.1f}t}}$'
ref_label_line2 = rf'$p_{{base}}(t) = {a_poly:.2f}t^3 + {b_poly:.1f}t^2 + {d_poly:.1f}$ (for $0 \leq t \leq {traj_duration}s$)' # Removed c_poly*t
axs[0].plot(time_vector, ref_pos, 'k--', label=f'Reference Trajectory:\n{ref_label_line1}\n{ref_label_line2}', linewidth=2)
axs[0].plot(time_vector, pos_1, label=f'Approach 1 (Brutal): Direct Target (Overall $V_{{max}}={V_TOTAL_MAX:.0f}$°/s)')
axs[0].plot(time_vector, pos_2, label=f'Approach 2 (Interp): Linear ($t_{{interp}}={t_interp}s$) then Follow')
label_A3 = f'Approach 3 (P+FF): $K_p={Kp_3}$, Budget $V_{{ctrl}}={V_controller_max_3:.0f}$, $V_{{ff}}={V_feedforward_max_3:.0f}$, Overall $V_{{max}}={V_TOTAL_MAX:.0f}$°/s'
axs[0].plot(time_vector, pos_3, label=label_A3)
axs[0].set_ylabel('Position (degrees)')
axs[0].set_title(f'Trajectory Following (Actual Start: {actual_pos_start}°), Ref Start: {ref_pos_active[0]:.1f}°')
axs[0].legend(loc='upper left', fontsize='small') 
axs[0].grid(True)

text_props = dict(transform=axs[0].transAxes, fontsize=8, va='top', ha='right',
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

# Approach 1 Text
formula_A1_lines = [
    rf'A1 (Brutal - Direct Target):',
    rf'  $\Delta p = \mathrm{{clip}}(p_{{ref}} - p_{{prev}}, \pm {V_TOTAL_MAX * dt:.2f})$',
    rf'  $(V_{{max}} = {V_TOTAL_MAX:.0f}°/s \Rightarrow \Delta p_{{max\_step}} = {V_TOTAL_MAX * dt:.2f}°)$', # Using \Rightarrow
    rf'  $p = p_{{prev}} + \Delta p$'
]
axs[0].text(0.97, 0.97, '\n'.join(formula_A1_lines), **text_props)

# Approach 2 Text
formula_A2_lines = [
    rf'A2 (Linear Interpolation):',
    rf'  For $t < {t_interp}s$ (interpolation):',
    rf'    $p(t) = p_{{start}} + (p_{{ref}}(0) - p_{{start}}) \frac{{t}}{{{t_interp}}}$',
    rf'  For $t \geq {t_interp}s$ (following):',
    rf'    $p(t) = p_{{ref}}(t-{t_interp}s)$'
]
axs[0].text(0.97, 0.72, '\n'.join(formula_A2_lines), **text_props)

# Approach 3 Text
formula_A3_lines = [
    rf'A3 (P Controller + Feedforward Velocity):',
    rf'  $v_{{ctrl\_cmd}} = K_p (p_{{ref}} - p_{{prev}})$ ($K_p={Kp_3}$)',
    rf'  $v_{{ctrl\_lim}} = \mathrm{{clip}}(v_{{ctrl\_cmd}}, \pm {V_controller_max_3:.0f}°/s)$',
    rf'  $v_{{ff\_cmd}} = v_{{ref}}$ (ref. velocity)',
    rf'  $v_{{ff\_lim}} = \mathrm{{clip}}(v_{{ff\_cmd}}, \pm {V_feedforward_max_3:.0f}°/s)$',
    rf'  $v_{{total\_cmd}} = v_{{ctrl\_lim}} + v_{{ff\_lim}}$',
    rf'  $v = \mathrm{{clip}}(v_{{total\_cmd}}, \pm {V_TOTAL_MAX:.0f}°/s)$',
    rf'  $p = p_{{prev}} + v \cdot \Delta t$'
]
axs[0].text(0.97, 0.40, '\n'.join(formula_A3_lines), **text_props)


# Plot Velocities
axs[1].plot(time_vector, ref_vel, 'k--', label='Reference Velocity', linewidth=2)
axs[1].plot(time_vector, vel_1, label='Velocity - Approach 1 (Brutal)')
axs[1].plot(time_vector, vel_2, label='Velocity - Approach 2 (Interp)')
axs[1].plot(time_vector, vel_3, label='Velocity - Approach 3 (P+FF Budgeted)')

axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Velocity (degrees/s)')
axs[1].set_title('Comparison of Trajectory Following Approaches (Velocity)')
axs[1].legend(loc='best', fontsize='small')
axs[1].grid(True)
all_vels = np.concatenate([vel_1, vel_2, vel_3, ref_vel])
vel_min_val, vel_max_val = np.min(all_vels), np.max(all_vels)
padding = max( (vel_max_val - vel_min_val) * 0.1, 10)
axs[1].set_ylim(vel_min_val - padding, vel_max_val + padding)


plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.suptitle('Comparison of Trajectory Initiation Strategies (P+FF with Speed Budgeting)', fontsize=16)
plt.subplots_adjust(top=0.92, right=0.95) 
plt.show()

# Print some values
print(f"Time step dt: {dt} s")
print(f"Reference trajectory 'recorded' start position p_ref(0): {ref_pos_active[0]:.2f} deg")
print(f"Reference trajectory 'recorded' start velocity v_ref(0): {ref_vel_active[0]:.2f} deg/s")
print(f"Robot actual start position: {actual_pos_start:.2f} deg")

print(f"\nMax absolute velocities encountered:")
print(f"  Reference: {np.max(np.abs(ref_vel)):.2f} deg/s")
print(f"  Approach 1 (Brutal): {np.max(np.abs(vel_1)):.2f} deg/s (Overall Limit: {V_TOTAL_MAX:.0f} deg/s)")
print(f"    Max pos step for A1: {V_TOTAL_MAX * dt:.2f} deg (from V_max={V_TOTAL_MAX:.0f} deg/s * dt={dt} s)")
print(f"  Approach 2 (Interp): {np.max(np.abs(vel_2)):.2f} deg/s")
print(f"  Approach 3 (P+FF Budgeted): {np.max(np.abs(vel_3)):.2f} deg/s (Overall Limit: {V_TOTAL_MAX:.0f} deg/s)")
print(f"    Max controller component (limited): {np.max(np.abs(debug_vel_controller_limited)):.2f} deg/s (Budget: {V_controller_max_3:.0f} deg/s)")
print(f"    Max feedforward component (limited): {np.max(np.abs(debug_vel_feedforward_limited)):.2f} deg/s (Budget: {V_feedforward_max_3:.0f} deg/s)")
print(f"    Max A3 total cmd before overall cap: {np.max(np.abs(debug_vel_total_cmd_A3)):.2f} deg/s")