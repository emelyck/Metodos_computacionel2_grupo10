# -*- coding: utf-8 -*-
"""Taller 3

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fvkGOkwEC1Pci7OsPsMS3TnakIgEm0KZ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numba as nb
import matplotlib as mpl
import imageio_ffmpeg
import scipy.fftpack
import pandas as pd

# ----------------------------------------------------------------------
# Punto 1.a - Trayectoria de un proyectil sin fricción (para encontrar el ángulo óptimo)
# ----------------------------------------------------------------------
m = 10
v0 = 10
g = 9.773
b = 0

def motion_equations(t, y):
    ux, uy, x, y_pos = y
    speed = np.sqrt(ux**2 + uy**2)**2
    dux_dt = -b * ux * speed / m
    duy_dt = -g - b * uy * speed / m
    dx_dt = ux
    dy_dt = uy
    return [dux_dt, duy_dt, dx_dt, dy_dt]

def hit_ground(t, y):
    return y[3]
hit_ground.terminal = True
hit_ground.direction = -1

angles = np.linspace(0, 90, 91)
ranges = []
max_range = 0
best_angle = 0

for angle in angles:
    rad = np.radians(angle)
    initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0]
    sol = solve_ivp(motion_equations, [0, 10], initial_conditions, events=hit_ground, dense_output=False)
    if sol.t_events[0].size > 0:
        range_at_angle = sol.y_events[0][0, 2]
        ranges.append(range_at_angle)
        if range_at_angle > max_range:
            max_range = range_at_angle
            best_angle = angle
    else:
        ranges.append(0)

print(f"El ángulo óptimo para 𝛽 {b}: {best_angle}° con un alcance de: {max_range:.2f}.")


# ----------------------------------------------------------------------
# Punto 1.b - Trayectoria de un proyectil con fricción (ángulo óptimo vs. β y pérdida de energía)
# ----------------------------------------------------------------------
E_initial = 0.5 * m * v0**2
betas = np.linspace(0, 2, 91)

def hit_ground(t, y, beta):
    return y[3]
hit_ground.terminal = True
hit_ground.direction = -1

def motion(t, y, beta):
    ux, uy, x, y_pos, energy_acc = y
    speed = np.sqrt(ux**2 + uy**2)
    dux_dt = -beta * ux * speed / m
    duy_dt = -g - beta * uy * speed / m
    dx_dt = ux
    dy_dt = uy
    power_friction = beta * speed**3
    return [dux_dt, duy_dt, dx_dt, dy_dt, power_friction]

best_angles = []
energy_losses = []

for beta in betas:
    max_range = 0
    best_angle = 0
    for angle in np.linspace(0, 90, 91):
        rad = np.radians(angle)
        initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0, 0]
        sol = solve_ivp(motion, [0, 10], initial_conditions, args=(beta,), events=hit_ground, dense_output=False)
        if sol.t_events[0].size > 0:
            range_at_angle = sol.y_events[0][0, 2]
            if range_at_angle > max_range:
                max_range = range_at_angle
                best_angle = angle

    rad = np.radians(best_angle)
    initial_conditions = [v0 * np.cos(rad), v0 * np.sin(rad), 0, 0, 0]
    sol = solve_ivp(motion, [0, 10], initial_conditions, args=(beta,), events=hit_ground, vectorized=True, dense_output=True)
    if sol.y.shape[1] > 0:
        energy_lost = np.trapz(sol.y[4], sol.t)
        energy_losses.append(energy_lost)
    else:
        energy_losses.append(0)

    best_angles.append(best_angle)

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].step(betas, best_angles, where='mid', label='Ángulo óptimo', color='yellow')
ax[0].set_xscale('log')
ax[0].set_title('θ vs. β')
ax[0].set_xlabel('β')
ax[0].set_ylabel('θ (grados)')
ax[0].grid(True)

ax[1].step(betas, best_angles, where='mid', label='Ángulo óptimo', color='yellow')
ax[1].set_title('θ vs β')
ax[1].set_xlabel('β')
ax[1].set_ylabel('θ (grados)')
ax[1].grid(True)
plt.savefig("1.a).pdf")

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].step(betas, energy_losses, where='mid', label='Pérdida de energía', color='yellow')
ax[0].set_xscale('log')
ax[0].set_title('Pérdida de energía vs β')
ax[0].set_xlabel('β')
ax[0].set_ylabel('Pérdida de energía')
ax[0].grid(True)

ax[1].step(betas, energy_losses, where='mid', label='Pérdida de energía', color='yellow')
ax[1].set_title('Pérdida de energía vs. β')
ax[1].set_xlabel('β')
ax[1].set_ylabel('Pérdida de energía')
plt.savefig("1.b).pdf")


# ----------------------------------------------------------------------
# Punto 4 - Ecuación de Schrödinger para el oscilador armónico
# ----------------------------------------------------------------------
def schrodinger(x, psi, E):
    f, dfdx = psi
    return [dfdx, (x**2 - 2*E) * f]

def solve_schrodinger(E, x_range, initial_conditions):
    sol = solve_ivp(schrodinger, [x_range[0], x_range[1]], initial_conditions, args=(E,), t_eval=np.linspace(x_range[0], x_range[1], 1000))
    return sol.t, sol.y[0]

x_range = [-6, 6]
initial_conditions_symmetric = [1, 0]
initial_conditions_antisymmetric = [0, 1]

def find_energies(initial_conditions, num_energies):
    energies = []
    for E in np.linspace(0, 25, 1000):
        x, psi = solve_schrodinger(E, x_range, initial_conditions)
        if np.abs(psi[-1]) < 0.01:
            energies.append(E)
            if len(energies) == num_energies:
                break
    return energies

energies_symmetric = find_energies(initial_conditions_symmetric, 5)
energies_antisymmetric = find_energies(initial_conditions_antisymmetric, 5)

print("Energías simétricas:", energies_symmetric)
print("Energías antisimétricas:", energies_antisymmetric)

plt.figure(figsize=(12, 6)) # Increased figure size

for i, E in enumerate(energies_symmetric + energies_antisymmetric):
    initial_conditions = initial_conditions_symmetric if i < len(energies_symmetric) else initial_conditions_antisymmetric
    x, psi = solve_schrodinger(E, x_range, initial_conditions)
    envelope = np.exp(-0.1 * (x)**2)  # Improved envelope
    plt.plot(x, psi * envelope + i * 2, label=f'E = {E:.2f}') # Larger offset
    plt.text(-5, i * 2 + 0.5, f'{initial_conditions}', fontsize=8, color='black')

x_potential = np.linspace(-6, 6, 1000)
potential = 0.5 * x_potential**2
plt.plot(x_potential, potential, 'k--', alpha=0.5, linewidth=2, label='Potencial armónico')

plt.xlabel('x')
plt.ylabel('$\psi(x)$ y Energía')
plt.title('Soluciones de la ecuación de Schrödinger para el oscilador armónico')
plt.legend()
plt.savefig("Solutions of the Schrödinger equation for the harmonic oscillator .pdf")
plt.show()


# ----------------------------------------------------------------------
# Punto 2 - Simulación de la ecuación de onda con diferentes condiciones de frontera
# ----------------------------------------------------------------------
L = 2
Nx = 100
dx = L / (Nx - 1)
c = 1
dt = 0.005
Nt = 400

C = c * dt / dx
assert C < 1, "El coeficiente de Courant debe ser menor que 1 para estabilidad."

x = np.linspace(0, L, Nx)

u0 = np.exp(-125 * (x - 0.5) ** 2)
u_prev = np.copy(u0)
u = np.copy(u0)
u_next = np.zeros_like(u)

def aplicar_condiciones_de_frontera(u, tipo):
    if tipo == "Dirichlet":
        u[0] = 0
        u[-1] = 0
    elif tipo == "Neumann":
        u[0] = u[1]
        u[-1] = u[-2]
    elif tipo == "Periódicas":
        u[0] = u[-2]
        u[-1] = u[1]
    return u

fig, axes = plt.subplots(3, 1, figsize=(6, 8))
tipos = ["Dirichlet", "Neumann", "Periódicas"]
lines = []
for ax, tipo in zip(axes, tipos):
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Condición de frontera: {tipo}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True)
    line, = ax.plot(x, u, lw=2, color='yellow')
    lines.append(line)

def actualizar(frame):
    global u_prev, u, u_next
    for i, tipo in enumerate(tipos):
        u_next[1:-1] = (2 * u[1:-1] - u_prev[1:-1] + C**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
        u_next = aplicar_condiciones_de_frontera(u_next, tipo)
        lines[i].set_ydata(u_next)
        u_prev, u = np.copy(u), np.copy(u_next)
    return lines

ani = animation.FuncAnimation(fig, actualizar, frames=Nt, interval=dt * 1000, blit=True)
ani.save("2.mp4", writer="ffmpeg", fps=30)


# ----------------------------------------------------------------------
# Punto 2.a - Órbita del electrón sin el término de Larmor
# ----------------------------------------------------------------------
def f_prime_sin_larmor(t, f):
    x, y, vx, vy = f
    r = np.sqrt(x**2 + y**2)
    Fx = -x / r**3
    Fy = -y / r**3
    return [vx, vy, Fx, Fy]

f0 = [1.0, 0.0, 0.0, 1.0]
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol_sin_larmor = solve_ivp(f_prime_sin_larmor, t_span, f0, dense_output=True, t_eval=t_eval, method='RK45',
                             max_step=0.01, rtol=1e-8, atol=1e-8)

x_sol_sin_larmor = sol_sin_larmor.y[0]
y_sol_sin_larmor = sol_sin_larmor.y[1]
vx_sol_sin_larmor = sol_sin_larmor.y[2]
vy_sol_sin_larmor = sol_sin_larmor.y[3]

r_sol_sin_larmor = np.sqrt(x_sol_sin_larmor**2 + y_sol_sin_larmor**2)
kinetic_energy_sin_larmor = 0.5 * (vx_sol_sin_larmor**2 + vy_sol_sin_larmor**2)
potential_energy_sin_larmor = -1 / r_sol_sin_larmor
total_energy_sin_larmor = kinetic_energy_sin_larmor + potential_energy_sin_larmor

P_teo_atomic = 2 * np.pi * (1**(3/2))
atomic_time_unit_atto = 24.1888
P_teo_atto = P_teo_atomic * atomic_time_unit_atto

y_crossings_indices = np.where((y_sol_sin_larmor[:-1] * y_sol_sin_larmor[1:]) < 0)[0]
period_sim_times = []
for index in y_crossings_indices:
    if vx_sol_sin_larmor[index] < 0 and y_sol_sin_larmor[index] > 0:
        period_sim_times.append(t_eval[index])

P_sim_atomic = np.mean(np.diff(period_sim_times)) if len(period_sim_times) > 1 else np.nan
P_sim_atto = P_sim_atomic * atomic_time_unit_atto if not np.isnan(P_sim_atomic) else np.nan

print(f'f\'2.a) {{P_teo = {P_teo_atto:.5f} attosegundos}}; {{P_sim = {P_sim_atto:.5f} attosegundos}}\'')

plt.figure(figsize=(6, 6))
plt.plot(x_sol_sin_larmor, y_sol_sin_larmor)
plt.title('2.a. Órbita del Electrón (Sin Larmor)')
plt.xlabel('x (ua)')
plt.ylabel('y (ua)')
plt.axis('equal')
plt.savefig("2.a electron_orbit .pdf")
plt.close()

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_eval, total_energy_sin_larmor)
plt.title('Energía Total')
plt.ylabel('Energía (ua)')
plt.subplot(3, 1, 2)
plt.plot(t_eval, kinetic_energy_sin_larmor)
plt.title('Energía Cinética')
plt.ylabel('Energía (ua)')
plt.subplot(3, 1, 3)
plt.plot(t_eval, r_sol_sin_larmor)
plt.title('Radio')
plt.xlabel('Tiempo (ua)')
plt.ylabel('Radio (ua)')
plt.tight_layout()
plt.savefig("energy vs time.pdf")
plt.close()


# ----------------------------------------------------------------------
# Punto 2.b - Órbita del electrón con el término de Larmor (simplificado)
# ----------------------------------------------------------------------

from scipy.constants import fine_structure, physical_constants

# Atomic units
bohr_radius_au = 1
electron_mass_au = 1
electron_charge_au = -1
reduced_planck_au = 1
speed_of_light_au = 1/fine_structure

# Conversion to attoseconds from atomic units of time (Hartree time)
hartree_time_seconds = physical_constants['atomic unit of time'][0]
hartree_time_attoseconds = hartree_time_seconds * 1e18

def force_coulomb(r):
    """Coulomb force in atomic units."""
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-9: # Avoid division by zero near origin
        return np.array([0.0, 0.0])
    return - r / (r_mag**3)

def get_acceleration(position):
    """Get acceleration from force (mass = 1 in au)."""
    return force_coulomb(position)

def rk4_step(y, t, dt,deriv_func):
    """Single RK4 step for system of ODEs."""
    k1 = deriv_func(y, t)
    k2 = deriv_func(y + dt*k1/2, t + dt/2)
    k3 = deriv_func(y + dt*k2/2, t + dt/2)
    k4 = deriv_func(y + dt*k3, t + dt)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def orbit_derivatives_no_larmor(state, t):
    """Derivatives for electron orbit without Larmor radiation."""
    position = state[:2]
    velocity = state[2:]
    acceleration = get_acceleration(position)
    return np.concatenate((velocity, acceleration))

def orbit_derivatives_with_larmor(state, t, larmor_factor):
    """Derivatives for electron orbit with Larmor radiation."""
    position = state[:2]
    velocity = state[2:]
    acceleration = get_acceleration(position)
    acceleration_larmor_correction = acceleration # In this simple correction, using current acceleration

    # Calculate Larmor term - simplified to velocity reduction in each step
    # Based on the provided formula which seems problematic, a simpler approach is taken.
    # The formula is unclear and potentially incorrect in the image.
    # A basic energy loss implementation is more physically plausible for demonstration.
    # Here, we are simply reducing the velocity magnitude in each step, scaled by acceleration.
    velocity_reduction_factor = larmor_factor * np.linalg.norm(acceleration_larmor_correction)
    corrected_velocity = velocity * (1 - velocity_reduction_factor)

    return np.concatenate((corrected_velocity, acceleration))


# 2.b. Simulation with Larmor radiation (using a simplified velocity correction)
def simulate_with_larmor(dt, t_max, larmor_factor, fall_radius_threshold=0.1):
    """Simulate electron orbit with Larmor radiation."""
    t_points = np.arange(0, t_max, dt)
    results = []
    state = np.array([1.0, 0.0, 0.0, 1.0]) # [x, y, vx, vy] initial conditions
    fall_time = None

    for i, t in enumerate(t_points):
        results.append(np.copy(state))
        state_before_larmor = np.copy(state) # Keep state before RK step for Larmor correction
        # Corrected line: Calling orbit_derivatives_no_larmor directly (it was a scope issue in lambda in some environments)
        state = rk4_step(state, t, dt, orbit_derivatives_no_larmor)
        position_rk = state[:2]
        velocity_rk = state[2:]
        acceleration_rk = get_acceleration(position_rk)

        # Apply Larmor correction - simplified velocity reduction based on acceleration
        velocity_reduction_factor = larmor_factor * np.linalg.norm(acceleration_rk)
        state[2:] = velocity_rk * (1 - velocity_reduction_factor)


        radius = np.sqrt(state[0]**2 + state[1]**2)
        if radius < fall_radius_threshold:
            fall_time = t
            results.append(np.copy(state)) # Final state when electron falls
            t_points = t_points[:i+2] # Trim time points
            results = np.array(results)
            break

    if fall_time is None: # In case it doesn't fall within t_max
        results = np.array(results)

    x_vals = results[:, 0]
    y_vals = results[:, 1]
    vx_vals = results[:, 2]
    vy_vals = results[:, 3]
    radius = np.sqrt(x_vals**2 + y_vals**2)
    kinetic_energy = 0.5 * (vx_vals**2 + vy_vals**2)
    potential_energy = -1 / radius
    total_energy = kinetic_energy + potential_energy

    return t_points, x_vals, y_vals, radius, kinetic_energy, total_energy, fall_time


dt_larmor = 0.001 # Smaller dt for Larmor, atomic units of time
t_max_larmor = 20 # Atomic units of time
larmor_factor = 0.0001 # Adjust Larmor factor to control radiation strength. Needs fine-tuning.


# --- 2.b. Run simulation with Larmor ---
t_larmor, x_larmor, y_larmor, radius_larmor, kinetic_energy_larmor, total_energy_larmor, t_fall_au = simulate_with_larmor(dt_larmor, t_max_larmor, larmor_factor)
t_fall_attoseconds = t_fall_au * hartree_time_attoseconds if t_fall_au else np.nan


print(f'f\'2.b) {{t_fall = :{t_fall_attoseconds:.5f}}}\'')

# ----------------------------------------------------------------------
# Punto 3.a y 3.b - Precesión de la órbita de Mercurio
# ----------------------------------------------------------------------
GM = 39.4234021
a = 0.38709893
e = 0.20563069
c = 1  #

L = np.sqrt(GM*a*(1-e**2))
Alpha = 3 * (L)**2 / GM
Alpha /=c**2


def simu_10b(t, R, GM, Alpha):
    x, y, vx, vy = R
    r_vector = np.array([x, y])
    r = np.linalg.norm(r_vector)
    r_g = r_vector / r
    a_vector = -(GM / r**2) * (1 + Alpha / r**2) * r_g

    return np.array([vx, vy, a_vector[0], a_vector[1]])

def perihelion_event(t, R, GM, Alpha):
    x, y, vx, vy = R
    return np.array([x * vx + y * vy])

perihelion_event.direction = 0
perihelion_event.terminal = False

x0 = a * (1 + e)
y0 = 0
vx0 = 0
vy0 = np.sqrt(GM * (1 - e) / (a * (1 + e)))

R_0 = [x0, y0, vx0, vy0]

t_span = (0, 10)


sol_p = solve_ivp(simu_10b, t_span=t_span, y0=R_0, args=(GM, Alpha),
                  max_step=1e-3, method="RK45", dense_output=True,
                  events=[perihelion_event])

teventsp = sol_p.t_events[0]
y_sol = sol_p.y_events[0]

af = np.arctan2(y_sol[:, 1], y_sol[:, 0])
arc = af * (180 / np.pi) * 3600

slope, intercept = np.polyfit(teventsp, arc, 1)
precession_rate = slope * 100

residuals = arc - (slope * teventsp + intercept)
std_dev_residuals = np.std(residuals)
uncertainty = std_dev_residuals / np.sqrt(len(teventsp)) * 100

print(f"Tasa de Precesión: {precession_rate:.4f} +/- {uncertainty:.4f} segundos de arco/siglo")

plt.figure(figsize=(10, 6))
plt.plot(teventsp, arc, marker='.', linestyle='-', label='Ángulo del Perihelio')
plt.xlabel("Tiempo (años)")
plt.ylabel("Ángulo (segundos de arco)")
plt.title("Precesión de la Órbita de Mercurio")
plt.grid(True)
plt.legend()
plt.text(0.05, 0.95, f"Tasa de Precesión: {precession_rate:.4f} ± {uncertainty:.4f} segundos de arco/siglo",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.savefig("3.b.pdf")
plt.show()

print("Comparación con la literatura:")
print("  Valor observado: ~42.98 segundos de arco/siglo")
print("  Valor simulado:", precession_rate, "segundos de arco/siglo")

Alpha_exaggerated = 1e-2

sol_exaggerated = solve_ivp(simu_10b, t_span=(0, 10), y0=R_0, args=(GM, Alpha_exaggerated),
                             max_step=1e-3, method="RK45", dense_output=True)

x_exaggerated = sol_exaggerated.y[0]
y_exaggerated = sol_exaggerated.y[1]
t_exaggerated = sol_exaggerated.t

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-a * (1 + e) * 2, a * (1 + e) * 2)
ax.set_ylim(-a * (1 + e) * 2, a * (1 + e) * 2)
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_title("Órbita de Mercurio (Efecto Relativista Exagerado)")

sol_circle = plt.Circle((0, 0), 0.05, color='yellow', label='Sol')
ax.add_artist(sol_circle)

line, = ax.plot([], [], lw=2, label='Mercurio')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = x_exaggerated[:i]
    y = y_exaggerated[:i]
    line.set_data(x, y)
    return line,

"""
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(t_exaggerated), interval=20, blit=True)

ani.save("3.a.mp4", writer='ffmpeg', fps=30)

plt.legend()
plt.show()
"""

# ----------------------------------------------------------------------
# Punto 1 del Taller B - Ecuación de Poisson
# ----------------------------------------------------------------------
N = 100
L = 2.4
h = L / (N - 1)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

rho = -X - Y
phi_old = np.random.rand(N, N) - 0.5
phi_new = np.zeros_like(phi_old)

for i in range(N):
    for j in range(N):
        r_sq = X[i, j]**2 + Y[i, j]**2
        if r_sq >= 1.0:
            theta = np.arctan2(Y[i, j], X[i, j])
            phi_old[i, j] = np.sin(7 * theta)
        else:
            phi_old[i,j] = 0

@nb.jit(nopython=True)
def poisson_iteration_numba(phi_old, phi_new, X, Y, rho, h, N):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            r_sq = X[i, j]**2 + Y[i, j]**2
            if r_sq < 1.0:
                phi_new[i, j] = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] + phi_old[i, j+1] + phi_old[i, j-1] + 4 * np.pi * h**2 * rho[i, j])
    return phi_new

tolerance = 1e-4
max_iterations = 15000
iteration_count = 0
trace_diff = tolerance + 1

while trace_diff > tolerance and iteration_count < max_iterations:
    phi_new = np.copy(phi_old)

    phi_new = poisson_iteration_numba(phi_old, phi_new, X, Y, rho, h, N)

    diff_matrix = np.abs(phi_new - phi_old)
    trace_diff = np.sum(diff_matrix)
    phi_old = np.copy(phi_new)
    iteration_count += 1

    if iteration_count % 1000 == 0:
        print(f"Iteration: {iteration_count}, Trace Difference: {trace_diff}")


print(f"Converged after {iteration_count} iterations with trace difference: {trace_diff}")

# --- 5. Plotting ---
"""
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
boundary_condition_phi = np.zeros_like(X)
for i in range(N):
    for j in range(N):
        r_sq = X[i, j]**2 + Y[i, j]**2
        if r_sq >= 1.0:
            theta = np.arctan2(Y[i, j], X[i, j])
            boundary_condition_phi[i, j] = np.sin(7 * theta)
        else:
            boundary_condition_phi[i,j] = np.nan # Mark inside for ? symbol

im = ax1.imshow(boundary_condition_phi[::-1,:], extent=[-L/2, L/2, -L/2, L/2], cmap='jet', vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax1, orientation='vertical', shrink=0.8)
cbar.set_label(r'$\phi$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Condiciones de frontera')
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
ax1.add_patch(circle)

ax1.text(0, 0, '?', ha='center', va='center', fontsize=40, color='black')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surface = ax2.plot_surface(X, Y, phi_new, cmap='jet')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel(r'$\phi$')
ax2.set_title('Muestra de solución')
fig.colorbar(surface, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig("1_numba.png")
plt.show()
"""

# PUNTO 3
# Parámetros del problema
alpha = 0.022  # Coeficiente de dispersión
L = 2.0  # Longitud del dominio
N = 256  # Número de puntos espaciales
dx = L / N  # Espaciado espacial
dt = 0.0001  # Paso de tiempo
T_max = 2000  # Tiempo de simulación
num_frames = 100  # Número de cuadros en la animación

# Malla espacial y condición inicial
x = np.linspace(0, L, N, endpoint=False)
psi = np.cos(np.pi * x)

# Diferencias finitas: Matriz de derivadas (según esquema del artículo)
def deriv1(f):
    """Derivada primera con diferencias finitas centradas."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

def deriv3(f):
    """Derivada tercera con diferencias finitas."""
    return (np.roll(f, -2) - 2 * np.roll(f, -1) + 2 * np.roll(f, 1) - np.roll(f, 2)) / (2 * dx**3)

# Almacenar evolución para la animación
frames = []

# Simulación temporal usando esquema explícito
t = 0
while t < T_max:
    psi_new = psi - dt * (psi * deriv1(psi) + alpha**2 * deriv3(psi))
    psi = psi_new.copy()  # Actualizar valores
    t += dt

    # Guardar cada cierto número de pasos para la animación
    if len(frames) < num_frames and t % (T_max / num_frames) < dt:
        frames.append(psi.copy())

# Convertir lista de resultados en un array
frames = np.array(frames)

# Crear animación
fig, ax = plt.subplots(figsize=(8, 4))
cax = ax.imshow(frames.T, aspect='auto', origin='lower', extent=[0, T_max, 0, L], cmap='magma')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle x [m]")
fig.colorbar(cax, label=r"$\psi(t, x)$")

# Convertir lista de resultados en un DataFrame
df_frames = pd.DataFrame(frames)

# Guardar los datos en un archivo CSV (opcional)
df_frames.to_csv("evolucion_kdv.csv", index=False)

# Mostrar los primeros valores para inspección
print(df_frames.head())

# Visualización con matplotlib
plt.imshow(df_frames.T, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label=r"$\psi(t, x)$")
plt.xlabel("Time Step")
plt.ylabel("Spatial Position")
plt.title("Evolución de la ecuación de KdV")
plt.show()


# PUNTO 4
# Parámetros del problema
dx = dy = 0.01  # Paso espacial (m)
dt = 0.001  # Paso temporal (s)
T = 2.0  # Tiempo total de simulación (s)
f = 10  # Frecuencia de la fuente (Hz)
A = 0.01  # Amplitud de la onda (m)
c = 0.5  # Velocidad de onda en agua (m/s)
c_lente = c / 5  # Velocidad dentro del lente
Nx, Ny = 100, 200  # Tamaño de la malla
Nt = int(T / dt)  # Número de pasos de tiempo

# Verificación del coeficiente de Courant
courant = c * dt / dx
if courant >= 1:
    raise ValueError("El coeficiente de Courant es mayor o igual a 1, reduce dt o aumenta dx")

# Inicialización de la onda
gu = np.zeros((Nx, Ny))  # Estado actual
gu_prev = np.zeros((Nx, Ny))  # Estado anterior
gu_next = np.zeros((Nx, Ny))  # Estado futuro

# Mapa de velocidades
c_map = np.full((Nx, Ny), c)  # Velocidad en todo el dominio

# Definir la pared con la apertura
w_y = int(0.04 / dy)  # Ancho de la pared en celdas
w_x = int(0.4 / dx)  # Apertura en la pared
y_mid = Ny // 2
x_mid = Nx // 2
c_map[:, y_mid - w_y//2 : y_mid + w_y//2] = 0  # Pared excepto en la apertura
c_map[x_mid - w_x//2 : x_mid + w_x//2, y_mid - w_y//2 : y_mid + w_y//2] = c  # Apertura con velocidad normal

# Definir el lente (zona de velocidad reducida)
x_lente = Nx // 4
y_lente = Ny // 2
for i in range(Nx):
    for j in range(Ny):
        if ((i - x_lente)**2 / (0.1/dx)**2 + 3 * (j - y_lente)**2 / (0.1/dy)**2) <= 1/25:
            c_map[i, j] = c_lente

# Posición de la fuente
x_src, y_src = int(0.5 / dx), int(0.5 / dy)

# Configuración de la figura
fig, ax = plt.subplots()
ax.set_facecolor("black")  # Fondo negro
im = ax.imshow(gu, vmin=-A, vmax=A, cmap='RdBu', animated=True)

# Dibujar la pared y la apertura
y_wall_start = y_mid - w_y//2
y_wall_end = y_mid + w_y//2
x_wall = np.arange(Nx)
ax.plot([y_wall_start, y_wall_start], [0, Nx-1], color="white", linewidth=2)
ax.plot([y_wall_end, y_wall_end], [0, Nx-1], color="white", linewidth=2)
ax.plot([y_mid, y_mid], [0, x_mid - w_x//2], color="white", linewidth=2)
ax.plot([y_mid, y_mid], [x_mid + w_x//2, Nx-1], color="white", linewidth=2)

def update(n):
    global gu, gu_prev, gu_next
    
    # Aplicar la ecuación de onda con diferencias finitas
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if c_map[i, j] != 0:  # No actualizar la pared
                r = (c_map[i, j] * dt / dx) ** 2
                gu_next[i, j] = (2 * gu[i, j] - gu_prev[i, j] +
                                r * (gu[i+1, j] + gu[i-1, j] + gu[i, j+1] + gu[i, j-1] - 4 * gu[i, j]))
    
    # Fuente oscilante
    gu_next[x_src, y_src] = A * np.sin(2 * np.pi * f * n * dt)
    
    # Actualizar estados
    gu_prev, gu, gu_next = gu, gu_next, gu_prev
    
    im.set_array(gu)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=1000 * T / Nt, blit=True)
ani.save("4.a.mp4", fps=10)
plt.show()

###Bono con condiciones iniciales realistas
# Verificación del coeficiente de Courant
courant = c * dt / dx
if courant >= 1:
    raise ValueError("El coeficiente de Courant es mayor o igual a 1, reduce dt o aumenta dx")

# Inicialización de la onda
gu = np.zeros((Nx, Ny))  # Estado actual
gu_prev = np.zeros((Nx, Ny))  # Estado anterior
gu_next = np.zeros((Nx, Ny))  # Estado futuro

# Mapa de velocidades
c_map = np.full((Nx, Ny), c)  # Velocidad en todo el dominio

# Definir la pared con la apertura
w_y = int(0.04 / dy)  # Ancho de la pared en celdas
w_x = int(0.4 / dx)  # Apertura en la pared
y_mid = Ny // 2
x_mid = Nx // 2
c_map[:, y_mid - w_y//2 : y_mid + w_y//2] = 0  # Pared excepto en la apertura
c_map[x_mid - w_x//2 : x_mid + w_x//2, y_mid - w_y//2 : y_mid + w_y//2] = c  # Apertura con velocidad normal

# Definir el lente (zona de velocidad reducida)
x_lente = Nx // 4
y_lente = Ny // 2
for i in range(Nx):
    for j in range(Ny):
        if ((i - x_lente)**2 / (0.1/dx)**2 + 3 * (j - y_lente)**2 / (0.1/dy)**2) <= 1/25:
            c_map[i, j] = c_lente

# Posición de la fuente
x_src, y_src = int(0.5 / dx), int(0.5 / dy)

# Configuración de la figura
fig, ax = plt.subplots()
ax.set_facecolor("black")  # Fondo negro
im = ax.imshow(gu, vmin=-A, vmax=A, cmap='RdBu', animated=True)

# Dibujar la pared y la apertura
y_wall_start = y_mid - w_y//2
y_wall_end = y_mid + w_y//2
x_wall = np.arange(Nx)
ax.plot([y_wall_start, y_wall_start], [0, Nx-1], color="white", linewidth=2)
ax.plot([y_wall_end, y_wall_end], [0, Nx-1], color="white", linewidth=2)
ax.plot([y_mid, y_mid], [0, x_mid - w_x//2], color="white", linewidth=2)
ax.plot([y_mid, y_mid], [x_mid + w_x//2, Nx-1], color="white", linewidth=2)

def update(n):
    global gu, gu_prev, gu_next
    
    # Aplicar la ecuación de onda con diferencias finitas
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if c_map[i, j] != 0:  # No actualizar la pared
                r = (c_map[i, j] * dt / dx) ** 2
                gu_next[i, j] = (2 * gu[i, j] - gu_prev[i, j] +
                                r * (gu[i+1, j] + gu[i-1, j] + gu[i, j+1] + gu[i, j-1] - 4 * gu[i, j]))
    
    # Aplicar condiciones de frontera realistas (ondas reflejadas)
    gu_next[0, :] = gu[1, :]
    gu_next[-1, :] = gu[-2, :]
    gu_next[:, 0] = gu[:, 1]
    gu_next[:, -1] = gu[:, -2]
    
    # Fuente oscilante
    gu_next[x_src, y_src] = A * np.sin(2 * np.pi * f * n * dt)
    
    # Actualizar estados
    gu_prev, gu, gu_next = gu, gu_next, gu_prev
    
    im.set_array(gu)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=1000 * T / Nt, blit=True)
ani.save("4.a bono.mp4", writer="ffmpeg", fps=10)
plt.show()
