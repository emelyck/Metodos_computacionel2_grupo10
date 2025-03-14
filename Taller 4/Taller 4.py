import numpy as np
import emcee
import matplotlib.pyplot as plt


########################
####### EJERCICIO 1
########################

# Función log-probabilidad para emcee: se extrae el parámetro escalar
def log_prob(theta, n=10, alpha=4/5):
    x = theta[0]
    k = np.arange(1, n+1)
    suma = np.sum(np.exp( - (x - k)**2 * k ) / (k**alpha))
    return np.log(suma)

# Parámetros de la simulación
N_total = 500000
N_caminantes = 20
N_pasos = int(np.ceil(N_total / N_caminantes))
dim = 1

# Posición inicial de los walkers
posicion_0 = np.random.randn(N_caminantes, dim)

# Configuración del sampler
sampler = emcee.EnsembleSampler(N_caminantes, dim, log_prob)
sampler.run_mcmc(posicion_0, N_pasos, progress=True)

# Extraer y aplanar las muestras a un array 1D
samples = sampler.get_chain(flat=True)[:, 0]

""""
plt.figure(figsize=(8,6))
plt.hist(samples, bins=200, density=True, alpha=0.6)
plt.xlabel("x")
plt.ylabel("Densidad de probabilidad")
plt.title("Histograma del muestreo")
plt.savefig("1.a.pdf")
plt.show()
"""

# Función g(x) definida vectorialmente
def g(x, n=10, alpha=4/5):
    x = np.atleast_1d(x)[:, None]
    k = np.arange(1, n+1)
    return np.sum(np.exp( - (x - k)**2 * k ) / (k**alpha), axis=1)

# Función f(x)
def f(x):
    return np.exp(-x**2)

# --- Cálculo de A y su incertidumbre ---
N = len(samples)
g_vals = g(samples)
f_vals = f(samples)

# ratio_i = f(x_i) / g(x_i)
ratio = f_vals / g_vals

# Promedio Q = (1/N) * sum(ratio)
Q = np.mean(ratio)

# Estimación de A
A_est = np.sqrt(np.pi) / Q

# Varianza de ratio (usamos ddof=1 para una estimación insesgada)
var_ratio = np.var(ratio, ddof=1)

# Desviación estándar de Q
sigma_Q = np.sqrt(var_ratio / N)

# Propagación de errores para A = sqrt(pi)/Q
sigma_A = (np.sqrt(np.pi) / Q**2) * sigma_Q

print(f"1.b) A = {A_est:.8f} ± {sigma_A:.8f}")

########################
####### EJERCICIO 2
########################

# Parámetros físicos (unidades en cm)
D1 = 50.0   # Distancia de la fuente a las rendijas
D2 = 50.0   # Distancia de las rendijas a la pantalla
lambda_ = 6.7e-5  # Longitud de onda: 670 nm = 6.7e-5 cm
A = 0.04    # Ancho de la apertura de la fuente (0.4 mm = 0.04 cm)
a = 0.01    # Ancho de cada rendija (0.1 mm = 0.01 cm)
d = 0.1     # Separación entre rendijas (0.1 cm)

# Definición del dominio para la fuente (variable s) y para las rendijas (variable r)
s_min, s_max = -A/2, A/2

# Rangos para las rendijas: se definen dos intervalos
rendija_izq = (-d/2 - a/2, -d/2 + a/2)
rendija_der = ( d/2 - a/2,  d/2 + a/2)

# Función log_probabilidad para emcee: se evalúa en (s, r)
def log_prob_fresnel(theta):
    s, r = theta
    if (s < s_min) or (s > s_max):
        return -np.inf
    if not ((rendija_izq[0] <= r <= rendija_izq[1]) or (rendija_der[0] <= r <= rendija_der[1])):
        return -np.inf
    return 0.0

# Configuración del muestreo MCMC para la integral de camino
n_caminantes = 50
n_steps = 2000  # 50 * 2000 = 100,000 muestras

# Inicialización de los caminantes:
posiciones_iniciales = []
for i in range(n_caminantes):
    s0 = np.random.uniform(s_min, s_max)
    if i < n_caminantes/2:
        r0 = np.random.uniform(rendija_izq[0], rendija_izq[1])
    else:
        r0 = np.random.uniform(rendija_der[0], rendija_der[1])
    posiciones_iniciales.append([s0, r0])
posiciones_iniciales = np.array(posiciones_iniciales)

# Crear y ejecutar el sampler de emcee
sampler_fresnel = emcee.EnsembleSampler(n_caminantes, 2, log_prob_fresnel)
sampler_fresnel.run_mcmc(posiciones_iniciales, n_steps, progress=True)

# Descartamos un burn-in (por ejemplo, las primeras 200 iteraciones)
burn_in = 200
cadenas = sampler_fresnel.get_chain(discard=burn_in, flat=True)  

# Extraemos las muestras para la fuente (s) y las rendijas (r)
s_muestras = cadenas[:, 0]
r_muestras = cadenas[:, 1]

# Función integrando para la integral de camino
def integrando_camino(s, r, z):
    fase = np.pi/(lambda_ * D1) * ((s - r)**2 + (z - r)**2)
    return np.exp(1j * fase)

# Valores de z en la pantalla
z_pantalla = np.linspace(-0.4, 0.4, 200)
I_numerica = np.zeros_like(z_pantalla, dtype=float)

# Cálculo de la integral para cada valor de z mediante promedio Monte Carlo
for idx, z_val in enumerate(z_pantalla):
    val_integrando = integrando_camino(s_muestras, r_muestras, z_val)
    amplitud = np.mean(val_integrando)
    I_numerica[idx] = np.abs(amplitud)**2

# Normalización de la intensidad numérica
I_numerica_norm = I_numerica / np.max(I_numerica)

# --- Cálculo del modelo clásico ---
# I_clásico(z) ∝ cos²[(π*d/lambda_)*sinθ] * [sinc((a/lambda_)*sinθ)]²
angulo_pantalla = np.arctan(z_pantalla / D2)

def sinc_func(x):
    return np.where(x == 0, 1.0, np.sin(x) / x)

I_clasico = (np.cos(np.pi * d / lambda_ * np.sin(angulo_pantalla)))**2 \
            * (sinc_func(a / lambda_ * np.sin(angulo_pantalla)))**2
I_clasico_norm = I_clasico / np.max(I_clasico)

####################
##### INTERPRETACIÓN
####################
print(
    "INTERPRETACIÓN EJERCICIO 2\n\n"
    "Se observa que ambos modelos coinciden en la envolvente general de la intensidad,\n"
    "pero el modelo clásico predice oscilaciones más pronunciadas y de mayor frecuencia\n"
    "que la simulación basada en integral de camino. Estas diferencias pueden \n"
    "atribuirse a la naturaleza probabilística del método de Monte Carlo utilizado en la\n"
    "integral de camino, que introduce fluctuaciones y suaviza los resultados en \n"
    "comparación con el modelo clásico el cual es determinista."
)

#plt.figure(figsize=(8, 6))
#plt.plot(z_pantalla, I_numerica_norm, label="Integral de camino (emcee)")
#plt.plot(z_pantalla, I_clasico_norm, label="Modelo clásico", linestyle="--")
#plt.xlabel("z (cm)")
#plt.ylabel("Intensidad normalizada")
#plt.title("Difracción de Fresnel: Integral de camino vs Modelo clásico")
#plt.legend()
#plt.savefig("2.pdf")
#plt.show()
