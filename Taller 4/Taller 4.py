import numpy as np
import emcee
import matplotlib.pyplot as plt
import re
import unicodedata
from collections import defaultdict
import random

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


########################
####### EJERCICIO 4
########################
## 4.a Limpiado de datos
# Load the text file
file_path = "The picture of Dorian Gray.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Remove the Project Gutenberg headers and footers
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE PICTURE OF DORIAN GRAY ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE PICTURE OF DORIAN GRAY ***"

start_idx = text.find(start_marker) + len(start_marker)
end_idx = text.find(end_marker)

if start_idx > len(start_marker) and end_idx > -1:
    text = text[start_idx:end_idx]

# Normalize text to remove special characters like accents and quotes
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[“”‘’'\"_]", "", text)  # Remove various types of quotes and underscores
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert text to lowercase
    return text

# Apply text normalization
cleaned_text = normalize_text(text)

# Replace artificial line breaks
cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\n\n", "#").replace("\n", "").replace("#", "\n\n")

# Save the cleaned text to a new file
cleaned_file_path = "The_picture_of_Dorian_Gray_cleaned.txt"
with open(cleaned_file_path, "w", encoding="utf-8") as file:
    file.write(cleaned_text)

# Provide the cleaned file
cleaned_file_path

## 4.b Entrenamiento y predicción
# Define n-gram size
n = 3  # You can adjust this value

# Create a dictionary to store frequency counts
ngram_freq = defaultdict(lambda: defaultdict(int))

# Iterate through the text to populate frequency table
for i in range(len(cleaned_text) - n):
    ngram = cleaned_text[i:i+n]  # Get the n-gram
    next_char = cleaned_text[i+n]  # Get the next character
    ngram_freq[ngram][next_char] += 1

# Convert frequency table to probabilities
ngram_prob = {}
for ngram, next_chars in ngram_freq.items():
    total = sum(next_chars.values())
    ngram_prob[ngram] = {char: count/total for char, count in next_chars.items()}

# Function to generate text using trained model
def generate_text(ngram_prob, length=1500):
    import random

    # Find a suitable starting n-gram (one that starts with "\n")
    start_ngrams = [ngram for ngram in ngram_prob.keys() if ngram.startswith("\n")]
    if not start_ngrams:
        start_ngrams = list(ngram_prob.keys())  # Fallback if no suitable n-gram
    current_ngram = random.choice(start_ngrams)

    # Generate text
    generated_text = current_ngram
    for _ in range(length - len(current_ngram)):
        if current_ngram in ngram_prob:
            next_chars = list(ngram_prob[current_ngram].keys())
            probabilities = list(ngram_prob[current_ngram].values())
            next_char = random.choices(next_chars, probabilities)[0]
        else:
            break  # Stop if no known continuation

        generated_text += next_char
        current_ngram = generated_text[-n:]  # Update n-gram

    return generated_text

# Generate a sample text
generated_text_sample = generate_text(ngram_prob, length=1500)

# Save generated text to a file
generated_file_path = "generated_text.txt"
with open(generated_file_path, "w", encoding="utf-8") as file:
    file.write(generated_text_sample)

# Provide the file for download
generated_file_path

## 4.c Análisis
# Define range of n-gram values to test
n_values = range(1, 8)
generated_files = {}

# Function to train and generate text for a given n
def train_and_generate(n, text_length=1500):
    ngram_freq = defaultdict(lambda: defaultdict(int))

    # Build frequency table
    for i in range(len(cleaned_text) - n):
        ngram = cleaned_text[i:i+n]
        next_char = cleaned_text[i+n]
        ngram_freq[ngram][next_char] += 1

    # Normalize frequencies
    ngram_prob = {ngram: {char: count / sum(chars.values()) for char, count in chars.items()} for ngram, chars in ngram_freq.items()}

    # Generate text
    start_ngrams = [ngram for ngram in ngram_prob.keys() if ngram.startswith("\n")]
    if not start_ngrams:
        start_ngrams = list(ngram_prob.keys())
    current_ngram = random.choice(start_ngrams)

    generated_text = current_ngram
    for _ in range(text_length - len(current_ngram)):
        if current_ngram in ngram_prob:
            next_chars = list(ngram_prob[current_ngram].keys())
            probabilities = list(ngram_prob[current_ngram].values())
            next_char = random.choices(next_chars, probabilities)[0]
        else:
            break

        generated_text += next_char
        current_ngram = generated_text[-n:]

    return generated_text

# Generate texts for different n values
for n in n_values:
    generated_text = train_and_generate(n)
    file_path = f"gen_text_n{n}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(generated_text)
    generated_files[n] = file_path

# Function to compute word match percentage
def compute_word_match_percentage(text):
    words = re.findall(r"\b[a-zA-Z']+\b", text)
    valid_words = sum(1 for word in words if word in english_words)
    return (valid_words / len(words)) * 100 if words else 0

# Compute percentages for each generated text
match_percentages = {}
for n, file_path in generated_files.items():
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    match_percentages[n] = compute_word_match_percentage(text)


# Load the list of 10,000 English words
word_list_path = "List of 10000 words.txt"

with open(word_list_path, "r", encoding="utf-8") as file:
    english_words = set(word.strip().lower() for word in file.readlines())

# Run the analysis with the word list
match_percentages = {}
for n, file_path in generated_files.items():
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    match_percentages[n] = compute_word_match_percentage(text)

# Re-generate and save the updated plot
plt.figure(figsize=(8, 5))
plt.plot(list(match_percentages.keys()), list(match_percentages.values()), marker="o", linestyle="-")
plt.xlabel("n-gram size (n)")
plt.ylabel("Percentage of valid English words")
plt.title("Effect of n-gram size on text coherence")
plt.grid(True)

plot_path = "4.pdf"
plt.savefig(plot_path)

# Provide updated plot file
plot_path


########################
####### EJERCICIO 5
########################
## 5.a Sistema determinista
# Parámetros dados
A = 1000  # Producción diaria de U-239
B = 20  # Extracción diaria de Pu-239
lambda_U239 = 42.66  # Constante de decaimiento de U-239 (por día)
lambda_Np239 = 0.294  # Constante de decaimiento de Np-239 (por día)

# Sistema de ecuaciones diferenciales
def dYdt(t, Y):
    U, Np, Pu = Y
    dU_dt = A - lambda_U239 * U
    dNp_dt = lambda_U239 * U - lambda_Np239 * Np
    dPu_dt = lambda_Np239 * Np - B * Pu
    return [dU_dt, dNp_dt, dPu_dt]

# Condiciones iniciales (sistema limpio)
Y0 = [0, 0, 0]

# Simulación de 30 días
t_span = (0, 30)
t_eval = np.linspace(0, 30, 300)  # 300 puntos para mejor resolución
sol = spi.solve_ivp(dYdt, t_span, Y0, t_eval=t_eval)

# Extraer soluciones
U_vals, Np_vals, Pu_vals = sol.y
time_vals = sol.t

# Detectar estabilidad: buscamos cuándo la variación relativa es menor a un umbral
threshold = 1e-3  # Cambio relativo menor al 0.1%
stable_index = np.argmax(
    (np.abs(np.diff(U_vals) / U_vals[:-1]) < threshold) &
    (np.abs(np.diff(Np_vals) / Np_vals[:-1]) < threshold) &
    (np.abs(np.diff(Pu_vals) / Pu_vals[:-1]) < threshold)
)
stabilization_time = time_vals[stable_index] if stable_index > 0 else None

# Graficar evolución temporal
plt.figure(figsize=(10, 5))
plt.plot(time_vals, U_vals, label="U-239")
plt.plot(time_vals, Np_vals, label="Np-239")
plt.plot(time_vals, Pu_vals, label="Pu-239")
plt.axvline(stabilization_time, color='gray', linestyle='dashed', label="Estabilización")
plt.xlabel("Tiempo (días)")
plt.ylabel("Cantidad")
plt.legend()
plt.title("Evolución Temporal de las Cantidades de U, Np y Pu")
plt.grid()

# Mostrar resultados en una tabla
df_results = pd.DataFrame({
    "Día": time_vals,
    "U-239": U_vals,
    "Np-239": Np_vals,
    "Pu-239": Pu_vals
})
print(df_results)

# Mostrar el tiempo de estabilización
stabilization_time


## 5.b Sistema estocástico
# Parámetros iniciales
A = 1000  # Producción de U-239 por día
B = 20  # Extracción de Pu-239 por día
lambda_U239 = 42.66  # Tasa de decaimiento de U-239 (1/día)
lambda_Np239 = 0.294  # Tasa de decaimiento de Np-239 (1/día)

# Reacciones (vectores de cambio)
R = np.array([
    [1, 0, 0],   # Creación de U-239
    [-1, 1, 0],  # Decaimiento de U-239 a Np-239
    [0, -1, 1],  # Decaimiento de Np-239 a Pu-239
    [0, 0, -1]   # Extracción de Pu-239
])

# Estado inicial del sistema [U, Np, Pu]
Y = np.array([0, 0, 0])

# Tiempo de simulación
t_max = 30  # Simulación de 30 días
t = 0

# Almacenar resultados
time_vals = []
U_vals = []
Np_vals = []
Pu_vals = []

# Simulación del proceso estocástico
while t < t_max:
    # Calcular tasas de reacción
    tasas = np.array([
        A,
        Y[0] * lambda_U239,
        Y[1] * lambda_Np239,
        Y[2] * B
    ])
    
    tasa_total = np.sum(tasas)
    if tasa_total == 0:
        break
    
    # Tiempo hasta la siguiente reacción (exponencial)
    tau = np.random.exponential(1 / tasa_total)
    
    # Elegir qué reacción ocurre
    r_index = np.random.choice(len(R), p=tasas / tasa_total)
    
    # Aplicar la reacción
    Y += R[r_index]
    t += tau

    # Guardar estado
    time_vals.append(t)
    U_vals.append(Y[0])
    Np_vals.append(Y[1])
    Pu_vals.append(Y[2])

# Graficar evolución temporal
plt.figure(figsize=(15, 10))
plt.plot(time_vals, U_vals, label="U-239", alpha=0.7)
plt.plot(time_vals, Np_vals, label="Np-239", alpha=0.7)
plt.plot(time_vals, Pu_vals, label="Pu-239", alpha=0.7)
plt.xlabel("Tiempo (días)")
plt.ylabel("Cantidad")
plt.legend()
plt.title("Evolución Temporal Estocástica del Sistema")
plt.grid()

# Guardar resultados en un DataFrame
df_results_stochastic = pd.DataFrame({
    "Día": time_vals,
    "U-239": U_vals,
    "Np-239": Np_vals,
    "Pu-239": Pu_vals
})
print(df_results_stochastic)

## Punto 5.c Simulación
# Número de simulaciones estocásticas
num_simulations = 100

# Almacenar todas las simulaciones
all_simulations = []

# Realizar 100 simulaciones estocásticas
for _ in range(num_simulations):
    Y = np.array([0, 0, 0])  # Estado inicial
    t = 0
    time_vals = []
    U_vals = []
    Np_vals = []
    Pu_vals = []

    while t < t_max:
        # Calcular tasas de reacción
        tasas = np.array([
            A,
            Y[0] * lambda_U239,
            Y[1] * lambda_Np239,
            Y[2] * B
        ])

        tasa_total = np.sum(tasas)
        if tasa_total == 0:
            break

        # Tiempo hasta la siguiente reacción (exponencial)
        tau = np.random.exponential(1 / tasa_total)

        # Elegir qué reacción ocurre
        r_index = np.random.choice(len(R), p=tasas / tasa_total)

        # Aplicar la reacción
        Y += R[r_index]
        t += tau

        # Guardar estado
        time_vals.append(t)
        U_vals.append(Y[0])
        Np_vals.append(Y[1])
        Pu_vals.append(Y[2])

    all_simulations.append((time_vals, U_vals, Np_vals, Pu_vals))

# Graficar
plt.figure(figsize=(10, 5))

# Graficar las 100 simulaciones estocásticas
for sim in all_simulations:
    plt.plot(sim[0], sim[1], color='blue', alpha=0.1)  # U-239
    plt.plot(sim[0], sim[2], color='orange', alpha=0.1)  # Np-239
    plt.plot(sim[0], sim[3], color='red', alpha=0.1)  # Pu-239

# Graficar la solución determinista en línea sólida
plt.plot(sol.t, sol.y[0], color='blue', label="Determinista U-239", linewidth=2)
plt.plot(sol.t, sol.y[1], color='orange', label="Determinista Np-239", linewidth=2)
plt.plot(sol.t, sol.y[2], color='red', label="Determinista Pu-239", linewidth=2)

# Configurar gráfico
plt.xlabel("Tiempo (días)")
plt.ylabel("Cantidad")
plt.title("Comparación de Simulación Determinista vs. 100 Simulaciones Estocásticas")
plt.legend()
plt.grid()

# Guardar en archivo PDF
plt.savefig("5.pdf")

## 5.d. Probabilidad de concentración crítica
# Reducir el número de simulaciones para mejorar el rendimiento
num_trials = 30
threshold_pu = 80  # Nivel crítico de Plutonio
critical_count = 0  # Contador de veces que se supera el umbral

# Realizar simulaciones
for _ in range(num_trials):
    Y = np.array([0, 0, 0])  # Estado inicial [U, Np, Pu]
    t = 0

    while t < t_max:
        # Calcular tasas de reacción solo si hay cantidad suficiente
        tasas = np.array([
            A,
            Y[0] * lambda_U239,
            Y[1] * lambda_Np239,
            Y[2] * B
        ])

        tasa_total = np.sum(tasas)
        if tasa_total == 0:
            break

        # Tiempo hasta la siguiente reacción (exponencial)
        tau = np.random.exponential(1 / tasa_total)

        # Seleccionar evento sin necesidad de np.random.choice
        random_val = np.random.rand() * tasa_total
        cumulative_sum = np.cumsum(tasas)
        r_index = np.searchsorted(cumulative_sum, random_val)

        # Aplicar la reacción
        Y += R[r_index]
        t += tau  # Evolución del tiempo

        # Revisar si alcanzó el nivel crítico de Plutonio
        if Y[2] >= threshold_pu:
            critical_count += 1
            break  # No es necesario seguir si ya alcanzó el umbral

# Calcular probabilidad
critical_probability = critical_count / num_trials

# Imprimir resultado
print(f"5) {critical_probability:.4f}")
