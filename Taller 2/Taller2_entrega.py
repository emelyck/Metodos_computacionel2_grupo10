import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime
from PIL import Image
from matplotlib.path import Path
from numba import jit

##################################
#########  PARTE 1
##################################
def datos_prueba(t_max: float, dt: float, amplitudes: NDArray[float], frecuencias: NDArray[float], ruido: float = 0.0) -> tuple[NDArray[float], NDArray[float]]:
    ts = np.arange(0., t_max, dt)
    ys = np.zeros_like(ts, dtype=float)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
    return ts, ys
    
# Parámetros iniciales
t_max = 1.0
dt = 0.001
amplitudes = np.array([1.0, 1.5, 1.2])
frecuencias = np.array([50, 500, 200])
ruido = 0.1

# Generar datos con y sin ruido
ts, ys_sin_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
_, ys_con_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=ruido)

def Fourier(t:NDArray[float], y:NDArray[float], f:NDArray[float]) -> complex:
      r = np.zeros((len(f), len(t)), dtype=complex)
      for j in range(len(f)):
            for i in range(len(t)):
                r [j,i] = (y[i]* np.exp(-2j * np.pi * t[i] * f[j]))
      return np.sum((r), axis=1 )


frecuencias= np.linspace(0, 300, 50)
Fourier_sin_ruido= Fourier(ts, ys_sin_ruido, frecuencias )
Fourier_con_ruido = Fourier(ts, ys_con_ruido, frecuencias)


fig, ax = plt.subplots(1,2, figsize=(10,6), layout='constrained')
ax[0].bar(frecuencias , abs(Fourier_sin_ruido),  width = 5  )
ax[0].set_title("Frecuencia (Hz) vs Magnitud sin ruido ")
ax[0].set_xlabel("Frecuencia (Hz)")
ax[0].set_ylabel("Magnitud")
ax[0].grid(True)
ax[1].bar(frecuencias , abs(Fourier_con_ruido),  width = 5)
ax[1].set_title("Frecuencia (Hz) vs Magnitud con ruido ")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Magnitud")
plt.savefig("1.a.pdf")

print("1.a) Aumento de ruido magnifica todas las frecuencias  y genera desorden al momento de observar la sinuosoidal.")


#1.b 

# Función para generar datos de prueba
def datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0):
    ts = np.arange(0., t_max, dt)
    ys = np.zeros_like(ts, dtype=float)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
    return ts, ys

# Función para calcular el FWHM en función de t_max
def obtener_FWHM_por_t(t_max_values, dt, amplitudes, frecuencias, ruido):
    fwhm_values = []
    for t_max in t_max_values:
        ts, ys = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido)
        
        # Transformada de Fourier
        fft_values = np.fft.fft(ys)
        fs = np.fft.fftfreq(len(ts), d=dt)
        abs_fft_values = np.abs(fft_values)

        # Encontrar picos y calcular FWHM
        peaks, _ = find_peaks(abs_fft_values, height=0.5)
        if len(peaks) > 0:
            widths, _, _, _ = peak_widths(abs_fft_values, peaks, rel_height=0.5)
            fwhm = widths[0] * (fs[1] - fs[0]) 
            fwhm_values.append(fwhm)
        else:
            fwhm_values.append(np.nan)

    # Eliminar valores NaN
    valid_indices = ~np.isnan(fwhm_values)
    return t_max_values[valid_indices], np.array(fwhm_values)[valid_indices]

# Parámetros iniciales
dt = 0.001
amplitudes = np.array([1.0])
frecuencias = np.array([50])  # Frecuencia central
ruido = 0.0
t_max_values = np.linspace(10, 300, 50)  # Valores de t_max

# Obtener los valores de FWHM
t_max_values_clean, fwhm_values_clean = obtener_FWHM_por_t(t_max_values, dt, amplitudes, frecuencias, ruido)

# Ajuste de la relación FWHM ~ 1/t
fit_params = np.polyfit(np.log(t_max_values_clean), np.log(fwhm_values_clean), 1)
ajuste_potencia = np.exp(fit_params[1]) * t_max_values_clean**fit_params[0]

# Graficar resultados
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica log-log con estilo moderno
axs[0].loglog(t_max_values_clean, fwhm_values_clean, 'o', label="Datos", color='darkred')
axs[0].loglog(t_max_values_clean, ajuste_potencia, '--', label="Ajuste", color='black')
axs[0].set_xlabel("t_max (s)")
axs[0].set_ylabel("FWHM (Hz)")
axs[0].set_title("FWHM vs t_max (log-log)")
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend()
axs[0].set_facecolor("#f5f5f5")

# Gráfica lineal con diferente estilo
axs[1].plot(t_max_values_clean, fwhm_values_clean, 's', label="Datos", color='blue')
axs[1].plot(t_max_values_clean, ajuste_potencia, ':', label="Ajuste", color='green')
axs[1].set_xlabel("t_max (s)")
axs[1].set_ylabel("FWHM (Hz)")
axs[1].set_title("FWHM vs t_max (lineal)")
axs[1].grid(True, linestyle='-.', alpha=0.5)
axs[1].legend()
axs[1].set_facecolor("#e8f4f8")

plt.tight_layout()
###1.C

# Cargar los datos usando el método de la tarea anterior
with open('OGLE-LMC-CEP-0001.dat', 'r') as file:
    lines = file.readlines()

# Procesar los datos
Listas = []
for line in lines:
    nueva_cadena = line.replace("\n", "").split()
    Listas.append(nueva_cadena)
Matriz = np.array(Listas, dtype=float)

time = Matriz[:, 0]  # Columna de tiempo (días)
intensity = Matriz[:, 1]  # Columna de intensidad
y_uncertainty = Matriz[:, 2]  # Columna de incertidumbre

# 1. Analizar las diferencias temporales para Nyquist
delta_t = np.diff(time)
if len(delta_t) == 0:
    raise ValueError("No se encontraron diferencias temporales. Verifica los datos cargados.")

# Histograma de diferencias temporales
plt.figure(figsize=(8, 6))
plt.hist(delta_t, bins=50, alpha=0.7, color='blue', label="Intervalos de tiempo")
plt.xlabel("Intervalo de tiempo (días)")
plt.ylabel("Frecuencia")
plt.title("Histograma de diferencias temporales")
plt.grid(True)
plt.savefig("1.c_histogram.pdf")

# Calcular la frecuencia de Nyquist usando la mediana
t_median = np.median(delta_t)
f_nyquist = 1 / (2 * t_median)
print(f"1.c) f Nyquist: {f_nyquist:.4f} 1/día")

# Aplicar un filtro pasa-bajas basado en Nyquist
fc = f_nyquist * 0.5  # Frecuencia de corte por debajo de Nyquist
b, a = butter(4, fc / f_nyquist, btype='low', analog=False)
filtered_intensity = filtfilt(b, a, intensity)

# 2. Transformada de Fourier para encontrar f_true
adjusted_intensity = filtered_intensity - np.mean(filtered_intensity)
frequencies = np.linspace(0, 8, 5000)
fourier_transform = np.array([np.sum(adjusted_intensity * np.exp(-2j * np.pi * f * time)) for f in frequencies])
amplitude_spectrum = np.abs(fourier_transform)

# Verificar si hay picos disponibles
peaks, _ = find_peaks(amplitude_spectrum, height=np.mean(amplitude_spectrum) + 0.1 * np.std(amplitude_spectrum))
if len(peaks) == 0:
    raise ValueError("No se encontraron picos en el espectrograma. Verifica los datos y el rango de frecuencias.")

# Encontrar el pico más prominente
prominent_peak_idx = peaks[np.argmax(amplitude_spectrum[peaks])]
f_true = frequencies[prominent_peak_idx]
print(f"1.c) f true: {f_true:.4f} 1/día")

# 3. Validación con la fase
phi = np.mod(f_true * time, 1)
plt.figure(figsize=(8, 6))
plt.scatter(phi, intensity, alpha=0.7, s=10, label="Datos ajustados")
plt.xlabel("Fase (φ)")
plt.ylabel("Intensidad")
plt.title("Intensidad vs Fase")
plt.grid(True)
plt.legend()
plt.savefig("1.c.pdf")

# Graficar el espectrograma
plt.figure(figsize=(8, 6))
plt.plot(frequencies, amplitude_spectrum, color='blue', label="Espectrograma")
plt.axvline(f_true, color='red', linestyle='--', label=f"f_true = {f_true:.4f} 1/día")
plt.xlabel("Frecuencia (1/día)")
plt.ylabel("Amplitud")
plt.title("Espectrograma de la señal")
plt.grid(True)
plt.legend()
plt.savefig("1.c_spectrogram.pdf")

print("Los resultados fueron guardados en 1.c.pdf y 1.c_spectrogram.pdf.")

# Conclusiones impresas
print("Los resultados fueron guardados en 1.c.pdf y 1.c_spectrogram.pdf.")
##################################
#########  PARTE 2. Transformada rápida
##################################
#2.a. Comparativa
# Paso 2: Aplicar la Transformada Rápida de Fourier
H_fft = np.fft.rfft(H)

# Paso 3: Obtener las frecuencias
Delta_t = np.mean(np.diff(t))  # Promedio de las diferencias de tiempo
n = len(H)
f = np.fft.rfftfreq(n, Delta_t)

# Paso 4: Encontrar la frecuencia de oscilación más significativa
mag = np.abs(H_fft)

# Encontramos el índice de la frecuencia máxima
pico_index = np.argmax(mag)
f_fast = f[pico_index]

print(f'La frecuencia de oscilación f_fast es: {f_fast} Hz')

def Fourier(t: np.ndarray, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = np.zeros((len(f), len(t)), dtype=complex)
    for j in range(len(f)):
        for i in range(len(t)):
            r[j, i] = y[i] * np.exp(-2j * np.pi * t[i] * f[j])
    return np.sum(r, axis=1)

# Paso 3: Definir las frecuencias
f_gen = np.linspace(0, 14, 1050)  # Ajusta el rango según sea necesario

# Calcular la transformada de Fourier
Fourier_transformada_gen = Fourier(t, H, f_gen)

# Paso 4: Calcular la magnitud y encontrar la frecuencia máxima
mag_gen = np.abs(Fourier_transformada_gen)
f_general = f_gen[np.argmax(mag_gen)]

# Paso 5: Visualizar los resultados
plt.figure(figsize=(15, 8))
plt.plot(f_gen, mag_gen)#, width=5, color='red')
plt.title("Frecuencia (Hz) vs Magnitud", fontsize=18)
plt.xlabel("Frecuencia (Hz)", fontsize=16)
plt.ylabel("Magnitud", fontsize=16)
plt.tick_params(axis='both', labelsize=13)
plt.grid(True)
plt.show()

# Imprimir la frecuencia máxima
print(f'La frecuencia máxima es: {f_general} Hz')


# Crear un subplot
fig, ax = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True, sharex=True)

# Gráfica 1: FFT
ax[0].plot(f, mag, "-", color='blue')
ax[0].set_title('FFT', fontsize=18)
ax[0].set_xlabel('Frecuencia (Hz)', fontsize=16)
ax[0].set_ylabel('Magnitud', fontsize=16)
ax[0].set_xlim(0, max(f))
ax[0].tick_params(axis='both', labelsize=13)
ax[0].grid()

# Gráfica 2: Método General
ax[1].plot(f_gen, mag_gen)#, width=5, color='red')
ax[1].set_title("Frecuencia (Hz) vs Magnitud", fontsize=18)
ax[1].set_xlabel("Frecuencia (Hz)", fontsize=16)
ax[1].set_ylabel("Magnitud", fontsize=16)
ax[1].tick_params(axis='both', labelsize=13)
ax[1].grid(True)

# Mostrar la figura
plt.show()


phi_fast = np.mod(f_fast * t, 1)
phi_general = np.mod(f_general * t, 1)

plt.figure(figsize=(15, 6))
plt.scatter(phi_fast, H, label='H vs. phi_fast', color='blue')
plt.scatter(phi_general, H, label='H vs. phi_general', color='red')
plt.title('H como función de las fases', fontsize=18)
plt.xlabel('Fase (phi)', fontsize=16)
plt.ylabel('H', fontsize=16)
plt.ylim(min(phi_fast)-1.5, max(phi_general)+0.2)
plt.legend()
plt.tick_params(axis='both', labelsize=13)
plt.grid(True)

plt.savefig('2.a.pdf')
plt.show()


#2.b Manchas solares
datos_filtrados = datos[
    (datos["Year"] < 2012) |  
    ((datos["Year"] == 2012) & (datos["Month"] == 1) & (datos["Day"] == 1))  
].copy()  

datos_filtrados["datetime"] = pd.to_datetime(datos_filtrados.drop("SSN", axis=1))
fechas = datos_filtrados["datetime"]
manchas = datos_filtrados["SSN"].to_numpy()

if len(manchas) > len(fechas):
    manchas_recortadas = manchas[:len(fechas)]  # Tomar solo los primeros valores para que coincidan
else:
    manchas_recortadas = manchas

# Transformada
transformada_sol = np.fft.rfft(manchas)
f_sol = np.fft.rfftfreq(len(fechas), d=1)

# Calcular la magnitud
mag_sol = np.abs(transformada_sol)

# Encontrar la frecuencia máxima
indice_max = np.argmax(mag_sol)
frecuencia_max_sol = f_sol[indice_max]

# Verificar la frecuencia máxima
print("Frecuencia máxima:", f"{frecuencia_max_sol:e}")

# Calcular el período en años
if frecuencia_max_sol != 0:
    P_solar = 1 / frecuencia_max_sol  # El período en días
    P_solar_años = P_solar / 365.25  # Convertir a años
    print(f'2.b.a) P_solar = {P_solar_años:.2f} años')
else:
    print("No se encontró una frecuencia significativa.")

# Visualizar la gráfica de la transformada en ejes log-log
plt.figure(figsize=(15, 10))
plt.loglog(f_sol[1:len(f_sol)//2], mag_sol[1:len(mag_sol)//2], marker='o', color='orange')
plt.title('Transformada de Fourier de las Manchas Solares', fontsize=18)
plt.xlabel('Frecuencia (1/días)', fontsize=16)
plt.ylabel('Magnitud', fontsize=16)
plt.tick_params(axis='both', labelsize=13)
plt.grid(True)
plt.show()

# Ejecutando el código, se observa un pico en la región entre 10^-4 y 10^-3 Hz
# Seleccionando los datos en esa región

mask = (f_sol >= 1e-4) & (f_sol <= 1e-3)

mag_filtrada = mag_sol[mask]
f_sol_filtrada = f_sol[mask]

indice_max_check = np.argmax(mag_filtrada)

frecuencia_max_sol_check = f_sol_filtrada[indice_max_check]
mag_max_sol_check = mag_filtrada[indice_max_check]

print("Frecuencia máxima limitando los datos al intervalo 10⁻⁴ y 10⁻³ Hz:", f"{frecuencia_max_sol_check:e}")

if frecuencia_max_sol_check != 0:
    P_solar_check = 1 / frecuencia_max_sol_check  # El período en días
    P_solar_check_años = P_solar_check / 365.25  # Convertir a años
    print(f'2.b.a) P_solar = {P_solar_check_años} años')
else:
    print("No se encontró una frecuencia significativa.")

# 2.b.b

# Parámetros
M = 50  # Número de armónicos
N = len(mag_sol)  # Longitud de los datos originales
start_date = pd.to_datetime('2012-01-02')
end_date = pd.to_datetime('2025-02-17')

# Crear un rango de días desde la primera fecha hasta la fecha final deseada
fechas_rango = pd.date_range(start=start_date, end=end_date, freq='D')
t = np.arange(len(fechas_rango))  # Días desde el inicio de la predicción

# Lista para almacenar los valores de la predicción
y_pred_manual = []

# Calcular la reconstrucción de la señal iterando sobre cada valor de t
for t_i in t:
    y_value = np.real((1/N) * np.sum(mag_sol[:M] * 0.5 * np.exp(2j * np.pi * f_sol[:M] * t_i)))
    y_pred_manual.append(y_value)

# Convertir la lista a un array de NumPy
y_pred_manual = np.array(y_pred_manual)
y_t=y_pred_manual[-1]
print(f'2.b.b n_manchas_hoy = {y_t}')


# Graficar la predicción utilizando el bucle for
plt.figure(figsize=(15, 6))
plt.plot(fechas_rango, y_pred_manual, label="Predicción de manchas solares", color='red')
plt.plot(fechas, manchas, label="Mediciones", color="orange")

# Formato del eje x
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Cada 2 años
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Mostrar solo el año

plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Número de manchas solares", fontsize=14)
plt.title("Manchas solares", fontsize=16)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

plt.savefig("2.b.pdf", format='pdf')

##################################
#########  PARTE 3
##################################
datos = pd.read_csv("Taller 2\\manchas.txt", sep=r'\s+', header=1, names=["Year", "Month", "Day", "SSN"], engine="python")
datos = datos.dropna()

datos_filtrados = datos[
    (datos["Year"] < 2012) |  
    ((datos["Year"] == 2012) & (datos["Month"] == 1) & (datos["Day"] == 1))  
].copy()  

datos_filtrados.loc[:, "Date"] = pd.to_datetime(datos_filtrados[["Year", "Month", "Day"]])

fechas = datos_filtrados["Date"].to_numpy() 
manchas = datos_filtrados["SSN"].to_numpy()  

# Transformada
transformada = np.fft.fft(manchas)
frecuencias = np.fft.fftfreq(len(fechas)) 

#Ver 3.1pdf
"""

fig, ax = plt.subplots(5, 2, figsize=(10, 12))
alpha = np.array([1, 10, 100, 1000, 2000])

for a in range(len(alpha)):
    transformada_filtro = transformada * np.exp(- (frecuencias * alpha[a])**2)
    
    # Transformada
    ax[a, 1].loglog(abs(frecuencias), abs(transformada_filtro), color="orange", label="Filtro")
    ax[a, 1].loglog(abs(frecuencias), abs(transformada), color="blue", label="Original")
    ax[a, 1].set_ylabel("Señal transformada", fontsize=12)
    ax[a, 1].set_xlabel("Frecuencia (Hz)", fontsize=12)
    ax[a, 1].legend()
    ax[a, 1].set_ylim(1e0, 1e6)

    # Inversa
    transformada_inv = np.fft.ifft(transformada_filtro).real
    ax[a, 0].plot(fechas, manchas, color="blue", label="Señal original")
    ax[a, 0].plot(fechas, transformada_inv, color="orange", label="Filtro")
    ax[a, 0].set_ylabel("Número de manchas\n solares (SSN)", fontsize=12)
    ax[a, 0].set_xlabel("Fecha", fontsize=12)
    ax[a, 0].legend()

    fig.text(1.02, 0.85 - a * 0.2, f"Alpha = {alpha[a]}", fontsize=10, color="red", ha="left")

plt.tight_layout()
"""


############
#3.a
##########
imagen = Image.open("Taller 2\\Noisy_Smithsonian_Castle.jpg").convert("L")
imagen_np = np.array(imagen).astype(float)

# Transformada
fft_imagen = np.fft.fft2(imagen_np)
fft_centrada = np.fft.fftshift(fft_imagen)

# Filtradoo de los datos
fft_centrada[215:550, 408:415] = 0  
fft_centrada[215:550, 608:616] = 0  
fft_centrada[0:374, 506:518] = 0    
fft_centrada[387:765, 506:518] = 0  

# Transformada inversa
fft_inversa = np.fft.ifft2(np.fft.ifftshift(fft_centrada))
imagen_filtrada = np.abs(fft_inversa)


############
#3.b
#############
# Cargar la imagen en escala de grises
imagen_2 = Image.open("Taller 2\\catto.png").convert("L")
img = np.array(imagen_2).astype(float)

# Transformada de Fourier
ft_img = np.fft.fft2(img)
ft_img_centrada = np.fft.fftshift(ft_img)

# Calcular el espectro en escala logarítmica
gato = np.log(1 + np.abs(ft_img_centrada))

# Definir los vértices de los dos cuadriláteros
vertices_1 = np.array([(210, 420), (210, 450), (370, 420), (370, 375)])  
vertices_2 = np.array([(385, 375), (385, 341), (540, 270), (540,334)])  

# Crear máscaras para ambos cuadriláteros
y, x = np.meshgrid(np.arange(gato.shape[1]), np.arange(gato.shape[0]))
coords = np.stack((x.ravel(), y.ravel()), axis=-1)

path_1 = Path(vertices_1)
mask_1 = path_1.contains_points(coords).reshape(gato.shape)

path_2 = Path(vertices_2)
mask_2 = path_2.contains_points(coords).reshape(gato.shape)

# Poner en negro las áreas dentro de ambos cuadriláteros en ft_img_centrada
ft_img_centrada[mask_1] = 0
ft_img_centrada[mask_2] = 0

# Calcular el espectro corregido en escala logarítmica
gato_corregido = np.log(1 + np.abs(ft_img_centrada))

ft_img_inversa = np.fft.ifftshift(ft_img_centrada)  
img_recuperada = np.fft.ifft2(ft_img_inversa) 
img_recuperada = np.abs(img_recuperada)  
