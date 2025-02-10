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
# Parámetros iniciales
t_max = 300  # Rango de tiempo ajustado
intervalo_tiempo = (10, 300)  # Rango de tiempo en segundos
amplitudes = np.array([1.0])  # Amplitud única
frecuencia_conocida = 50  # Frecuencia conocida en Hz
dt = 0.001  # Resolución en el tiempo
total_frequencies = 500  # Suficientes puntos en el eje de frecuencias

def calculate_fwhm(frequencies: NDArray[np.float64], amplitudes: NDArray[np.float64]) -> float:
    """Calcular el ancho completo a media altura (FWHM)."""
    peak_index = np.argmax(amplitudes)  # Índice del pico más alto
    peak_height = amplitudes[peak_index]  # Altura máxima
    half_height = peak_height / 2  # Mitad de altura

    # Encontrar índices donde cruza la mitad de altura
    left_index = np.where(amplitudes[:peak_index] <= half_height)[0][-1]
    right_index = np.where(amplitudes[peak_index:] <= half_height)[0][0] + peak_index

    return frequencies[right_index] - frequencies[left_index]  # Ancho del pico

@jit(nopython=True)
def Fourier(t: NDArray[np.float64], y: NDArray[np.float64], f: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Calcular la transformada de Fourier de una señal."""
    result = np.zeros(len(f), dtype=np.complex128)
    for j in range(len(f)):
        result[j] = np.sum(y * np.exp(-2j * np.pi * t * f[j]))
    return result

@jit(nopython=True)
def datos_prueba(t_max: float, dt: float, amplitudes: NDArray[np.float64], frecuencias: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generar datos de prueba con una frecuencia conocida."""
    ts = np.linspace(0, t_max, int(t_max / dt))
    ys = np.zeros_like(ts)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    return ts, ys

# Generar datos para un rango de tiempo ajustado
ts, ys = datos_prueba(t_max, dt, amplitudes, np.array([frecuencia_conocida]))
intervalo_indices = (int(intervalo_tiempo[0] / dt), int(intervalo_tiempo[1] / dt))
ts_intervalo = ts[intervalo_indices[0]:intervalo_indices[1]]
ys_intervalo = ys[intervalo_indices[0]:intervalo_indices[1]]

# Calcular transformada de Fourier
frecuencias = np.linspace(0, 100, total_frequencies)  # Rango de frecuencias
fourier_result = Fourier(ts_intervalo, ys_intervalo, frecuencias)
amplitudes_fourier = np.abs(fourier_result)  # Amplitud absoluta de la transformada
fwhm = calculate_fwhm(frecuencias, amplitudes_fourier)  # Calcular el FWHM

# Imprimir el valor de FWHM
print(f"Intervalo de tiempo: {intervalo_tiempo[0]}s - {intervalo_tiempo[1]}s, FWHM: {fwhm:.4f} Hz")

# Graficar el FWHM como barra
plt.figure(figsize=(8, 6))
plt.bar([frecuencia_conocida], [fwhm], width=5, alpha=0.7, color="green", label="FWHM")
plt.yscale("log")
plt.xscale("log")
plt.title("FWHM para la frecuencia conocida")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("FWHM (Hz)")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("1.b_single_bar.pdf")  # Guardar la gráfica

print("1.b) Se generó la gráfica 1.b_single_bar.pdf mostrando el FWHM.")

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
