import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime
from PIL import Image
from matplotlib.path import Path

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