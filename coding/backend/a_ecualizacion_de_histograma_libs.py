import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar la imagen
def mostrar_imagen(titulo, imagen):
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Cargar la imagen en escala de grises
ruta_imagen = 'img.jpeg'  # Reemplaza con la ruta de tu imagen
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen fue cargada correctamente
if imagen is None:
    print("Error al cargar la imagen.")
else:
    # Aplicar ecualización de histograma
    imagen_ecualizada = cv2.equalizeHist(imagen)
    
    # Mostrar imagen original y ecualizada en un subplot
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(imagen)
    plt.title('Imagen Original')

    plt.subplot(122)    
    plt.imshow(imagen_ecualizada)
    plt.title('Imagen Ecualizado')

    plt.show()
    

    # Mostrar los histogramas antes y después de la ecualización
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(imagen.ravel(), 256, [0, 256])
    plt.title('Histograma Original')

    plt.subplot(122)
    plt.hist(imagen_ecualizada.ravel(), 256, [0, 256])
    plt.title('Histograma Ecualizado')

    plt.show()
