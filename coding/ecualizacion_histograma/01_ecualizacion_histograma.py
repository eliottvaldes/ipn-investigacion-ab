import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_HEIGHT = 500 # Altura máxima de la imagen para evitar altos tiempos de ejecución

def read_image(path):
    """
    Lee una imagen y la convierte de BGR a RGB.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # obtener el tamaño de la imagen. Se debe redimencionar solo en caso de que el alto sea mayor a 250 prixeles. El alto maximo es de 250 y el ancho se ajusta proporcionalmente
    alto, ancho = img.shape[:2]
    print(f'Mediadas de imagen:\tAlto: {alto}, Ancho: {ancho}')
    if alto > MAX_HEIGHT:
        proporcion = MAX_HEIGHT / alto
        img = cv2.resize(img, (int(ancho * proporcion), MAX_HEIGHT))
        alto, ancho = img.shape[:2]
        print(f'Mediadas de imagen ajustadas:\t Alto: {alto}, Ancho: {ancho}')
    return img

def equalize_hist_cv2(img):
    """
    Realiza la ecualización del histograma usando funciones de OpenCV.
    """
    # Convertir a espacio de color YCrCb
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Separar los canales
    y, cr, cb = cv2.split(img_y_cr_cb)
    # Ecualizar el canal de luminancia
    y_eq = cv2.equalizeHist(y)
    # Unir los canales
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    # Convertir de vuelta a RGB
    img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
    return img_eq

def hist_equalization(channel):
    """
    Ecualiza el histograma de un canal de imagen usando numpy.
    """
    # Calcular el histograma
    hist, bins = np.histogram(channel.flatten(), 256, [0,256])
    # Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    # Normalizar la CDF
    cdf_normalized = cdf * hist.max() / cdf.max()
    # Máscara los valores cero del CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Ecualizar
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Rellenar los valores enmascarados con cero
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # Mapear los valores del canal original a los nuevos valores ecualizados
    channel_eq = cdf[channel]
    return channel_eq

def equalize_hist_numpy(img):
    """
    Realiza la ecualización del histograma usando operaciones nativas de numpy.
    """
    # Convertir a espacio de color YCrCb
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Separar los canales
    y, cr, cb = cv2.split(img_y_cr_cb)
    # Ecualizar el canal de luminancia
    y_eq = hist_equalization(y)
    # Unir los canales
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    # Convertir de vuelta a RGB
    img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
    return img_eq

def plot_images(original, cv2_eq, numpy_eq, entropies):
    """
    Muestra las imágenes en un solo plot.
    """
    plt.figure(figsize=(15,5))
    titles = ['Imagen Original', 'Ecualización con cv2', 'Ecualización desde cero']
    # Agregar todas las entropías a los títulos
    for i in range(3):
        titles[i] += f'\nEntropía de Shannon: {entropies[i]:.6f}'
    
    images = [original, cv2_eq, numpy_eq]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.show()
    
    
def calculate_shannon_entropy(img):
    """Calcular la entropía de Shannon de una imagen en color."""
    # Convertir la imagen a un arreglo 1D
    img_flat = img.flatten()
    # Calcular el histograma (counts)
    hist_counts, _ = np.histogram(img_flat, bins=256, range=(0, 256))
    # Normalizar para obtener probabilidades
    total_pixels = np.sum(hist_counts)
    if total_pixels == 0:
        return 0  # Evitar división por cero
    probabilities = hist_counts / total_pixels
    # Eliminar ceros para evitar log(0)
    probabilities = probabilities[probabilities > 0]
    # Calcular la entropía de Shannon
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    return shannon_entropy
    

def main():
    # Ruta de la imagen
    image_path = '../files/bones/4360.png'
    # Leer imagen
    img = read_image(image_path)
    img_entropy = calculate_shannon_entropy(img)
    # Ecualización con cv2
    img_cv2_eq = equalize_hist_cv2(img)
    img_cv2_eq_entropy = calculate_shannon_entropy(img_cv2_eq)
    # Ecualización con numpy
    img_numpy_eq = equalize_hist_numpy(img)
    img_numpy_eq_entropy = calculate_shannon_entropy(img_numpy_eq)
    # Mostrar resultados
    plot_images(img, img_cv2_eq, img_numpy_eq, [img_entropy, img_cv2_eq_entropy, img_numpy_eq_entropy])
    
    # Mostrar las entropías en consola para cada imagen
    print(f'Entropía de Shannon de la imagen original: {img_entropy:.6f}')
    print(f'Entropía de Shannon de la imagen ecualizada con cv2: {img_cv2_eq_entropy:.6f}')
    print(f'Entropía de Shannon de la imagen ecualizada desde cero: {img_numpy_eq_entropy:.6f}')

if __name__ == "__main__":
    main()
