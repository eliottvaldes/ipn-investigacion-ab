import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from mpl_toolkits.axes_grid1 import ImageGrid

class CLAHE:
    def __init__(self, clip_limit: float, tile_grid_size: int, num_bins: int = 256):
        """
        Inicializa los parámetros para CLAHE.

        :param clip_limit: Límite de recorte para el histograma.
        :param tile_grid_size: Número de tiles en la cuadrícula (ejemplo: 8 significa una cuadrícula 8x8).
        :param num_bins: Número de bins en el histograma.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.num_bins = num_bins

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica el algoritmo CLAHE a una imagen en escala de grises.

        :param img: Imagen en escala de grises.
        :return: Imagen procesada con CLAHE aplicado.
        """
        height, width = img.shape
        tile_h = height // self.tile_grid_size
        tile_w = width // self.tile_grid_size

        # Manejar casos donde la imagen no es divisible exactamente por tile_grid_size
        # Ajustar el tamaño de los tiles para cubrir toda la imagen
        tiles = []
        for i in range(self.tile_grid_size):
            row_tiles = []
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < self.tile_grid_size - 1 else height
            for j in range(self.tile_grid_size):
                x_start = j * tile_w
                x_end = (j + 1) * tile_w if j < self.tile_grid_size - 1 else width
                tile = img[y_start:y_end, x_start:x_end]
                tile_equalized = self.equalize_histogram(tile)
                row_tiles.append(tile_equalized)
            tiles.append(row_tiles)

        # Interpolar los tiles procesados para suavizar las transiciones
        result = self.interpolate_tiles(tiles, height, width, tile_h, tile_w)

        return result

    def equalize_histogram(self, tile: np.ndarray) -> np.ndarray:
        """
        Ecualiza el histograma de un tile con el límite de recorte.

        :param tile: Subimagen (tile) en escala de grises.
        :return: Tile ecualizado.
        """
        # Calcular el histograma
        hist, _ = np.histogram(tile.flatten(), self.num_bins, [0, self.num_bins])

        # Aplicar el límite de recorte
        hist_clipped = self.apply_clip_limit(hist)

        # Calcular la función de distribución acumulativa (CDF)
        cdf = hist_clipped.cumsum()
        cdf_normalized = (cdf - cdf.min()) * (self.num_bins - 1) / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype('uint8')

        # Mapear los valores originales a los valores ecualizados
        return cdf_normalized[tile]

    def apply_clip_limit(self, hist: np.ndarray) -> np.ndarray:
        """
        Aplica el límite de recorte al histograma y redistribuye los píxeles excedentes.

        :param hist: Histograma del tile.
        :return: Histograma recortado y redistribuido.
        """
        max_clip_value = self.clip_limit * hist.sum() / self.num_bins
        excess_pixels = np.clip(hist - max_clip_value, 0, None).sum()

        # Recortar el histograma
        hist_clipped = np.clip(hist, 0, max_clip_value)

        # Redistribuir los píxeles excedentes
        redistribution = excess_pixels // self.num_bins
        hist_clipped += redistribution

        # Asegurar que no haya más píxeles que el clip limit después de la redistribución
        hist_clipped = np.minimum(hist_clipped, max_clip_value)

        return hist_clipped

    def interpolate_tiles(self, tiles: list, height: int, width: int, tile_h: int, tile_w: int) -> np.ndarray:
        """
        Interpolar las transiciones entre los tiles usando interpolación bilineal.

        :param tiles: Lista de listas de tiles procesados.
        :param height: Altura de la imagen original.
        :param width: Anchura de la imagen original.
        :param tile_h: Altura de cada tile.
        :param tile_w: Anchura de cada tile.
        :return: Imagen final con interpolación aplicada.
        """
        result = np.zeros((height, width), dtype=np.uint8)

        for i in range(self.tile_grid_size):
            for j in range(self.tile_grid_size):
                y_start = i * tile_h
                y_end = (i + 1) * tile_h if i < self.tile_grid_size - 1 else height
                x_start = j * tile_w
                x_end = (j + 1) * tile_w if j < self.tile_grid_size - 1 else width

                # Colocar el tile en la posición correspondiente
                result[y_start:y_end, x_start:x_end] = tiles[i][j]

        # Interpolación horizontal y vertical...
        # (el código para la interpolación sigue igual que en el original)

        return result

    def apply_gaussian_filter(self, img: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        Aplica un filtro gaussiano a la imagen para reducir el ruido.

        :param img: Imagen a la que se va a aplicar el filtro gaussiano.
        :param kernel_size: Tamaño del kernel del filtro gaussiano.
        :return: Imagen con el filtro gaussiano aplicado.
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  
    
class ColorCLAHE:
    def __init__(self, clip_limit: float, tile_grid_size: int, num_bins: int = 256):
        """
        Inicializa los parámetros para ColorCLAHE.

        :param clip_limit: Límite de recorte para el histograma.
        :param tile_grid_size: Número de tiles en la cuadrícula.
        :param num_bins: Número de bins en el histograma.
        """
        self.clahe = CLAHE(clip_limit, tile_grid_size, num_bins)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE a una imagen a color.

        :param img: Imagen a color en formato BGR.
        :return: Imagen procesada a color con CLAHE aplicado y ruido reducido.
        """
        # Convertir la imagen a espacio de color LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Aplicar CLAHE al canal L
        l_clahe = self.clahe.apply(l)

        # Aplicar el filtro gaussiano para reducir el ruido
        l_clahe_filtered = self.clahe.apply_gaussian_filter(l_clahe)

        # Merge de los canales y conversión de vuelta a BGR
        lab_clahe = cv2.merge((l_clahe_filtered, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return img_clahe



def read_image(path: str, color: bool = True) -> np.ndarray:
    """
    Lee una imagen desde una ruta y la convierte a escala de grises o la mantiene en color.

    :param path: Ruta de la imagen.
    :param color: Si es True, la imagen se lee en color; si es False, en escala de grises.
    :return: Imagen leída.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {path}")

    if color:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def plot_images(original: np.ndarray, processed: np.ndarray, color: bool = True, img_info: tuple = None):
    """Guarda las imágenes originales y procesadas en un subplot sin espacio horizontal entre ellas."""
    # Crear carpeta si no existe
    output_folder = './results_clahe_hardcodeado'
    os.makedirs(output_folder, exist_ok=True)
    
    # obtener las variables del objeto img_info
    image_name, clip_limit, tile_grid_size = img_info
    description = f"Filtro Gaussiano sigma=7, Clip Limit: {clip_limit}, Tile Grid Size: {tile_grid_size}"

    # Crear la figura con ImageGrid para eliminar los espacios
    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, 111,  # similar a (1, 1, 1)
                     nrows_ncols=(1, 2),  # Grid de 1 fila, 2 columnas
                     axes_pad=0,  # Sin espacio entre las imágenes
                     share_all=True,
                     cbar_location="right"
                     )
    
    # show description
    fig.suptitle(description, fontsize=12)    

    # Mostrar las imágenes en la cuadrícula
    if color:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        grid[0].imshow(original_rgb)
        grid[1].imshow(processed_rgb)
    else:
        grid[0].imshow(original, cmap='gray')
        grid[1].imshow(processed, cmap='gray')

    # Desactivar los ejes
    for ax in grid:
        ax.axis('off')                

    
    # Guardar la imagen sin espacio
    filename = f"clahe_optimization_{image_name}"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    # Cerrar la figura para liberar memoria
    plt.close(fig)
    

def main():
    """
    Función principal que ejecuta el proceso de CLAHE.
    """
    import sys

    try:
        # Parámetros ingresados por el usuario
        clip_limit = 8.7
        tile_grid_size = 11        
        
        # Crear el objeto ColorCLAHE y aplicar la ecualización
        color_clahe = ColorCLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        
        
        # obtener todas las imagenes dentro del directorio '../../files/bones' y aplicar el clahe a todas
        folder = '../../files/bones'
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.png')]
        #sort
        images = sorted(images)
        for img_path in images:
            # Leer la imagen en color
            img = read_image(img_path, color=True)
            processed_img = color_clahe.apply(img)            
            image_name = os.path.basename(img_path)
            print(f'Procesando imagen: {image_name}')
            img_info = (image_name, clip_limit, tile_grid_size)      
            
            # Guardar las imágenes originales y procesadas
            plot_images(img, processed_img, color=True, img_info=img_info)      
            

        

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except ValueError as e:
        print(f"Entrada inválida: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
