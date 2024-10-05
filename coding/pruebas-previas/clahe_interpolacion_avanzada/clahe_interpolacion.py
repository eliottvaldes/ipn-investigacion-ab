import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

"""Función principal que ejecuta el proceso de CLAHE."""
import sys
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def read_image(path: str, color: bool = True) -> np.ndarray:
    """Lee una imagen desde una ruta y la convierte a escala de grises o la mantiene en color."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {path}")

    return image if color else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def plot_images(original: np.ndarray, processed: np.ndarray, color: bool = True, img_info: tuple = None):
    """Guarda las imágenes originales y procesadas en un subplot sin espacio horizontal entre ellas."""
    # Crear carpeta si no existe
    output_folder = './results_clahe_hardcodeado'
    os.makedirs(output_folder, exist_ok=True)
    
    # obtener las variables del objeto img_info
    image_name, clip_limit, tile_grid_size = img_info    
    description = f"Interpolación bicúbica, Clip Limit: {clip_limit}, Tile Grid Size: {tile_grid_size}"

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

class CLAHE:
    def __init__(self, clip_limit: float, tile_grid_size: int):
        """Inicializa la clase CLAHE con los parámetros especificados."""
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Aplica el algoritmo CLAHE a una imagen en escala de grises."""
        height, width = img.shape
        tile_h = height // self.tile_grid_size
        tile_w = width // self.tile_grid_size

        # Ajustar los tamaños para incluir el resto si la imagen no es divisible exactamente
        tile_h_remainder = height % self.tile_grid_size
        tile_w_remainder = width % self.tile_grid_size

        # Crear lista para almacenar los tiles procesados
        tiles = []

        for i in range(self.tile_grid_size):
            row_tiles = []
            y_start = i * tile_h
            y_end = y_start + tile_h
            if i == self.tile_grid_size - 1:
                y_end += tile_h_remainder  # Añadir el resto al último tile

            for j in range(self.tile_grid_size):
                x_start = j * tile_w
                x_end = x_start + tile_w
                if j == self.tile_grid_size - 1:
                    x_end += tile_w_remainder  # Añadir el resto al último tile

                tile = img[y_start:y_end, x_start:x_end]
                tile_equalized = self.equalize_histogram(tile)
                row_tiles.append(tile_equalized)
            tiles.append(row_tiles)

        # Interpolar los tiles procesados para suavizar las transiciones
        result = interpolate_tiles(tiles, height, width)

        return result

    def equalize_histogram(self, tile: np.ndarray) -> np.ndarray:
        """Ecualiza el histograma de un tile con el límite de recorte."""
        # Calcular el histograma del tile
        hist, bins = np.histogram(tile.flatten(), 256, [0, 256])

        # Aplicar el límite de recorte
        hist_clipped = self.apply_clip_limit(hist, tile.size)

        # Calcular la función acumulativa del histograma ajustado
        cdf = hist_clipped.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype('uint8')

        # Mapear los valores originales a los valores ecualizados
        return cdf_normalized[tile]

    def apply_clip_limit(self, hist: np.ndarray, tile_size: int) -> np.ndarray:
        """Aplica el límite de recorte al histograma y redistribuye los píxeles excedentes."""
        max_clip_value = self.clip_limit * tile_size / 256
        excess_pixels = np.clip(hist - max_clip_value, 0, None).sum()

        # Recortar el histograma
        hist_clipped = np.clip(hist, 0, max_clip_value)

        # Redistribuir los píxeles excedentes
        redistribution = excess_pixels / 256
        hist_clipped += redistribution

        return hist_clipped

class ColorCLAHE:
    def __init__(self, clip_limit: float, tile_grid_size: int):
        """Inicializa la clase ColorCLAHE con los parámetros del algoritmo CLAHE."""
        self.clahe = CLAHE(clip_limit, tile_grid_size)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Aplica el algoritmo CLAHE a una imagen en color."""
        # Convertir la imagen a espacio de color LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Aplicar CLAHE al canal L
        l_clahe = self.clahe.apply(l)

        # Combinar los canales L, A y B y convertir la imagen de nuevo a BGR
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return img_clahe

def interpolate_tiles(tiles: list, height: int, width: int) -> np.ndarray:
    """Interpolar las transiciones entre los tiles usando interpolación bicúbica."""
    # Crear la imagen resultante
    result = np.zeros((height, width), dtype=np.uint8)

    # Calcular las posiciones de los tiles
    tile_grid_size_y = len(tiles)
    tile_grid_size_x = len(tiles[0])
    grid_y = [0]
    grid_x = [0]

    for i in range(tile_grid_size_y):
        grid_y.append(grid_y[-1] + tiles[i][0].shape[0])
    for j in range(tile_grid_size_x):
        grid_x.append(grid_x[-1] + tiles[0][j].shape[1])

    # Ubicar los tiles en la imagen resultante
    for i in range(tile_grid_size_y):
        for j in range(tile_grid_size_x):
            y_start, y_end = grid_y[i], grid_y[i + 1]
            x_start, x_end = grid_x[j], grid_x[j + 1]
            result[y_start:y_end, x_start:x_end] = tiles[i][j]

    # Crear mallas para la interpolación
    x_coords = np.linspace(0, width, width)
    y_coords = np.linspace(0, height, height)

    # Crear la función de interpolación bicúbica
    spline = RectBivariateSpline(y_coords, x_coords, result, kx=3, ky=3)

    # Interpolar toda la imagen
    xx, yy = np.meshgrid(x_coords, y_coords)
    interpolated_image = spline.ev(yy, xx).astype(np.uint8)

    return interpolated_image

def main():
    

    try:
        # Parámetros por defecto
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
            img = read_image(img_path, color=True)
            processed_img = color_clahe.apply(img)
            
            image_name = os.path.basename(img_path)
            print(f'Procesando imagen: {image_name}')
            img_info = (image_name, clip_limit, tile_grid_size)            
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
