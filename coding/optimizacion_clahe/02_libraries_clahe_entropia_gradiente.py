import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Constantes para PSO
NUM_PARTICLES = 100
NUM_DIMENSIONS = 2
VARIABLE_RANGES = np.array([[1, 10], [1, 50]])  # Rangos para clip_limit y tile_grid_size
COGNITIVE_PARAMETER = 1.6322
SOCIAL_PARAMETER = 0.14
INERTIA_WEIGHT = 0.6
NUM_GENERATIONS = 300

MAX_HEIGHT = 500  # Altura máxima de la imagen para evitar altos tiempos de ejecución
EARLY_STOPPING = 5

# =============================================================================
# DEFINICIÓN DE FUNCIONES
# =============================================================================
def read_image(path):
    """Leer la imagen en color y normalizarla."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"La imagen en la ruta '{path}' no se encontró.")
    # Obtener el tamaño de la imagen y redimensionar si es necesario
    alto, ancho = img.shape[:2]
    print(f'Medidas de imagen:\tAlto: {alto}, Ancho: {ancho}')
    if alto > MAX_HEIGHT:
        proporcion = MAX_HEIGHT / alto
        img = cv2.resize(img, (int(ancho * proporcion), MAX_HEIGHT))
        alto, ancho = img.shape[:2]
        print(f'Medidas de imagen ajustadas:\t Alto: {alto}, Ancho: {ancho}')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    img = img.astype(np.float64) / 255.0  # Normalizar la imagen a [0, 1]
    return img

def apply_clahe(img, clip_limit, tile_grid_size):
    """Aplicar CLAHE (histograma adaptativo) a una imagen en color."""
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])  # Aplicar CLAHE en el canal de luminancia
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_clahe.astype(np.float64) / 255.0  # Normalizar la imagen a [0, 1]

def calculate_spatial_entropy(img):
    """Calcular la entropía espacial de una imagen en color."""
    grad_mags = np.zeros(img.shape[:2])

    for i in range(3):  # Para cada canal de color
        channel = img[:, :, i]
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mags += grad_mag  # Sumar las magnitudes de gradiente

    grad_mags = grad_mags / np.max(grad_mags)
    hist_counts, _ = np.histogram(grad_mags.flatten(), bins=256, range=(0, 1), density=False)
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    probabilities = probabilities[probabilities > 0]
    spatial_entropy = -np.sum(probabilities * np.log2(probabilities))
    return spatial_entropy

def initialize_particles():
    """Inicializar partículas y velocidades dentro de los rangos dados."""
    lower_bounds, upper_bounds = VARIABLE_RANGES[:, 0], VARIABLE_RANGES[:, 1]
    ranges = upper_bounds - lower_bounds
    particles = np.random.uniform(lower_bounds, upper_bounds, (NUM_PARTICLES, NUM_DIMENSIONS))
    velocities = np.random.uniform(-ranges / 2, ranges / 2, (NUM_PARTICLES, NUM_DIMENSIONS))
    return particles, velocities

def evaluate_particles(particles, img):
    """Evaluar partículas aplicando CLAHE y calculando la entropía espacial."""
    entropies = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        clip_limit, tile_grid_size = particles[i]
        output_img = apply_clahe(img, clip_limit, tile_grid_size)
        entropy = calculate_spatial_entropy(output_img)
        entropies[i] = entropy
    return entropies

def update_velocity(particles, velocities, personal_best_positions, local_best_positions):
    """Actualizar las velocidades basadas en las mejores posiciones locales."""
    rp = np.random.uniform(size=(NUM_PARTICLES, NUM_DIMENSIONS))
    rg = np.random.uniform(size=(NUM_PARTICLES, NUM_DIMENSIONS))
    cognitive = COGNITIVE_PARAMETER * rp * (personal_best_positions - particles)
    social = SOCIAL_PARAMETER * rg * (local_best_positions - particles)
    inertia = INERTIA_WEIGHT * velocities
    return inertia + cognitive + social

def update_positions(particles, velocities):
    """Actualizar posiciones basadas en las velocidades."""
    return particles + velocities

def apply_bounds(particles):
    """Asegurar que las partículas se mantengan dentro de los límites definidos."""
    lower_bounds, upper_bounds = VARIABLE_RANGES[:, 0], VARIABLE_RANGES[:, 1]
    return np.clip(particles, lower_bounds, upper_bounds)

def local_pso(img):
    """Algoritmo PSO con enfoque local para optimizar clip_limit y tile_grid_size."""
    particles, velocities = initialize_particles()
    evaluations = evaluate_particles(particles, img)
    personal_best_positions = particles.copy()
    personal_best_evaluations = evaluations.copy()

    local_best_positions = personal_best_positions.copy()
    last_evaluations = []

    for generation in range(NUM_GENERATIONS):
        print(f'Generación: {generation + 1}/{NUM_GENERATIONS} =>')        
        velocities = update_velocity(particles, velocities, personal_best_positions, local_best_positions)
        particles = update_positions(particles, velocities)
        particles = apply_bounds(particles)
        evaluations = evaluate_particles(particles, img)

        for i in range(NUM_PARTICLES):
            if evaluations[i] > personal_best_evaluations[i]:
                personal_best_evaluations[i] = evaluations[i]
                personal_best_positions[i] = particles[i]

        for i in range(NUM_PARTICLES):
            left_neighbor = (i - 1) % NUM_PARTICLES
            right_neighbor = (i + 1) % NUM_PARTICLES

            neighborhood_evaluations = [
                personal_best_evaluations[left_neighbor],
                personal_best_evaluations[i],
                personal_best_evaluations[right_neighbor]
            ]
            max_eval = max(neighborhood_evaluations)
            if max_eval == personal_best_evaluations[left_neighbor]:
                local_best_positions[i] = personal_best_positions[left_neighbor]
            elif max_eval == personal_best_evaluations[i]:
                local_best_positions[i] = personal_best_positions[i]
            else:
                local_best_positions[i] = personal_best_positions[right_neighbor]
                
        last_evaluations.append(np.max(evaluations))
        if len(last_evaluations) > EARLY_STOPPING + 1:
            last_evaluations.pop(0)
            if len(set(last_evaluations)) == 1:
                print('***Se detiene la ejecución debido a que los últimos 5 valores de la evaluación son iguales\n')
                break

        print(f'\tMejor entropía global: {np.max(personal_best_evaluations):.6f} \tMejor entropía local: {np.max(evaluations):.6f}')
        print(f'\tClip Limit: {personal_best_positions[np.argmax(personal_best_evaluations)][0]:.6f} \tTile Grid Size: {personal_best_positions[np.argmax(personal_best_evaluations)][1]:.6f}')
        
    best_index = np.argmax(personal_best_evaluations)
    best_global_position = personal_best_positions[best_index]
    best_global_evaluation = personal_best_evaluations[best_index]

    return best_global_position, best_global_evaluation


def save_subplot(ax, filename):
    """Guardar un subplot en la carpeta 'results'."""
    if not os.path.exists('results'):
        os.makedirs('results')
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join('results', filename), bbox_inches=bbox)

def main(image_path):
    input_image = read_image(image_path)

    initial_entropy = calculate_spatial_entropy(input_image)
    print(f'Entropía Espacial Inicial: {initial_entropy:.6f}\n')

    best_global_position, best_global_evaluation = local_pso(input_image)
    optimized_clip_limit, optimized_tile_grid_size = best_global_position
    print(f'\nClip Limit óptimo: {optimized_clip_limit:.6f}')
    print(f'Tile Grid Size óptimo: {optimized_tile_grid_size:.6f}')
    print(f'Entropía Espacial Optimizada: {best_global_evaluation:.6f}')

    output_image = apply_clahe(input_image, optimized_clip_limit, optimized_tile_grid_size)

    output_entropy = calculate_spatial_entropy(output_image)
    print(f'Entropía Espacial de la Imagen Procesada: {output_entropy:.6f}')
    
    entropy_difference = output_entropy - initial_entropy
    if entropy_difference > 0:
        print(f'Resultó en un aumento de {entropy_difference:.6f} en la entropía espacial.')
    else:
        print(f'Resultó en un empeoramiento de {entropy_difference:.6f} en la entropía espacial.')

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(input_image)
    ax1.set_title(f'Imagen Original\nEntropía Espacial: {initial_entropy:.6f}')
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(output_image)
    ax2.set_title(f'Imagen Procesada ({output_image.shape[0]} x  {output_image.shape[1]})\nEntropía Espacial: {output_entropy:.6f} \nClip Limit: {optimized_clip_limit:.6f} \nTile Grid Size: {optimized_tile_grid_size:.6f}')                  
    ax2.axis('off')

    plt.tight_layout()

    # Guardar la figura en la carpeta 'results'
    base_filename = os.path.basename(image_path)
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    output_filename = os.path.join(results_folder, f'clahe-libs_entropia-gradiente_{base_filename}')
    plt.savefig(output_filename)
    print(f'Imagen guardada en: {output_filename}')
    
    

if __name__ == '__main__':
    main_path = '../files/'
    file_folder = 'bones/'    
    
    # obtener todas las imagenes dentro de main_filder + file_folder y por cada carpeta ejecutar main
    for filename in os.listdir(main_path + file_folder):
        if filename.endswith('.png'):
            print(f'Procesando imagen: {filename}')
            main(main_path + file_folder + filename)
            print('\n')
