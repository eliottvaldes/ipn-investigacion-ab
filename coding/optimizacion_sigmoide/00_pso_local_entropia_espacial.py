import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Constantes para PSO
NUM_PARTICLES = 250
NUM_DIMENSIONS = 2
VARIABLE_RANGES = np.array([[1, 10], [0, 1]])  # Rangos para alpha y delta
COGNITIVE_PARAMETER = 1.6322
SOCIAL_PARAMETER = 0.14
INERTIA_WEIGHT = 0.6
NUM_GENERATIONS = 30

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

def apply_sigmoid(img, alpha, delta):
    """Aplicar la función sigmoide modificada a una imagen en color."""
    output_img = 1 / (1 + np.exp(alpha * (delta - img)))
    # Reescalar la imagen al rango [0, 1]
    output_img = cv2.normalize(output_img, None, 0, 1, cv2.NORM_MINMAX)
    return output_img

def calculate_spatial_entropy(img):
    """Calcular la entropía espacial de una imagen en color."""
    # Inicializar la magnitud del gradiente
    grad_mags = np.zeros(img.shape[:2])

    # Calcular gradientes para cada canal y sumar las magnitudes
    for i in range(3):  # Para cada canal de color
        channel = img[:, :, i]
        # Calcular gradientes en direcciones x e y
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        # Calcular magnitud del gradiente
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mags += grad_mag  # Sumar las magnitudes de gradiente

    # Normalizar las magnitudes de gradiente al rango [0, 1]
    grad_mags = grad_mags / np.max(grad_mags)

    # Calcular el histograma de las magnitudes de gradiente
    hist_counts, _ = np.histogram(grad_mags.flatten(), bins=256, range=(0, 1), density=False)
    # Calcular probabilidades
    total_pixels = np.sum(hist_counts)
    probabilities = hist_counts / total_pixels
    # Eliminar ceros para evitar log(0)
    probabilities = probabilities[probabilities > 0]
    # Calcular entropía espacial
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
    """Evaluar partículas aplicando la sigmoide y calculando la entropía espacial."""
    entropies = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        alpha, delta = particles[i]
        output_img = apply_sigmoid(img, alpha, delta)
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
    """Algoritmo PSO con enfoque local para optimizar alpha y delta."""
    particles, velocities = initialize_particles()
    evaluations = evaluate_particles(particles, img)
    personal_best_positions = particles.copy()
    personal_best_evaluations = evaluations.copy()

    # Inicializar las mejores posiciones locales
    local_best_positions = personal_best_positions.copy()
    
    # Implementación de early stopping
    last_evaluations = []

    for generation in range(NUM_GENERATIONS):
        print(f'Generación: {generation + 1}/{NUM_GENERATIONS} =>')        
        # Actualizar velocidades y posiciones
        velocities = update_velocity(particles, velocities, personal_best_positions, local_best_positions)
        particles = update_positions(particles, velocities)
        particles = apply_bounds(particles)
        evaluations = evaluate_particles(particles, img)

        # Actualizar mejores personales
        for i in range(NUM_PARTICLES):
            if evaluations[i] > personal_best_evaluations[i]:
                personal_best_evaluations[i] = evaluations[i]
                personal_best_positions[i] = particles[i]

        # Actualizar mejores locales en un vecindario de anillo
        for i in range(NUM_PARTICLES):
            left_neighbor = (i - 1) % NUM_PARTICLES
            right_neighbor = (i + 1) % NUM_PARTICLES

            # Obtener la mejor evaluación entre la partícula y sus vecinos
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
                
        # Implementación de early stopping
        last_evaluations.append(np.max(evaluations))
        if len(last_evaluations) > EARLY_STOPPING + 1:
            last_evaluations.pop(0)
            if len(set(last_evaluations)) == 1:
                print('***Se detiene la ejecución debido a que los últimos 5 valores de la evaluación son iguales\n')
                break
        
        # Mostrar mejor evaluación de la generación    
        print(f'\tMejor entropía global: {np.max(personal_best_evaluations):.6f} \tMejor entropía local: {np.max(evaluations):.6f}')
        # Mostrar alpha y delta de la mejor partícula
        print(f'\tAlpha: {personal_best_positions[np.argmax(personal_best_evaluations)][0]:.6f} \tDelta: {personal_best_positions[np.argmax(personal_best_evaluations)][1]:.6f}')
        

    # Encontrar el mejor global
    best_index = np.argmax(personal_best_evaluations)
    best_global_position = personal_best_positions[best_index]
    best_global_evaluation = personal_best_evaluations[best_index]

    return best_global_position, best_global_evaluation


# Funcion auxiliar para guardar las imagenes resultantes
def save_subplot(ax, filename):
    """Guardar un subplot en la carpeta 'results'."""
    # Asegurar que la carpeta 'results' existe, sino crearla
    if not os.path.exists('results'):
        os.makedirs('results')
    fig = ax.figure
    # Dibujar la figura para que el renderizador esté definido
    fig.canvas.draw()
    # Obtener el área delimitada del subplot
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    # Guardar solo el subplot en la carpeta 'results'
    fig.savefig(os.path.join('results', filename), bbox_inches=bbox)


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def main(image_path):
    # Leer la imagen de entrada
    input_image = read_image(image_path)

    # Calcular la entropía inicial de la imagen
    initial_entropy = calculate_spatial_entropy(input_image)
    print(f'Entropía Espacial Inicial: {initial_entropy:.6f}\n')

    # Ejecutar PSO para optimizar alpha y delta
    best_global_position, best_global_evaluation = local_pso(input_image)
    optimized_alpha, optimized_delta = best_global_position
    print(f'\nAlpha óptimo: {optimized_alpha:.6f}')
    print(f'Delta óptimo: {optimized_delta:.6f}')
    print(f'Entropía Espacial Optimizada: {best_global_evaluation:.6f}')

    # Aplicar la función sigmoide con los parámetros optimizados
    output_image = apply_sigmoid(input_image, optimized_alpha, optimized_delta)

    # Calcular la entropía de la imagen resultante
    output_entropy = calculate_spatial_entropy(output_image)
    print(f'Entropía Espacial de la Imagen Procesada: {output_entropy:.6f}')
    
    # Mostrar la diferencia entre las entropías inicial y optimizada
    entropy_difference = output_entropy - initial_entropy
    if entropy_difference > 0:
        print(f'Resultó en un aumento de {entropy_difference:.6f} en la entropía espacial.')
    else:
        print(f'Resultó en un empeoramiento de {entropy_difference:.6f} en la entropía espacial.')

    # Mostrar las imágenes original y procesada en color
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(input_image)
    ax1.set_title(f'Imagen Original\nEntropía Espacial: {initial_entropy:.6f}')
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(output_image)
    ax2.set_title(f'Imagen Procesada\nEntropía Espacial: {output_entropy:.6f} \nAlpha: {optimized_alpha:.6f} \nDelta: {optimized_delta:.6f}')
    ax2.axis('off')

    plt.tight_layout()

    # Guardar los subplots en la carpeta 'results'
    save_subplot(ax1, f'posl_ent_esp_ori_{os.path.basename(image_path)}')
    save_subplot(ax2, f'posl_ent_esp_opt_{os.path.basename(image_path)}')

    plt.show()

if __name__ == '__main__':
    main_path = '../files/'
    file_folder = 'random/'    
    file_name = '02_pintura_techo_c.png
    main(main_path + file_folder + file_name)