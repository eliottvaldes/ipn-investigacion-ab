# Franco Calderas Sergio Alberto 5BV1
# Tarea 2 Primera Optimizacion con algoritmos geneticos
import math
import numpy as np
import random
from PIL import Image

# FUNCION PARA CALCULAR LA ENTROPIA DE UNA IMAGEN
def calcularEntropia(imagen):
    histograma, _ = np.histogram(imagen, bins=256, range=(0, 256), density=True)
    histograma = histograma[histograma > 0]  # Ignorar las probabilidades cero
    entropia = -np.sum(histograma * np.log2(histograma))
    return entropia

# FUNCION PARA APLICAR LA SIGMOIDE A LA IMAGEN
def aplicarSigmoide(imagenOriginal, alpha, delta):
    imagenGrises = imagenOriginal.convert("L")  # Convertir a escala de grises
    datos = np.asarray(imagenGrises) / 255.0    # Convertir la imagen a un array y normalizar de 0 a 1

    sigmoide = 1 / (1 + np.exp(-alpha * (datos - delta)))  # Aplicar la funcion sigmoide
    imagen_transformada = np.uint8(sigmoide * 255) # Reescalar los pixeles de 0 a 255
    return imagen_transformada

# FUNCION OBJETIVO PARA CALCULAR LA ENTROPIA DE UNA IMAGEN
def calcularAptitud(imagenOriginal, variables):
    alpha = variables[0]  # Factor de contraste
    delta = variables[1]  # Punto medio de la curva
    imagen_transformada = aplicarSigmoide(imagenOriginal, alpha, delta) # Aplicar la funcion sigmoide a la imagen
    aptitud = calcularEntropia(imagen_transformada) # Calcular la entropia de la imagen transformada
    return - aptitud

def generarCadenaBits(L):
    cadena_bits = ""
    for i in range(L):
        bit = random.choice(["0", "1"]) # Tipo de valores que puede tomar
        cadena_bits += bit
    return cadena_bits  
def pasarReal(cadena_bits, li, ls, L):
    entero = int(cadena_bits, 2)
    real = li + ((entero * (ls - li)) / (2 ** L - 1))
    return real  
    x = variables[0]
    y = variables[1]
    aptitud = 20 + (x**2 - 10*math.cos(2*math.pi*x)) + (y**2 - 10*math.sin(2*math.pi*y)) # Evaluar X & Y en la Funcion Objetivo
    return aptitud
def concatenar(individuos):
    resultado = ''.join(individuos)
    return resultado
def cruzamientoDosPuntos(parte1, parte2, parte3):
    hijo = parte1 + parte2 + parte3
    return hijo

# FUNCION PRINCIPAL DEL ALGORITMO GENETICO
def algoritmoGenetico(poblacion, probaCruzam, probaMutaci, longitudCadenaBits, maxGeneracion, numGeneracion):
    aptitudes = [individuo[2] for individuo in poblacion]
    cadenaMenorAptitud = aptitudes.index(min(aptitudes))
    valoresRealesMenores = poblacion[cadenaMenorAptitud][1]
    minimaAptitud = poblacion[cadenaMenorAptitud][2]
    print("La Generacion Actual es:", numGeneracion - 1, ", Mejor valor actual:", - minimaAptitud)
    #######################################################################################################################################################
    # FRENO DE LA RECURSION
    if maxGeneracion == 0:
        print("\nGeneracion:", numGeneracion - 1) 
        aptitudes = [individuo[2] for individuo in poblacion] # Obtener las aptitudes de la poblacion final
        cadenaMenorAptitud = aptitudes.index(min(aptitudes)) # Obtener el indice del individuo con la menor aptitud
        valoresRealesMenores = poblacion[cadenaMenorAptitud][1] # Obtener los valores reales del mejor individuo
        minimaAptitud = poblacion[cadenaMenorAptitud][2] # Obtener el valor de la aptitud de cadena de bits seleccionada
        print("El individuo con menor aptitud es:", valoresRealesMenores, - minimaAptitud) # Mostrar al individuo con aptitude
        return
    #######################################################################################################################################################
    # SELECCION DE PADRES POR TORNEO 
    padres = [] # Arreglo para guardar a los padres seleccionados en el torneo
    a = random.sample(poblacion, len(poblacion)) # Hacer la primer permutacion
    b = random.sample(poblacion, len(poblacion)) # Hacer la segunda permutacion
    for i in range(len(poblacion)):
        aptitud_a = a[i][2] # Aptitud del primer individuo de la permutacion a
        aptitud_b = b[i][2] # Aptitud del primer individuo de la permutacion b
        
        if aptitud_a < aptitud_b:
            padres.append(a[i][0]) # Agregar la cadena de bits del primer individuo de la permutacion a como padre
        else:
            padres.append(b[i][0]) # Agregar la cadena de bits del primer individuo de la permutacion b como padre
    #######################################################################################################################################################
    # CRUZAMIENTO EN DOS PUNTOS
    hijos = [] # Arreglo para guardar a los hijos generador por los padres 
    randCruza = round(random.uniform(0, 1), 1)  # Generar un numero random para comparar con probaCruzam
    longitud = longitudCadenaBits # (L * 2) - 1  # Longitud total de las cadenas de bits de los individuos
    pto1 = random.randint(0, longitud - 2) # Primer punto de corte
    pto2 = random.randint(0, longitud - 1) # Segundo punto de corte

    # Verificar si pto2 es menor o iguala pto1 y generar un nuevo valor si es necesario
    while pto2 <= pto1:
        pto2 = random.randint(0, longitud - 1)
            
    for i in range(0,len(poblacion),2):
        if randCruza <= probaCruzam:
            hijo1 = cruzamientoDosPuntos(padres[i][:pto1], padres[i + 1][pto1:pto2], padres[i][pto2:]) # Hacer al primer hijo
            hijos.append(hijo1) # Guardar al hijo1 en el arreglo de hijos
            hijo2 = cruzamientoDosPuntos(padres[i + 1][:pto1], padres[i][pto1:pto2], padres[i + 1][pto2:]) # Hacer al segundo hijo
            hijos.append(hijo2) # Guardar al hijo1 en el arreglo de hijos
        else:
            hijo1 = padres[i] # Los padres pasan como hijos a la siguiente generacion
            hijos.append(hijo1) 
            hijo2 = padres[i + 1]
            hijos.append(hijo2)
    #######################################################################################################################################################
    # MUTACION
    for i in range(len(hijos)):
        randMutac = random.uniform(0, 1) # Generar un numero random para comparar con probaMutaci
        if randMutac <= probaMutaci:
            ptoMut = random.randint(0, len(hijos[0]) - 1) # Posicion aleatoria del hijo donde se va a mutar
            if hijos[i][ptoMut] == '1':
                hijos[i] = hijos[i][:ptoMut] + "0" + hijos[i][ptoMut + 1:]
            elif hijos[i][ptoMut] == '0':
                hijos[i] = hijos[i][:ptoMut] + "1" + hijos[i][ptoMut + 1:]
    #######################################################################################################################################################
    # ELITISMO
    aptitudesPoblacion = [individuo[2] for individuo in poblacion]  # Obtener las aptitudes de la poblacion actual
    indiceMenorAptitud = aptitudesPoblacion.index(min(aptitudesPoblacion)) # Obtener el indice del individuo con la menor aptitud
    individuoMenorAptitud = poblacion[indiceMenorAptitud] # Guardar al individuo con la menor aptitud
    #######################################################################################################################################################
    # DECODIFICACION DE LOS NUEVOS INDIVIDUOS Y SUSTITUCION
    nuevaPoblacion = [] # Arreglo para guardar a la nueva generacion
    
    # Obtener las cadenas de bits de los hijos del arreglo de hijos
    for hijo in hijos:
        cadena_bits_hijo = hijo  # Obtener la cadena de bits del hijo
        variableX = cadena_bits_hijo[:L]  # Obtener la primera parte de la cadena de bits
        variableY = cadena_bits_hijo[L:]  # Obtener la segunda parte de la cadena de bits

        # Obtener los valores reales de cada variable
        valorReal1 = round(pasarReal(variableX, li, ls, L), 2)
        valorReal2 = round(pasarReal(variableY, li, ls, L), 2)

        aptitudHijo = round(calcularAptitud(imagenOriginal, [valorReal1, valorReal2]), 2) # Calcular la aptitud de las nuevas cadenas de bits
        nuevaPoblacion.append((cadena_bits_hijo, [valorReal1, valorReal2], aptitudHijo)) # Guardar los datos del nuevo individuo en la nueva poblacion
    
    indiceAleatorio = random.randint(0, len(nuevaPoblacion) - 1)  # Generar un indice aleatorio del individuo de la nuevaPoblacion para sustituir por Elitismo
    
    nuevaPoblacion[indiceAleatorio] = individuoMenorAptitud # Reemplazar el individuo aleatorio por el individuo de menor aptitud del Elitismo
    #######################################################################################################################################################
    # LLAMADA RECURSIVA DE LA FUNCION
    algoritmoGenetico(nuevaPoblacion, probaCruzam, probaMutaci, longitudCadenaBits, maxGeneracion - 1, numGeneracion + 1)
    #######################################################################################################################################################
    
#############################################################################################################################################################
# CARGAR LA IMAGEN ORIGINAL
ruta_imagen = "../files/01_radiografia_mano_c.jpeg"
imagenOriginal = Image.open(ruta_imagen)
print("La Entropia de la Imagen Original es:", round(calcularEntropia(imagenOriginal), 4))

###########################################################################################################################################################
# PARAMETROS DEL ALGORITMO
poblacion = []  # Arreglo para guardar a toda la poblacion de individuos
limites = [(0, 10), (0, 1)]    # Arreglo para guardar los limites ls & li de cada variable
precisiones = [3, 3]   # Arreglo para guardar la precision de cada variable

maxGeneracion = 40
numIndiv = 150

probaCruzam = 0.9
probaMutaci = 0.3

#############################################################################################################################################################
# GENERAR POBLACION INICIAL
for i in range(numIndiv):
    individuo = []  # Arreglo donde se guardara la cada de bits de cada individuo
    valores_reales = [] # Arreglo para guardar el valor real de cada variable
    longitudCadenaBits = 0  # Variable para almacenar la longitud total de la cadena de bits del individuo
    for j in range(2):
        li, ls = limites[j]
        precision = precisiones[j]

        L = int(math.log2((ls - li) * 10**precision) + 0.9) # Calcular el largo de la cadena de bits del individuo por cada variable
        cadena_bits = generarCadenaBits(L) # Generar la cadena de bits para el individuo por cada variable
        individuo.append(cadena_bits) # Guardar temporalmente la cadena de bits de cada variable del individuo
        longitudCadenaBits += L  # Sumar el largo de la cadena de bits de esta variable a la longitud total
                
        valor_real = round(pasarReal(cadena_bits, li, ls, L), 2) # Pasar cada cadena de bits a valor real
        valores_reales.append(valor_real) # Guardar el valore real de las variables

    aptitud = calcularAptitud(imagenOriginal, valores_reales) # Calcular la aptitud para cada individuo con los valores reales de las variables
    individuos = concatenar(individuo) # Concatenar las las variables X & Y del individuo
    poblacion.append((individuos, valores_reales, aptitud)) # Guardar todos lo datos del individuo en la poblacion

#############################################################################################################################################################
# PRIMER LLAMADA A LA FUNCION algoritmoGenetico
numGeneracion = 1
algoritmoGenetico(poblacion, probaCruzam, probaMutaci, longitudCadenaBits, maxGeneracion, numGeneracion)
