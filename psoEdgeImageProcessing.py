import cv2
import numpy as np
import pywt

# Clase para las partículas en PSO
class Particle:
    def __init__(self, subband):
        self.position = np.array([np.random.randint(1, subband.shape[0]-1), 
                                  np.random.randint(1, subband.shape[1]-1)], dtype='float64')
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.best_position = self.position.copy()
        self.best_value = objective_function(self.position, subband)

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand()
        r2 = np.random.rand()

        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self, subband):
        self.position += self.velocity
        self.position = np.clip(self.position, [1, 1], [subband.shape[0]-2, subband.shape[1]-2])

        current_value = objective_function(self.position, subband)
        if current_value > self.best_value:
            self.best_position = self.position.copy()
            self.best_value = current_value

# Función objetivo basada en la magnitud del gradiente
def objective_function(particle_position, subband):
    i, j = particle_position.astype(int)
    gradient_magnitude = subband[i, j]
    
    # Derivadas de píxeles vecinos (clusters locales)
    neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j-1),           (i, j+1),
        (i+1, j-1), (i+1, j), (i+1, j+1)
    ]
    
    local_gradient = 0
    for ni, nj in neighbors:
        if 0 <= ni < subband.shape[0] and 0 <= nj < subband.shape[1]:
            local_gradient += abs(gradient_magnitude - subband[ni, nj])
    
    return local_gradient  # Maximizar la diferencia en el cluster

# Función para aplicar DWT y descomponer la imagen en subbandas
def dwt_decomposition(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH, coeffs

# PSO aplicado en cada subbanda de la imagen
def pso_on_subband(subband, num_particles=15, iterations=50):
    particles = [Particle(subband) for _ in range(num_particles)]
    global_best_position = particles[0].best_position.copy()
    global_best_value = particles[0].best_value

    for i in range(iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, 0.5, 2.0, 2.0)
            particle.update_position(subband)

            if particle.best_value > global_best_value:
                global_best_position = particle.best_position.copy()
                global_best_value = particle.best_value
        print("Iteracion ",i,global_best_value)

    return particles, global_best_position

# Función de umbralización automática
def automatic_thresholding(image):
    # Convertir la imagen de flotante a uint8
    image_scaled = np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image)))
    
    _, thresholded_image = cv2.threshold(image_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

# Función principal para la detección de bordes usando DWT y PSO
def edge_detection_with_dwt_pso(image):
    # Descomposición de la imagen en subbandas
    LL, LH, HL, HH, coeffs = dwt_decomposition(image)

    # Aplicar PSO a cada subbanda
    _, best_LL = pso_on_subband(HH)
    _, best_LH = pso_on_subband(LH)
    _, best_HL = pso_on_subband(HL)
    _, best_HH = pso_on_subband(HH)

    # Crear imágenes con bordes optimizados en las subbandas
    edges_LL = np.zeros_like(LL)
    edges_LH = np.zeros_like(LH)
    edges_HL = np.zeros_like(HL)
    edges_HH = np.zeros_like(HH)

    edges_LL[best_LL[0].astype(int), best_LL[1].astype(int)] = 255
    edges_LH[best_LH[0].astype(int), best_LH[1].astype(int)] = 255
    edges_HL[best_HL[0].astype(int), best_HL[1].astype(int)] = 255
    edges_HH[best_HH[0].astype(int), best_HH[1].astype(int)] = 255

    # Reconstrucción de la imagen con iDWT (Transformada Inversa)
    reconstructed_image = pywt.idwt2((edges_LL, (LH, HL, LH)), 'haar')

    # Umbralización automática para obtener bordes finales
    final_edges = automatic_thresholding(reconstructed_image)

    return final_edges

# Cargar la imagen de entrada en escala de grises
imagen_original= cv2.imread('/home/rodrigovr/workspace/PSO-Sobel-ImageProcessing/Lenna.png', cv2.IMREAD_GRAYSCALE)

# Aplicar la detección de bordes con DWT y PSO
print("DWT CON PSO".center(50,'-'))
imagen_bordes_pso = edge_detection_with_dwt_pso(imagen_original)

#Funcion de obtencion de bordes con Sobel
def funcion_sobel(imagenEngris):
    # Aplicar Sobel en ambas direcciones
    sobelx = cv2.Sobel(imagenEngris, cv2.CV_64F, 1, 0, ksize=3)  # Sobel en X
    sobely = cv2.Sobel(imagenEngris, cv2.CV_64F, 0, 1, ksize=3)  # Sobel en Y

    # Calcular la magnitud del gradiente
    magnitud = np.sqrt(sobelx**2 + sobely**2)

    # Aplicar un umbral para definir los bordes
    _, imagen_bordes = cv2.threshold(magnitud, 150, 255, cv2.THRESH_BINARY)

    return imagen_bordes

imagen_bordes_sobel = funcion_sobel(imagen_original)

# Mostrar las imágenes
cv2.imshow('Imagen Original', imagen_original)
cv2.imshow('Bordes Detectados DWT-PSO', imagen_bordes_pso)
cv2.imshow('Bordes Detectados Con Sobel', imagen_bordes_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen con los bordes detectados
cv2.imwrite('bordes_detectados_dwt_pso.png', imagen_bordes_pso)
cv2.imwrite('bordes_detectados_sobel.png', imagen_bordes_sobel)


#--- Metricas de calidad entre las imagenes ---
def mse(imageA, imageB):
    # Asegúrate de que las imágenes tengan el mismo tamaño
    assert imageA.shape == imageB.shape
    
    # Calcula el error cuadrático medio (MSE)
    err = np.sum((imageA - imageB) ** 2)
    mse_value = err / float(imageA.shape[0] * imageA.shape[1])
    return mse_value

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # Las imágenes son idénticas
    max_pixel = 255.0  # Suponiendo imágenes de 8 bits
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

#Se imprimen calidad de las imagenes
print("\n");
print("METRICAS DE CALIDAD ENTRE LAS IMAGENES".center(50,'-'))
#Metricas de calidad entre PSO y Original
print("PSNR entre DWT-PSO y Original: ", psnr(imagen_bordes_pso, imagen_original))
print("MSE entre DWT-PSO y Original: ", mse(imagen_bordes_pso, imagen_original))
#Metricas de calidad entre Sobel y Original
print("PSNR entre Sobel y Original: ", psnr(imagen_bordes_sobel, imagen_original))
print("MSE entre Sobel y Original: ", mse(imagen_bordes_sobel, imagen_original))