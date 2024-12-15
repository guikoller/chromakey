import cv2
import numpy as np

# Caminhos das imagens
INPUT_IMAGE = './img/3.bmp'
BACKGROUND_IMAGE = './background/shrek.png'

def main():
    # Carregar a imagem
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converter para HSV
    img_float = img.astype(np.float32) / 255.0  # Normalizar para [0, 1]

    # Definir o intervalo de verde padrão e tolerância
    green_hue = (60, 10)  # Central hue for green and a tolerance
    lower_saturation = 50  # Min saturation to exclude whites/grays
    lower_value = 50       # Min value to exclude dark areas

    # Calcular os limites baseados no intervalo
    lower_green = np.array([green_hue[0] - green_hue[1], lower_saturation, lower_value])
    upper_green = np.array([green_hue[0] + green_hue[1], 255, 255])

    # Criar a máscara para verde
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # Refinar a máscara para evitar invasão
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fechar buracos pequenos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remover ruídos pequenos
    
    # Ajustar as bordas com Canny
    edges = cv2.Canny(mask, 50, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    mask[edges_dilated > 0] = 255

    # Inverter a máscara para capturar o foreground
    mask_inv = cv2.bitwise_not(mask)
    mask_inv_float = mask_inv.astype(np.float32) / 255.0

    # Remoção de spill verde
    spill_removal = np.copy(img_float)
    spill_removal[:, :, 1] -= (spill_removal[:, :, 1] * mask_inv_float) * 0.5  # Reduzir spill
    spill_removal = np.clip(spill_removal, 0, 1)

    # Carregar e redimensionar o background
    background = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    background = cv2.resize(background, (img.shape[1], img.shape[0]))

    # Combinar foreground e background
    fg = spill_removal * mask_inv_float[:, :, None]  # Foreground
    bg = background * (1 - mask_inv_float[:, :, None])  # Background
    result = fg + bg

    # Converter para uint8 para exibição
    result = (result * 255).astype(np.uint8)

    # Exibir resultados
    cv2.imshow("Foreground", (fg * 255).astype(np.uint8))
    cv2.imshow("Result", result)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
