import cv2
import numpy as np

# Caminhos das imagens
INPUT_IMAGE = './img/2.bmp'
BACKGROUND_IMAGE = './background/shrek.png'

def main():
    # Carregar a imagem
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    img_float = img.astype(np.float32) / 255.0  # Normalizar para [0, 1]
    
    # Calcular o índice de verdice
    green_index = img_float[:, :, 1] - 0.5 * (img_float[:, :, 0] + img_float[:, :, 2])
    
    # Criar uma máscara baseada no índice de verdice
    threshold = 0.1
    mask = (green_index < threshold).astype(np.float32)  # Mask for foreground
    
    # Refinar a máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove pequenos buracos
    mask = cv2.GaussianBlur(mask, (11, 11), 0)  # Suavizar transições
    
    # Remoção de spill verde
    spill_removal = np.copy(img_float)
    spill_removal[:, :, 1] -= green_index * 0.3  # Reduz o verde nas áreas de spill
    spill_removal = np.clip(spill_removal, 0, 1)
    
    # Carregar e redimensionar o background
    background = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    background = cv2.resize(background, (img.shape[1], img.shape[0]))
    
    # Aplicar blending com alpha baseado na máscara
    mask = mask[:, :, None]  # Expandir dimensões para operação em cores
    fg = spill_removal * mask
    bg = background * (1 - mask)
    result = fg + bg
    
    # Converter para uint8 para exibição
    result = (result * 255).astype(np.uint8)
    
    # Exibir resultados
    cv2.imshow("Foreground", (fg * 255).astype(np.uint8))
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
