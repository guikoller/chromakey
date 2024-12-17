#===============================================================================
# Exemplo: Trabalho 5.
#-------------------------------------------------------------------------------
# Aluno: Guilherme Corrêa Koller
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import os
import sys
import cv2
import numpy as np

INPUT_FOLDER = 'img'
OUTPUT_FOLDER = 'output'
BACKGROUND_IMAGE = './background/shrek.png'

def open_all_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            img_path = os.path.join(folder, filename)
            img = open_image(img_path)
            if img is not None:
                images.append((img, filename))
    return images

def open_image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()
    return img

def chromakey(img, background_img):
    img_float = img.astype(np.float32) / 255.0  # Normalizar para [0, 1]
    
    # Calcular o índice de verdice
    green_index = img_float[:, :, 1] - 0.5 * (img_float[:, :, 0] + img_float[:, :, 2])
    
    # Criar uma máscara baseada no índice de verdice
    threshold = 0.1
    mask = (green_index < threshold).astype(np.float32)  # Mask for foreground
    
    # Refinar a máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove pequenos buracos
    mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Suavizar transições
    
    # Remoção de spill verde
    spill_removal = np.copy(img_float)
    spill_removal[:, :, 1] -= green_index * 0.3  # Reduz o verde nas áreas de spill
    spill_removal = np.clip(spill_removal, 0, 1)
    
    # Redimensionar o background para corresponder ao tamanho da imagem de entrada
    background = cv2.resize(background_img, (img.shape[1], img.shape[0]))
    
    # Aplicar blending com alpha baseado na máscara
    mask = mask[:, :, None]  # Expandir dimensões para operação em cores
    fg = spill_removal * mask
    bg = background * (1 - mask)
    result = fg + bg
    
    # Converter para uint8 para exibição
    result = (result * 255).astype(np.uint8)
    
    return result

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    background_img = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    images = open_all_images(INPUT_FOLDER)
    
    for img, filename in images:
        result = chromakey(img, background_img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), result)
        print(f"Processed: {filename}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()