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
       
def green_index(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    green_hue_lower = 30  # ~70 graus
    green_hue_upper = 80  # ~170 graus
    
    h, s, v = cv2.split(hsv)
    
    mask = np.logical_and(h >= green_hue_lower, h <= green_hue_upper)
    
    green_idx = np.zeros_like(h, dtype=float)
    green_idx[mask] = (1 - abs(h[mask] - ((green_hue_lower + green_hue_upper) / 2)) / 
                       ((green_hue_upper - green_hue_lower) / 2))
    
    green_idx = green_idx * (s / 255.0) * (v / 255.0)
        
    return green_idx

def merge(mask, img, background_img):
    result = np.zeros_like(img)
    
    # Convert to HSV to handle green removal
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:  # White: use background
                result[i, j] = background_img[i, j]
            elif mask[i, j] == 0:   # Black: use foreground
                result[i, j] = img[i, j]
            else:  # Remove green and blend
                # Set saturation to 0 for pixels with green components
                s[i, j] = 0
                # Convert back to BGR
                temp = cv2.cvtColor(cv2.merge([h[i:i+1, j:j+1], s[i:i+1, j:j+1], v[i:i+1, j:j+1]]), cv2.COLOR_HSV2BGR)
                # Blend with background
                alpha = mask[i, j] / 255.0
                result[i, j] = (temp[0, 0] * (1 - alpha) + background_img[i, j] * alpha).astype(np.uint8)
    
    return result
    
    
def chromakey(img, background_img):
    background_img = cv2.resize(background_img, (img.shape[1], img.shape[0]))

    mask = green_index(img)    
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # result = merge(mask, img, background_img)
    
    return mask

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    background_img = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR)
    images = open_all_images(INPUT_FOLDER)
    
    for img, filename in images:
        result = chromakey(img, background_img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), result)
        print(f"Processed: {filename}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()