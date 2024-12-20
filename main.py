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
    
    green_hue_lower = 25
    green_hue_upper = 80
    
    h, s, v = cv2.split(hsv)
    
    mask = np.logical_and(h >= green_hue_lower, h <= green_hue_upper)
    
    # calcula índice de verde baseado na distância do centro da faixa verde
    green_idx = np.zeros_like(h, dtype=float)
    green_idx[mask] = (1 - abs(h[mask] - ((green_hue_lower + green_hue_upper) / 2)) / 
                       ((green_hue_upper - green_hue_lower) / 2))
    
    # ajusta índice baseado na saturação e valor
    green_idx = green_idx * (s / 255.0) * (v / 255.0)
    
    # desloca histograma e normaliza
    green_idx = green_idx + 0.2
    green_idx = np.clip(green_idx, 0, 1)
    green_idx = cv2.normalize(green_idx, None, 0, 1, cv2.NORM_MINMAX)
        
    return green_idx

def merge(mask, img, background_img):
    result = np.zeros_like(img, dtype=np.float32)
    
    mask_255 = (mask * 255).astype(np.uint8)
    
    img_float = img.astype(np.float32) / 255.0
    bg_float = background_img.astype(np.float32) / 255.0
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for i in range(mask_255.shape[0]):
        for j in range(mask_255.shape[1]):
            if mask_255[i, j] == 0: # foreground se máscara for preta
                result[i, j] = img_float[i, j]
            elif mask_255[i, j] == 255:  #background se máscara for branca
                result[i, j] = bg_float[i, j]
            else:
                h, s, v = img_hsv[i, j]
                
                # reduz saturação em áreas esverdeadas
                if 35 <= h <= 85:  
                    s = s * 0.2
                
                pixel = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
                pixel_float = pixel.astype(np.float32) / 255.0
                
                #mistura com background usando alpha
                alpha = mask_255[i, j] / 255.0
                result[i, j] = pixel_float * (1 - alpha) + bg_float[i, j] * alpha
    
    return result

def chromakey(img, background_img):
    background_img = cv2.resize(background_img, (img.shape[1], img.shape[0]))
    mask = green_index(img)
    result = merge(mask, img, background_img)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

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