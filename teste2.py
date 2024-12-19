import cv2
import numpy as np

def chromakey(frame, background_img):
    img_float = frame.astype(np.float32) / 255.0  # Normalizar para [0, 1]
    
    # Calcular o índice de verdice
    green_index = img_float[:, :, 1] - 0.5 * (img_float[:, :, 0] + img_float[:, :, 2])
    
    # Criar uma máscara baseada no índice de verdice com valores entre 0 e 1
    threshold = 0.5
    mask = np.clip((green_index - threshold) / (1 - threshold), 0, 1)
    
    # Refinar a máscara usando operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Remove pequenos buracos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove ruídos
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Suavizar transições
    
    # Redimensionar o background para corresponder ao tamanho da imagem de entrada
    background = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))
    
    # Usar a máscara para combinar o fundo com o foreground, preservando as sombras
    result = img_float * (1 - mask[:, :, None]) + background * mask[:, :, None]
    
    # Converter para uint8 para exibição
    result = (result * 255).astype(np.uint8)
    
    return result

def main():
    fundo = cv2.imread('./background/white.png')
    frame = cv2.imread('./img/1.bmp')
    frame = cv2.flip(frame, 1)
    fundo = fundo[0:frame.shape[0], 0:frame.shape[1]]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([30, 100, 100])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = mask.astype(np.uint8)

    fundo = cv2.bitwise_and(fundo, fundo, mask=mask)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.medianBlur(res, 5)

    norm = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 50, 255, 1)[1]
    norm = np.invert(norm)
    norm = cv2.dilate(norm, None, iterations=1)

    edged = cv2.erode(norm, None, iterations=1)
    res2 = cv2.bitwise_xor(frame, cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))
    res2 = cv2.bitwise_or(frame, res2)

    final1 = cv2.hconcat([frame, hsv])
    final2 = cv2.hconcat([fundo, fundo + res2])
    final3 = cv2.vconcat([final1, final2])

    cv2.imshow('Resultado', final3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()