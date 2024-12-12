import sys
import numpy as np
import cv2
#===============================================================================

INPUT_IMAGE =  './img/0.bmp'
BACKGROUND_IMG = './background/shrek.png'

#===============================================================================
def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    bcg = cv2.imread (BACKGROUND_IMG, cv2.IMREAD_COLOR)
    
    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255
    
    bcg = bcg.reshape ((bcg.shape [0], bcg.shape [1], bcg.shape [2]))
    bcg = bcg.astype (np.float32) / 255
    
    
    
    
    cv2.imshow("img", img)
    cv2.imshow("bcg", bcg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main ()
#===============================================================================