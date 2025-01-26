import cv2
import numpy as np
import os
import sys


def get_images(dir):
    images = []
    for f in os.listdir(dir):
        if f.endswith('jpg'):
            images.append(cv2.imread(os.path.join(dir, f)))
            
    return images


if __name__ == '__main__':
    # command: python3 main.py <path_to_images>
    
    if len(sys.argv) != 2:
        print('Usage: python3 main.py <path_to_images>')
        sys.exit(1)
    
    images = get_images(sys.argv[1])
    

