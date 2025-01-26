import cv2
import numpy as np
import os
import random
import sys


def get_images(dir) -> list:
    images = []
    for f in os.listdir(dir):
        if f.endswith('jpg'):
            images.append(cv2.imread(os.path.join(dir, f)))
            
    return images


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_all_to_gray(images: list): 
    return [convert_to_gray(image) for image in images]


def stitch_images(images: list):
    return images[random.randint(0, len(images)-1)]


if __name__ == '__main__':
    # command: python3 main.py <path_to_images>
    
    # incorrect command line args
    if len(sys.argv) != 2:
        print('Usage: python3 main.py <path_to_images>')
        sys.exit(1)
    
    dir_path = sys.argv[1]
    
    # check if the directory is valid
    if not os.path.isdir(dir_path):
        print(f'Error: {dir_path} is not a valid directory')
        sys.exit(1)
    
    images = get_images(dir_path)
    
    # check if images are found in the directory
    if images == []:
        print(f'No images found in the directory {dir_path}')
        sys.exit(1)
    
    

