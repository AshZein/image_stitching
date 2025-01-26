import cv2
import numpy as np

# FAST algorithm (Features from accelerated segment test)
def is_keypoint(image, x, y, threshold=100):
    circle_offsets = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1),
                      (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)]
    
    center_intensity = image[y, x]
    brighter_count = 0
    darker_count = 0
    
    for offset in circle_offsets:
        offset_x = x + offset[0]
        offset_y = y + offset[1]
        
        # boundary check
        if offset_x < 0 or offset_x >= image.shape[1] or offset_y < 0 or offset_y >= image.shape[0]:
            continue
        
        neighbour_intensity = image[offset_y, offset_x]
        
        if neighbour_intensity > center_intensity + threshold:
            brighter_count += 1
        elif neighbour_intensity < center_intensity - threshold:
            darker_count += 1
            
    return brighter_count >= 12 or darker_count >= 12  

def fast_algorithm(image, threshold=100):
    keypoints = []
    
    for y in range(3, image.shape[0] - 3):
        for x in range(3, image.shape[1] - 3):
            if is_keypoint(image, x, y, threshold):
                keypoints.append((x, y))
                
    return  keypoints


# Orientation Assignment
def compute_orientation(image, keypoint, patch_size=31):
    x, y = keypoint
    
    patch = image[y - patch_size//2:y + patch_size//2 + 1, x - patch_size//2:x + patch_size//2 + 1]
    
    # gradient computations
    Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    
    # gradient magnitude and orientation
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    orientations = np.arctan2(Iy, Ix)
    
    #histogram
    hist, bin_edges = np.histogram(orientations, bins=36, range=(-np.pi, np.pi), weights=magnitude)
    
    dominant = bin_edges[np.argmax(hist)]
    
    return dominant