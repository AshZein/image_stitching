import cv2
import numpy as np

def non_maximum_suppression(keypoints, response, radius=3):
    suppressed_keypoints = []
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        if response[y, x] == np.max(response[max(0, y-radius):min(response.shape[0], y+radius+1), max(0, x-radius):min(response.shape[1], x+radius+1)]):
            suppressed_keypoints.append(keypoint)
    return suppressed_keypoints

# FAST algorithm (Features from Accelerated Segment Test)
def is_keypoint(image, x, y, threshold=100):
    circle_offsets = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1),
                      (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)]
    
    center_intensity = image[y, x]
    brighter_count = 0
    darker_count = 0
    
    for offset in circle_offsets:
        offset_x = x + offset[0]
        offset_y = y + offset[1]
        
        # Boundary check
        if offset_x < 0 or offset_x >= image.shape[1] or offset_y < 0 or offset_y >= image.shape[0]:
            continue
        
        neighbour_intensity = image[offset_y, offset_x]
        
        if neighbour_intensity > min(center_intensity + threshold, 255):
            brighter_count += 1
        elif neighbour_intensity < max(center_intensity - threshold, 0):
            darker_count += 1
            
    return brighter_count >= 12 or darker_count >= 12  


def filter_keypoints(keypoints, min_distance=10):
    filtered_keypoints = []
    for i, kp1 in enumerate(keypoints):
        keep = True
        for j, kp2 in enumerate(keypoints):
            if i != j and cv2.norm(kp1.pt, kp2.pt) < min_distance:
                keep = False
                break
        if keep:
            filtered_keypoints.append(kp1)
    return filtered_keypoints


def fast_algorithm(image, threshold=100):
    keypoints = []
    response = np.zeros(image.shape, dtype=np.float32)
    
    for y in range(3, image.shape[0] - 3):
        for x in range(3, image.shape[1] - 3):
            if is_keypoint(image, x, y, threshold):
                response[y, x] = image[y, x]
                keypoints.append(cv2.KeyPoint(x, y, 1))
    
    # Apply non-maximum suppression
    keypoints = non_maximum_suppression(keypoints, response)
    filter_keypoints(keypoints)
    return keypoints



# Orientation Assignment
def compute_orientation(image, keypoint, patch_size=31):
    x, y = keypoint.pt
    x, y = int(x), int(y)
    
    if x - patch_size // 2 < 0 or x + patch_size // 2 >= image.shape[1] or y - patch_size // 2 < 0 or y + patch_size // 2 >= image.shape[0]:
        return 0  # Skip keypoints too close to the border
    
    patch = image[y - patch_size // 2:y + patch_size // 2 + 1, x - patch_size // 2:x + patch_size // 2 + 1]
    
    # Gradient computations
    Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude and orientation
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    orientations = np.arctan2(Iy, Ix)
    
    # Histogram
    hist, bin_edges = np.histogram(orientations, bins=36, range=(-np.pi, np.pi), weights=magnitude)
    
    dominant = bin_edges[np.argmax(hist)]
    
    return dominant

def all_orientations(image, keypoints):
    for keypoint in keypoints:
        orientation = compute_orientation(image, keypoint)
        keypoint.angle = orientation

# BRIEF descriptor
def generate_brief_pattern(patch_size, num_pairs):
    np.random.seed(0)  # For reproducibility
    pattern = np.random.randint(0, patch_size, (num_pairs, 4))
    return pattern

def compute_brief_descriptor(image, keypoint, pattern, patch_size):
    x, y = keypoint.pt
    x, y = int(x), int(y)
    
    if x - patch_size // 2 < 0 or x + patch_size // 2 >= image.shape[1] or y - patch_size // 2 < 0 or y + patch_size // 2 >= image.shape[0]:
        return np.zeros(len(pattern), dtype=np.uint8)  # Skip keypoints too close to the border
    
    # Extract the patch around the keypoint
    patch = image[y - patch_size // 2:y + patch_size // 2 + 1, x - patch_size // 2:x + patch_size // 2 + 1]
    
    descriptor = []
    for p in pattern:
        x1, y1, x2, y2 = p
        if patch[y1, x1] < patch[y2, x2]:
            descriptor.append(1)
        else:
            descriptor.append(0)
    
    return np.array(descriptor, dtype=np.uint8)

def brief_keypoint_descriptors(image, keypoints, patch_size=31, num_pairs=256):
    pattern = generate_brief_pattern(patch_size, num_pairs)
    descriptors = []
    
    for keypoint in keypoints:
        descriptor = compute_brief_descriptor(image, keypoint, pattern, patch_size)
        descriptors.append(descriptor)
    
    return np.array(descriptors)

# Put all together as ORB
def compute_orb(image):
    """Compute keypoints and descriptors of an image using ORB"""
    # FAST Algorithm
    keypoints = fast_algorithm(image)
    
    # Orientation Assignment
    all_orientations(image, keypoints)
    
    # BRIEF descriptor 
    descriptors = brief_keypoint_descriptors(image, keypoints)
    
    return keypoints, descriptors