import cv2
import numpy as np


def hamming_distance(descripor1, descripor2):
    return np.count_nonzero(descripor1 != descripor2)

def bruteforce_matcher(descriptors1, descriptors2):
    matches = []
    for i in range(len(descriptors1)):
        min_distance = float('inf')
        min_index = -1
        for j in range(len(descriptors2)):
            distance = hamming_distance(descriptors1[i], descriptors2[j])
            if distance < min_distance:
                min_distance = distance
                min_index = j
        matches.append(cv2.DMatch(i, min_index, min_distance))
        
    return matches
