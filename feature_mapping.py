import cv2
import numpy as np


def hamming_distance(descripor1, descripor2):
    return np.count_nonzero(descripor1 != descripor2)

def bruteforce_matcher(descriptors1, descriptors2):
    matches = []
    for i in range(len(descriptors1)):
        distances = []
        for j in range(len(descriptors2)):
            distance = hamming_distance(descriptors1[i], descriptors2[j])
            distances.append((distance, j))
        distances.sort(key=lambda x: x[0])
        matches.append((cv2.DMatch(i, distances[0][1], distances[0][0]), cv2.DMatch(i, distances[1][1], distances[1][0])))
    return matches


def ratio_test(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches