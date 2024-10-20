# utils.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_between_class_variance(histogram, thresholds):
    thresholds = np.sort(thresholds)
    L = len(histogram)
    thresholds = np.concatenate(([0], thresholds, [L - 1]))
    sigma_b = 0
    total_mean = (histogram * np.arange(L)).sum()
    total_prob = histogram.sum()

    for i in range(len(thresholds) - 1):
        start = int(thresholds[i])
        end = int(thresholds[i + 1])
        prob = histogram[start:end + 1].sum()
        if prob == 0:
            continue
        mean = (histogram[start:end + 1] * np.arange(start, end + 1)).sum() / prob
        sigma_b += prob * ((mean - total_mean) ** 2)

    return sigma_b

def evaluate_segmentation(original, segmented):
    psnr = peak_signal_noise_ratio(original, segmented, data_range=255)
    ssim = structural_similarity(original, segmented, data_range=255)
    return psnr, ssim
