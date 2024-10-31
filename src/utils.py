import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import glob
import os

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

def load_images_from_folder(folder, extensions=['.jpg', '.jpeg', '.png']):
    images = []
    image_names = []
    for ext in extensions:
        files = glob.glob(os.path.join(folder, '*' + ext))
        for file in files:
            # Read the image in color mode
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is not None:
                # Convert to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Resize to 512x512
                image = cv2.resize(image, (512, 512))
                images.append(image)
                image_names.append(os.path.basename(file))
    return images, image_names
