# main.py

import cv2
import numpy as np
from bfa import BeeForagingAlgorithm
from utils import evaluate_segmentation
import os

def main():
    # Load the image
    image_path = '../data/images/lena.png'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Error: Image not found at {image_path}')
        return

    # Define parameters
    num_thresholds = 3  # Number of thresholds for MLT
    population_size = 30
    max_generations = 50
    nsc = 0.7  # Neighborhood shrinking coefficient

    # Initialize the BFA
    bfa = BeeForagingAlgorithm(
        image=image,
        num_thresholds=num_thresholds,
        population_size=population_size,
        max_generations=max_generations,
        nsc=nsc
    )

    # Run the optimization
    optimal_thresholds, best_fitness = bfa.optimize()

    print(f'Optimal Thresholds: {optimal_thresholds}')
    print(f'Best Fitness: {best_fitness}')

    # Apply the thresholds to segment the image
    segmented_image = bfa.apply_thresholds(optimal_thresholds)

    # Create results directory if it doesn't exist
    os.makedirs('data/results/', exist_ok=True)

    # Save the segmented image
    result_image_path = f'data/results/segmented_lena_{num_thresholds}t.png'
    cv2.imwrite(result_image_path, segmented_image)
    print(f'Segmented image saved to {result_image_path}')

    # Evaluate the segmentation quality
    psnr, ssim = evaluate_segmentation(image, segmented_image)
    print(f'PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

if __name__ == '__main__':
    main()
