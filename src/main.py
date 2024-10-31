import cv2
import numpy as np
from bfa import BeeForagingAlgorithm
from utils import evaluate_segmentation, load_images_from_folder
import os
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import logging
import seaborn as sns
import time

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Program started")

    # Load all images from the folder
    image_folder = '../data/images/'
    images, image_names = load_images_from_folder(image_folder)
    if not images:
        print(f'Error: No images found in {image_folder}')
        return

    # Define parameters for grid search
    num_thresholds_list = [2, 3, 4]
    population_sizes = [20, 30, 40]
    max_generation_options = [30, 50, 70]
    nsc_options = [0.5, 0.7, 0.9]
    limit_stagnations = [5, 10, 15]
    adaptive_nsc_options = [True, False]

    results_dir = '../data/results/'
    os.makedirs(results_dir, exist_ok=True)

    # Results DataFrame
    results_df = pd.DataFrame(columns=['Image', 'Num_Thresholds', 'Params', 'Fitness', 'PSNR', 'SSIM', 'Time'])

    convergence_data = []

    # --- Grid Search Phase ---
    print("Starting Grid Search...")
    for population_size, max_generations, nsc, limit_stagnation, adaptive_nsc in product(
            population_sizes, max_generation_options, nsc_options, limit_stagnations, adaptive_nsc_options):

        params = {
            'population_size': population_size,
            'max_generations': max_generations,
            'nsc': nsc,
            'limit_stagnation': limit_stagnation,
            'adaptive_nsc': adaptive_nsc
        }
        print(f"\nTesting with parameters: {params}")

        for num_thresholds in num_thresholds_list:
            for image, image_name in zip(images, image_names):
                print(f'\nProcessing {image_name} with {num_thresholds} thresholds')

                bfa = BeeForagingAlgorithm(
                    image=image,
                    num_thresholds=num_thresholds,
                    population_size=population_size,
                    max_generations=max_generations,
                    nsc=nsc,
                    limit_stagnation=limit_stagnation,
                    adaptive_nsc=adaptive_nsc
                )

                optimal_thresholds, best_fitness, best_fitness_history, avg_fitness_history, thresholds_history, total_time = bfa.optimize()
                segmented_image = bfa.apply_thresholds(optimal_thresholds)

                psnr, ssim = evaluate_segmentation(image, segmented_image)
                new_data = pd.DataFrame({
                    'Image': [image_name],
                    'Num_Thresholds': [num_thresholds],
                    'Params': [str(params)],
                    'Fitness': [best_fitness],
                    'PSNR': [psnr],
                    'SSIM': [ssim],
                    'Time': [total_time]
                })
                results_df = pd.concat([results_df, new_data], ignore_index=True)

                # Save convergence data
                convergence_data.append({
                    'image_name': image_name,
                    'num_thresholds': num_thresholds,
                    'params': params,
                    'best_fitness_history': best_fitness_history,
                    'avg_fitness_history': avg_fitness_history,
                    'thresholds_history': thresholds_history
                })

    # Save grid search results to CSV
    results_df.to_csv(f'{results_dir}results_summary.csv', index=False)
    print('\nGrid Search complete. Results saved to results_summary.csv')

    # Generate convergence plots
    for image_name in set([d['image_name'] for d in convergence_data]):
        for num_thresholds in num_thresholds_list:
            plt.figure()
            for data in convergence_data:
                if data['image_name'] == image_name and data['num_thresholds'] == num_thresholds:
                    label = f"PS:{data['params']['population_size']}, G:{data['params']['max_generations']}, nsc:{data['params']['nsc']}, LS:{data['params']['limit_stagnation']}, ANSC:{data['params']['adaptive_nsc']}"
                    plt.plot(data['best_fitness_history'], label=label)
            plt.title(f'Convergence Curves - {image_name} - {num_thresholds} Thresholds')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{results_dir}{image_name[:-4]}_{num_thresholds}_convergence_comparison.png')
            plt.close()

    # --- Identify Optimal Parameters and Re-run BFA ---
    print("\nIdentifying Optimal Parameters and Re-running BFA...")
    # Read the results_summary.csv
    results_df = pd.read_csv(f'{results_dir}results_summary.csv')

    # Create a DataFrame to store final results
    final_results_df = pd.DataFrame(columns=['Image', 'Num_Thresholds', 'Optimal_Params', 'Fitness', 'PSNR', 'SSIM', 'Time'])

    for image, image_name in zip(images, image_names):
        for num_thresholds in num_thresholds_list:
            # Filter results for the current image and num_thresholds
            subset = results_df[(results_df['Image'] == image_name) & (results_df['Num_Thresholds'] == num_thresholds)]
            if subset.empty:
                continue

            # Find the row with the best Fitness
            best_row = subset.loc[subset['Fitness'].idxmax()]
            optimal_params_str = best_row['Params']
            optimal_params = eval(optimal_params_str)  # Convert string back to dictionary

            print(f"\nOptimal parameters for {image_name} with {num_thresholds} thresholds:")
            print(optimal_params)

            # Re-run BFA with optimal parameters
            bfa = BeeForagingAlgorithm(
                image=image,
                num_thresholds=num_thresholds,
                population_size=optimal_params['population_size'],
                max_generations=optimal_params['max_generations'],
                nsc=optimal_params['nsc'],
                limit_stagnation=optimal_params['limit_stagnation'],
                adaptive_nsc=optimal_params['adaptive_nsc']
            )

            optimal_thresholds, best_fitness, best_fitness_history, avg_fitness_history, thresholds_history, total_time = bfa.optimize()
            segmented_image = bfa.apply_thresholds(optimal_thresholds)
            result_image_path = f'{results_dir}{image_name[:-4]}_{num_thresholds}t_optimal.png'
            cv2.imwrite(result_image_path, segmented_image)
            print(f'Segmented image saved to {result_image_path}')

            # Evaluate the segmentation quality
            psnr, ssim = evaluate_segmentation(image, segmented_image)
            print(f'Final PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

            # Save final results
            final_data = pd.DataFrame({
                'Image': [image_name],
                'Num_Thresholds': [num_thresholds],
                'Optimal_Params': [str(optimal_params)],
                'Fitness': [best_fitness],
                'PSNR': [psnr],
                'SSIM': [ssim],
                'Time': [total_time]
            })
            final_results_df = pd.concat([final_results_df, final_data], ignore_index=True)

            # Plot convergence curve
            plt.figure()
            plt.plot(best_fitness_history)
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.title(f'Convergence Curve - {image_name} ({num_thresholds} thresholds) - Optimal Params')
            convergence_plot_path = f'{results_dir}{image_name[:-4]}_{num_thresholds}t_convergence_optimal.png'
            plt.savefig(convergence_plot_path)
            plt.close()

            # Plot threshold evolution
            thresholds_history = np.array(thresholds_history)
            plt.figure()
            for i in range(thresholds_history.shape[1]):
                plt.plot(thresholds_history[:, i], label=f'Threshold {i+1}')
            plt.xlabel('Generation')
            plt.ylabel('Threshold Value')
            plt.title(f'Threshold Evolution - {image_name} ({num_thresholds} thresholds)')
            plt.legend()
            threshold_plot_path = f'{results_dir}{image_name[:-4]}_{num_thresholds}t_thresholds_optimal.png'
            plt.savefig(threshold_plot_path)
            plt.close()

            # Display and save original and segmented images
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(segmented_image, cmap='gray')
            axs[1].set_title(f'Segmented Image - {num_thresholds} Thresholds')
            axs[1].axis('off')

            plt.suptitle(f'{image_name} Segmentation')
            plt.tight_layout()
            plt.savefig(f'{results_dir}{image_name[:-4]}_{num_thresholds}t_segmentation.png')
            plt.close()

    # Save final results to CSV
    final_results_df.to_csv(f'{results_dir}final_results.csv', index=False)
    print('\nAll processing complete. Final results saved to final_results.csv')

    # Perform statistical analysis and generate additional plots
    # Convert Params column to separate columns
    params_df = pd.DataFrame(results_df['Params'].apply(eval).tolist())
    full_df = pd.concat([results_df.drop('Params', axis=1), params_df], axis=1)

    # Compute correlation matrix
    corr = full_df[['population_size', 'max_generations', 'nsc', 'limit_stagnation', 'adaptive_nsc', 'Fitness', 'PSNR', 'SSIM']].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{results_dir}correlation_matrix.png')
    plt.close()

    # Generate scatter plots for parameter effects
    parameters = ['population_size', 'max_generations', 'nsc', 'limit_stagnation', 'adaptive_nsc']
    metrics = ['Fitness', 'PSNR', 'SSIM']

    for param in parameters:
        for metric in metrics:
            plt.figure()
            for num_thresholds in num_thresholds_list:
                subset = full_df[full_df['Num_Thresholds'] == num_thresholds]
                plt.scatter(subset[param], subset[metric], label=f'{num_thresholds} Thresholds')
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.title(f'Effect of {param} on {metric}')
            plt.legend()
            plt.savefig(f'{results_dir}effect_{param}_{metric}.png')
            plt.close()

    # Group by parameters and compute statistics
    grouped = full_df.groupby(['population_size', 'max_generations', 'nsc', 'limit_stagnation', 'adaptive_nsc'])[['Fitness', 'PSNR', 'SSIM']].agg(['mean', 'std']).reset_index()

    # Save to CSV
    grouped.to_csv(f'{results_dir}parameter_statistics.csv', index=False)

    # Boxplots of Fitness values for different nsc values
    plt.figure()
    sns.boxplot(x='nsc', y='Fitness', data=full_df)
    plt.title('Fitness Distribution for Different nsc Values')
    plt.savefig(f'{results_dir}fitness_boxplot_nsc.png')
    plt.close()

    logging.info("Program completed")

if __name__ == '__main__':
    main()
