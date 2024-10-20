# bfa.py

import numpy as np
from utils import calculate_between_class_variance
import random

class BeeForagingAlgorithm:
    def __init__(self, image, num_thresholds, population_size, max_generations, nsc=0.7):
        self.image = image
        self.num_thresholds = num_thresholds
        self.population_size = population_size
        self.max_generations = max_generations
        self.nsc = nsc  # Neighborhood shrinking coefficient
        self.histogram = self.calculate_histogram()
        self.L = 256  # Assuming 8-bit grayscale images

        # Parameters based on the paper
        self.num_scouts = int(0.1 * population_size)
        self.num_recruits = population_size - self.num_scouts
        self.num_foragers = int(0.5 * self.num_recruits)
        self.num_onlookers = self.num_recruits - self.num_foragers

        # Neighborhood size initialization
        self.neighborhood_size = (self.L - 1) / 10  # Initial neighborhood size
        self.neighborhood_sizes = []  # To track neighborhood sizes per food source

        # Stagnation limit
        self.limit_stagnation = int(2 * np.log(0.1) / np.log(self.nsc))

    def calculate_histogram(self):
        hist, _ = np.histogram(self.image.flatten(), bins=256, range=(0, 256))
        return hist / hist.sum()

    def initialize_scouts(self):
        # Initialize scout bees (random food sources)
        scouts = []
        for _ in range(self.num_scouts):
            thresholds = np.sort(
                np.random.choice(range(1, self.L - 1), self.num_thresholds, replace=False)
            )
            scouts.append(thresholds)
        return np.array(scouts)

    def evaluate_fitness(self, population):
        fitness_values = np.array([
            calculate_between_class_variance(self.histogram, thresholds)
            for thresholds in population
        ])
        return fitness_values

    def select_food_sources(self, scouts, fitness_values):
        # Select top K food sources
        K = self.num_foragers  # Number of selected food sources
        indices = np.argsort(fitness_values)[-K:]
        selected_sources = scouts[indices]
        selected_fitness = fitness_values[indices]
        # Initialize stagnation counters and neighborhood sizes
        self.stagnation_counters = [0] * K
        self.neighborhood_sizes = [self.neighborhood_size] * K
        return selected_sources, selected_fitness

    def forager_search(self, selected_sources):
        new_sources = []
        for idx, source in enumerate(selected_sources):
            neighborhood = self.neighborhood_sizes[idx]
            # Multidimensional search
            new_source = source + np.random.uniform(-neighborhood, neighborhood, size=source.shape)
            new_source = np.clip(new_source, 1, self.L - 2)
            new_source = np.sort(new_source.astype(int))
            new_sources.append(new_source)
        return np.array(new_sources)

    def onlooker_search(self, selected_sources, selected_fitness):
        probabilities = selected_fitness / selected_fitness.sum()
        # Ensure at least one onlooker per food source
        assignments = []
        for idx in range(len(selected_sources)):
            num_onlookers = max(1, int(probabilities[idx] * self.num_onlookers))
            assignments.extend([idx] * num_onlookers)
        # Adjust to total number of onlookers
        if len(assignments) > self.num_onlookers:
            assignments = assignments[:self.num_onlookers]
        elif len(assignments) < self.num_onlookers:
            assignments.extend(random.choices(range(len(selected_sources)), k=self.num_onlookers - len(assignments)))

        # Single-dimensional search
        new_sources = []
        for idx in assignments:
            source = selected_sources[idx].copy()
            dim = np.random.randint(0, len(source))
            direction = np.random.choice([-1, 1])
            source[dim] += direction * self.neighborhood_sizes[idx]
            source = np.clip(source, 1, self.L - 2)
            source = np.sort(source.astype(int))
            new_sources.append((idx, source))
        return new_sources

    def global_search(self, num_new_scouts):
        new_scouts = []
        for _ in range(num_new_scouts):
            thresholds = np.sort(
                np.random.choice(range(1, self.L - 1), self.num_thresholds, replace=False)
            )
            new_scouts.append(thresholds)
        return np.array(new_scouts)

    def optimize(self):
    # Initialization Phase
        scouts = self.initialize_scouts()
        fitness_values = self.evaluate_fitness(scouts)
        selected_sources, selected_fitness = self.select_food_sources(scouts, fitness_values)

        best_fitness = np.max(selected_fitness)
        best_thresholds = selected_sources[np.argmax(selected_fitness)]

    # Main Loop
        for gen in range(self.max_generations):
            # Local Search Phase
            # Forager Bees
            forager_sources = self.forager_search(selected_sources)
            forager_fitness = self.evaluate_fitness(forager_sources)

            # Update food sources and stagnation counters
            for idx in range(len(selected_sources)):
                if forager_fitness[idx] > selected_fitness[idx]:
                    selected_sources[idx] = forager_sources[idx]
                    selected_fitness[idx] = forager_fitness[idx]
                    self.stagnation_counters[idx] = 0  # Reset stagnation counter
                    # Update best solution
                    if selected_fitness[idx] > best_fitness:
                        best_fitness = selected_fitness[idx]
                        best_thresholds = selected_sources[idx]
                else:
                    self.stagnation_counters[idx] += 1
                    # Apply neighborhood shrinking
                    self.neighborhood_sizes[idx] *= self.nsc

            # Onlooker Bees
            onlooker_assignments = self.onlooker_search(selected_sources, selected_fitness)
            for (idx, source) in onlooker_assignments:
                fitness = calculate_between_class_variance(self.histogram, source)
                if fitness > selected_fitness[idx]:
                    selected_sources[idx] = source
                    selected_fitness[idx] = fitness
                    self.stagnation_counters[idx] = 0  # Reset stagnation counter
                    # Update best solution
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_thresholds = source
                else:
                    self.stagnation_counters[idx] += 1

            # Global Search Phase
            num_new_scouts = self.num_scouts - len(selected_sources)
            new_scouts = self.global_search(num_new_scouts)
            new_fitness = self.evaluate_fitness(new_scouts)

            # Combine all sources ensuring they are 2D arrays
            all_sources = np.vstack([selected_sources, new_scouts.reshape(-1, selected_sources.shape[1])])
            all_fitness = np.concatenate((selected_fitness, new_fitness))

            # Select top K sources for next iteration
            indices = np.argsort(all_fitness)[-self.num_foragers:]
            selected_sources = all_sources[indices]
            selected_fitness = all_fitness[indices]
            # Adjust stagnation counters and neighborhood sizes
            self.stagnation_counters = [self.stagnation_counters[i % len(self.stagnation_counters)] for i in indices]
            self.neighborhood_sizes = [self.neighborhood_sizes[i % len(self.neighborhood_sizes)] for i in indices]

            # Update best solution
            current_best_fitness = np.max(selected_fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_thresholds = selected_sources[np.argmax(selected_fitness)]

            print(f'Generation {gen+1}/{self.max_generations}, Best Fitness: {best_fitness:.4f}')

        return best_thresholds, best_fitness


    def apply_thresholds(self, thresholds):
        thresholds = np.sort(thresholds)
        segmented = np.zeros_like(self.image)
        levels = np.linspace(0, 255, self.num_thresholds + 1, endpoint=True)

        thresholds = np.concatenate(([0], thresholds, [255]))
        for i in range(len(thresholds) - 1):
            mask = (self.image >= thresholds[i]) & (self.image < thresholds[i + 1])
            segmented[mask] = levels[i]

        segmented = segmented.astype(np.uint8)
        return segmented
