import numpy as np


def ackley(x):
    d = len(x)  # number of dimensions
    term1 = -0.2 * np.sqrt(0.5 * np.sum(x ** 2))
    term2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(term1) - np.exp(term2 / d) + 20 + np.e


# Define the Genetic Algorithm
def genetic_algorithm_ackley(pop_size=400, num_generations=1000, crossover_prob=0.8, mutation_prob=0.14, elitism=True):
    lower_bound = -35
    upper_bound = 35
    num_genes = 5
    gene_range = upper_bound - lower_bound

    # Initialize the population
    population = np.random.rand(pop_size, num_genes) * gene_range + lower_bound
    # Main loop for generations
    for gen in range(num_generations):
        # Evaluate the fitness of each individual
        fitness = np.array([ackley(x) for x in population])

        # Select parents based on fitness (tournament selection)
        parents = np.zeros_like(population)
        for i in range(pop_size):
            # Randomly select two individuals
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if fitness[idx1] < fitness[idx2]:
                parents[i] = population[idx1]
            else:
                parents[i] = population[idx2]

        # Perform crossover
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < crossover_prob:
                crossover_point = np.random.randint(1, num_genes)
                parents[i, crossover_point:] = parents[i + 1, crossover_point:]
                parents[i + 1, crossover_point:] = parents[i, crossover_point:]

        # Perform mutation
        for i in range(pop_size):
            for j in range(num_genes):
                if np.random.rand() < mutation_prob:
                    parents[i, j] += np.random.normal(0, 0.1)

        # Clip the individuals to the feasible search space
        parents = np.clip(parents, lower_bound, upper_bound)

        # Evaluate the fitness of the mutated individuals
        fitness_parents = np.array([ackley(x) for x in parents])

        # Replace the old population with the mutated individuals
        if elitism:
            # Keep the best individual from the previous generation
            best_idx = np.argmin(fitness)
            new_population = np.zeros_like(population)
            new_population[0] = population[best_idx]
            new_population[1:] = parents[1:]
            fitness[0] = fitness[best_idx]
            fitness[1:] = fitness_parents[1:]
            population = new_population
        else:
            population = parents
            fitness = fitness_parents

        # Print the best objective value in this generation
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_obj_val = fitness[best_idx]
        print("Generation: {}, Best Objective Value: {}".format(gen + 1, best_obj_val))

    return best_solution, best_obj_val


if __name__ == "__main__":
    # Test the Genetic Algorithm
    best_solution, best_obj_val = genetic_algorithm_ackley()
    print("Best solution:", best_solution)
    print("Best objective value:", best_obj_val)
