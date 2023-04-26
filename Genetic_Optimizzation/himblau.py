import numpy as np
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def genetic_algorithm_himmelblau(pop_size=100, num_generations=100, crossover_prob=0.8, mutation_prob=0.1,
                                 elitism=True):
    lower_bound = -5
    upper_bound = 5
    num_genes = 2
    gene_range = upper_bound - lower_bound
    population = np.random.rand(pop_size, num_genes) * gene_range + lower_bound

    for gen in range(num_generations):
        fitness = np.array([himmelblau(x[0], x[1]) for x in population])
        parents = np.zeros_like(population)
        for i in range(pop_size):
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if fitness[idx1] < fitness[idx2]:
                parents[i] = population[idx1]
            else:
                parents[i] = population[idx2]
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < crossover_prob:
                crossover_point = np.random.randint(1, num_genes)
                parents[i, crossover_point:] = parents[i + 1, crossover_point:]
                parents[i + 1, crossover_point:] = parents[i, crossover_point:]

        for i in range(pop_size):
            for j in range(num_genes):
                if np.random.rand() < mutation_prob:
                    parents[i, j] += np.random.normal(0, 1)

        parents = np.clip(parents, lower_bound, upper_bound)
        fitness_parents = np.array([himmelblau(x[0], x[1]) for x in parents])

        if elitism:
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

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_obj_val = fitness[best_idx]
        print("Generation: {}, Best Objective Value: {}".format(gen + 1, best_obj_val))

    return best_solution, best_obj_val


if __name__ == "__main__":
    best_solution, best_obj_val = genetic_algorithm_himmelblau()
    print("Best solution:", best_solution)
    print("Best objective value:", best_obj_val)
