import random

# Define the cost functions


def J1(x1, x2):
    return (x1 ** 2 + x2 ** 2 - x1 * x2 - 10) * 1e-3


def J2(x1, x2):
    return x1 ** 2 + (x2 - 2) ** 2 - 100


# Define the Genetic Algorithm parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
ELITE_SIZE = 10
NUM_GENERATIONS = 50

# Define the selection function


def selection(population):
    sorted_population = sorted(population, key=lambda x: x[2])
    elite = sorted_population[:ELITE_SIZE]
    selection_pool = [random.choice(elite)
                      for i in range(len(population) - ELITE_SIZE)]
    return selection_pool




# Define the crossover function


def crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

# Define the mutation function


def mutation(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += random.gauss(0, 1)
    return individual


# Initialize the population
population = []
for i in range(POPULATION_SIZE):
    x1 = random.uniform(-10, 10)
    x2 = random.uniform(-10, 10)
    population.append([x1, x2, None])

# Run the Genetic Algorithm
for generation in range(NUM_GENERATIONS):
    # Evaluate the fitness of each individual
    for i in range(len(population)):
        x1, x2, _ = population[i]
        fitness1 = J1(x1, x2)
        fitness2 = J2(x1, x2)
        population[i][2] = fitness1 + fitness2

    # Select the parents for the next generation
    selection_pool = selection(population)

    # Create the next generation
    new_population = []
    for i in range(POPULATION_SIZE - ELITE_SIZE):
        parent1 = random.choice(selection_pool)
        parent2 = random.choice(selection_pool)
        child = crossover(parent1, parent2)
        child = mutation(child)
        new_population.append(child)

    # Preserve the elite individuals
    population = sorted(population, key=lambda x: x[2])
    elite = population[:ELITE_SIZE]
    population = elite + new_population

    # Output the best individual in each generation
    best_individual = min(population, key=lambda x: x[2])
    print(
        f'Generation {generation}: {best_individual[0]}, {best_individual[1]}, cost={best_individual[2]}')

# Output the final best individual
best_individual = min(population, key=lambda x: x[2])
print(
    f'Final solution: {best_individual[0]}, {best_individual[1]}, cost={best_individual[2]}')
