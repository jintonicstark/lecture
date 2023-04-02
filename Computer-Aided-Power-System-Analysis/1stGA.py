import matplotlib.pyplot as plt
import random

# Initialize for make graph
fitness_scores_history = []
population_history = []
generation_numbers = []

# Step 1: Encoding of population
# Define the size of the population and the number of genes in each solution
POPULATION_SIZE = 100
GENE_SIZE = 2


# Generate an initial population of random candidate solutions
population = []
for i in range(POPULATION_SIZE):
    solution = [random.uniform(-10, 10) for j in range(GENE_SIZE)]
    population.append(solution)


# Step 2: Selection of fitness function
# Define the fitness function for the problem
def fitness(solution):
    # Calculate the fitness value of the solution
    x1, x2 = solution
    j1 = (x1**2 + x2**2 - x1*x2 - 10) * 1e-3
    j2 = x1**2 + (x2-2)**2 - 100
    return -(j1 + j2)  # minimize the objective function


# Step 3: Evaluation of fitness function
# Evaluate the fitness of each candidate solution in the population
fitness_scores = []
for solution in population:
    score = fitness(solution)
    fitness_scores.append(score)

# Step 4: Parents Selection
# Select a set of parent solutions from the population


def select_parents(population, fitness_scores):
    # Use tournament selection to select two parents with high fitness scores
    indices = random.sample(range(len(population)), 5)
    indices_fitness = [(index, fitness_scores[index]) for index in indices]
    indices_fitness.sort(key=lambda x: x[1], reverse=True)
    return population[indices_fitness[0][0]], population[indices_fitness[1][0]]

def roulette_wheel_selection(population, fitness_scores):
    # Use roulette wheel selection to select two parents with high fitness scores
    total_fitness = sum(fitness_scores)
    r = random.uniform(0, total_fitness)
    index = 0
    partial_sum = fitness_scores[index]
    while partial_sum < r:
        index += 1
        partial_sum += fitness_scores[index]
    parent1 = population[index]

    r = random.uniform(0, total_fitness)
    index = 0
    partial_sum = fitness_scores[index]
    while partial_sum < r:
        index += 1
        partial_sum += fitness_scores[index]
    parent2 = population[index]

    return parent1, parent2

# Step 5: Cross over
# Define crossover process with crossover rate


def uniform_crossover(parent1, parent2):
    new_solution = [0] * GENE_SIZE
    for i in range(GENE_SIZE):
        if random.random() < CROSSOVER_RATE:
            new_solution[i] = parent1[i]
        else:
            new_solution[i] = parent2[i]
    return new_solution


def random_crossover(parent1, parent2):
    gene_size = len(parent1)
    new_solution = [0] * gene_size
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, gene_size - 1)
        new_solution[:crossover_point] = parent1[:crossover_point]
        new_solution[crossover_point:] = parent2[crossover_point:]
    else:
        new_solution = parent1
    return new_solution


# Repeat the process for a predefined number of generations
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.1
MUTATION_RANGE = 1
CROSSOVER_RATE = 0.6
for i in range(NUM_GENERATIONS):
    # Parents Selection
    parent1, parent2 = roulette_wheel_selection(population, fitness_scores)

    # Step 5:Crossover process

    # Crossover process with crossover rate
    new_solution = [0] * GENE_SIZE
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, GENE_SIZE - 1)
        new_solution[:crossover_point] = parent1[:crossover_point]
        new_solution[crossover_point:] = parent2[crossover_point:]
    else:
        new_solution = parent1

    # Crossover process
    new_solution = uniform_crossover(parent1, parent2)

    # Step 6: Mutatio Process
    # Mutation Process
    for j in range(GENE_SIZE):
        if random.random() < MUTATION_RATE:
            new_solution[j] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
            new_solution[j] = min(max(new_solution[j], -10), 10)

    # Add the new solution to the population
    population.append(new_solution)

    # Evaluate the fitness of each candidate solution in the population
    fitness_scores.append(fitness(new_solution))

    # Remove the worst solution from the population
    worst_index = fitness_scores.index(min(fitness_scores))
    del population[worst_index]
    del fitness_scores[worst_index]

    # 
    fitness_scores_history.append(max(fitness_scores))
    population_history.append(new_solution)
    generation_numbers.append(i)

# Evaluate the fitness of each candidate solution in the final population

for solution in population:
    score = fitness(solution)
    fitness_scores.append(score)

# Return the best solution found
best_index = fitness_scores.index(max(fitness_scores))
best_solution = population[best_index]
best_fitness = fitness_scores[best_index]

print("Best solution found: ", best_solution)
print("Best fitness score: ", best_fitness)




# Plot the fitness scores against the generation numbers
plt.plot(generation_numbers, fitness_scores_history)
plt.xlabel('Generation')
plt.ylabel('Fitness score')
plt.title('Fitness scores over generations')
plt.show()

# Plot the population against the generation numbers
plt.plot(generation_numbers, population_history)
plt.xlabel('Generation')
plt.ylabel('population')
plt.title('population over generations')
plt.show()