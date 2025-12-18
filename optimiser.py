import numpy as np
import matplotlib.pyplot as plt
import random
import os


def load_data(filename):
    filepath = os.path.join('data', filename)
    
    with open(filepath, 'r') as f:
        raw_data = f.read().split()
        data = [int(x) for x in raw_data]

    n = data[0]
    matrix_data = data[1:]
    matrix_size = n * n

    if len(matrix_data) == matrix_size * 2:
        print(f"Found 2 matrices in {filename}.")
        cost_flat = matrix_data[0 : matrix_size]
        time_flat = matrix_data[matrix_size : matrix_size * 2]
        

    elif len(matrix_data) == matrix_size:
        print(f"Found 1 matrix in {filename}. Generating random Time matrix.")
        cost_flat = matrix_data
        # Generate random time values between 10 and 100
        time_flat = [random.randint(10, 100) for _ in range(matrix_size)]
        
    else:
        raise ValueError(f"File data length {len(matrix_data)} doesn't match N={n}")

    cost_matrix = np.array(cost_flat).reshape(n, n)
    time_matrix = np.array(time_flat).reshape(n, n)

    return cost_matrix, time_matrix


filename = 'assign700.txt'
cost_matrix, time_matrix = load_data(filename)


def init_population(pop_size, n):

    population = []
    for _ in range(pop_size):
        # Create a list [0, 1, 2, ..., n-1]
        solution = list(range(n))
        # Shuffle it randomly
        random.shuffle(solution)
        population.append(solution)
    return population

def evaluate(solution, cost_matrix, time_matrix):

    total_cost = 0
    total_time = 0
    n = len(solution)
    
    for i in range(n):
        task = solution[i]
        total_cost += cost_matrix[i][task]
        total_time += time_matrix[i][task]
        
    return total_cost, total_time

def crossover_pmx(parent1, parent2):
    n = len(parent1)
    
    cut1, cut2 = sorted(random.sample(range(n), 2))
    
    offspring = [-1] * n
    

    offspring[cut1:cut2+1] = parent1[cut1:cut2+1]
    
    for i in range(n):
        if i < cut1 or i > cut2:
            candidate = parent2[i]
            while candidate in offspring:
                index_in_p1 = offspring.index(candidate)
                candidate = parent2[index_in_p1]
            offspring[i] = candidate
            
    return offspring

def mutate_swap(solution, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution


def is_dominated(sol_a_scores, sol_b_scores):
    cost_a, time_a = sol_a_scores
    cost_b, time_b = sol_b_scores
    return (cost_a >= cost_b and time_a >= time_b) and (cost_a > cost_b or time_a > time_b)

def run_optimization():
    filename = 'assign100.txt'
    cost_matrix, time_matrix = load_data(filename)
    n = cost_matrix.shape[0]
    
    POP_SIZE = 100
    GENERATIONS = 100
    
    population = init_population(POP_SIZE, n)
    print(f"Starting optimization for {GENERATIONS} generations...")
    
    for gen in range(GENERATIONS):
        scores = [evaluate(p, cost_matrix, time_matrix) for p in population]
        non_dominated = []
        for i in range(POP_SIZE):
            dominated = False
            for j in range(POP_SIZE):
                if i != j and is_dominated(scores[i], scores[j]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(population[i])
        
        if len(non_dominated) < POP_SIZE // 2:
            remainder = POP_SIZE - len(non_dominated)
            non_dominated.extend(init_population(remainder, n))
            
        new_population = []
        while len(new_population) < POP_SIZE:
            p1 = random.choice(non_dominated)
            p2 = random.choice(non_dominated)
            child = crossover_pmx(p1, p2)
            
            child = mutate_swap(child, mutation_rate=0.2)
            
            new_population.append(child)
            
        population = new_population
        
        if gen % 10 == 0:
            print(f"Generation {gen}: Best Cost so far = {min(s[0] for s in scores)}")
    final_scores = [evaluate(p, cost_matrix, time_matrix) for p in population]
    costs = [s[0] for s in final_scores]
    times = [s[1] for s in final_scores]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(costs, times, c='blue', label='Solutions')
    plt.xlabel('Total Cost')
    plt.ylabel('Total Time')
    plt.title('Optimization Results: Cost vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    run_optimization()