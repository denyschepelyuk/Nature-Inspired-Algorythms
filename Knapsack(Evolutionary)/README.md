# Knapsack Problem - Evolutionary Algorithm

This repository contains an implementation of an **Evolutionary Algorithm** to solve the **0/1 Knapsack Problem**. The algorithm evolves a population of potential solutions over several generations, using selection, crossover, and mutation operations to maximize the value of items in the knapsack without exceeding its weight capacity.

## How It Works
The algorithm starts with a all-zero population of binary individuals (bitstrings) representing item selections. Over multiple generations, the fittest individuals are selected and modified to evolve better solutions. The evolution process includes:
- **Fitness Evaluation**: Individuals are scored based on the total value of selected items while respecting the weight constraint.
- **Tournament Selection**: The best solution among randomly chosen candidates is selected.
- **Uniform Crossover**: Two parents exchange genes to form new offspring.
- **Mutation**: Random changes are introduced to promote diversity.
- **Repair Mechanism**: Infeasible solutions (overweight knapsacks) are adjusted by randomly removing items until they become valid.

## Hyperparameters
The hyperparameters define the behavior of the evolutionary process and are dynamically adjusted based on the problem size. Here is a breakdown of the key hyperparameters and why they are set as they are:

- **Population Size (`population_size`)**:
  - Initially set to `bits * 10` (10 times the number of items).
  - Reduced for larger problems to balance computation time.
  - For `bits >= 100`, set to `bits * 2`.
  - For `bits >= 500`, set to `bits * 0.7`.
  - For `bits >= 1000`, set to `bits * 0.5`.
  - **Rationale**: Larger populations provide diversity, but excessive sizes slow down execution. The scaling balances these aspects.

- **Number of Generations (`generations`)**:
  - Initially set to `bits`.
  - For `bits >= 100`, reduced to `bits * 0.5`.
  - For `bits >= 500`, reduced to `bits * 0.3`.
  - For `bits >= 1000`, reduced to `bits * 0.2`.
  - **Rationale**: Larger problem sizes require fewer generations per item, as selection pressure and crossover quickly refine good solutions.

- **Tournament Size (`tournament_size`)**:
  - Set to `6`.
  - **Rationale**: Balances selective pressure; higher values lead to premature convergence, while lower values reduce selection efficiency.

- **Crossover Rate (`crossover_rate`)**:
  - Set to `0.9` (90% chance of crossover).
  - **Rationale**: Encourages frequent recombination of good traits while still allowing some individuals to pass on genes unchanged.

- **Mutation Rate (`mutation_rate`)**:
  - Set to `0.6`, modified by `mutation_const = 1 / bits`.
  - **Rationale**: Ensures that smaller problems mutate less aggressively, while larger problems have a better chance of exploring the search space.

## Functions

### **Data Handling**
- `load_knapsack_data(filename)`: Loads the problem instance from a file.

### **Evolutionary Algorithm**
- `initialize_population(size, items, W)`: Creates a all-zero initial population.
- `fitness(individual, items, W)`: Evaluates how good a solution is.
- `evaluate_population(population, items, W)`: Ranks individuals and calculates best/average fitness.
- `generate_new_population(population, items, W, params)`: Produces a new generation using selection, crossover, and mutation.

### **Genetic Operators**
- `tournament_selection(population, items, W, k)`: Selects the best individual from a random subset.
- `uniform_crossover(parent1, parent2, crossover_rate)`: Produces two offspring by randomly mixing genes.
- `mutate(individual, mutation_rate)`: Randomly flips bits with a given probability.
- `repair_individual(ind, items, capacity)`: Ensures individuals do not exceed knapsack capacity.

### **Visualization & Output**
- `plot(title, best_fitness, avg_fitness, params)`: Plots the best and average fitness over generations.
- `line_fw_print(strs, widths, flush=True, char='|')`: Prints formatted progress updates.
- `run_evolution(params, test_data)`: Main function to run the algorithm and print progress.

## Running the Code

To run the algorithm on a test file:
```sh
python3 solution.py tests/100.txt
```

Example Output:
```
Generation | Best Fit | Avg. Fit | Total Time (seconds)
10/50:       7126       4934       0.142          
20/50:       9147       7179       0.311          
30/50:       9147       7612       0.484          
40/50:       9147       7637       0.656          
50/50:       9147       7423       0.829          
```

A plot will be generated after all computations, showing the progression of fitness scores over generations.
