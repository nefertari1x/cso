# Competitive Swarm Optimizer (CSO)

This repository contains an implementation of the Competitive Swarm Optimizer (CSO) algorithm with neighborhood control for large-scale optimization problems. The implementation is based on the paper "A Competitive Swarm Optimizer for Large Scale Optimization".

## Features

- Implementation of CSO with ring topology neighborhood control
- Support for high-dimensional optimization problems (100D, 500D, 1000D)
- CEC2008 benchmark functions for testing
- Convergence tracking and visualization
- Configurable parameters for different problem dimensions
- Multiple independent runs support

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage

```python
from cso import CSO

# Define your objective function
def objective_function(x):
    return sum(x**2)  # Example: Sphere function

# Initialize CSO
optimizer = CSO(
    obj_func=objective_function,
    dim=100,          # Problem dimension
    pop_size=100,     # Population size
    max_iter=1000,    # Maximum iterations
    lb=-100,          # Lower bound
    ub=100,           # Upper bound
    phi=0.0           # Social factor
)

# Run optimization
best_solution, best_fitness, convergence = optimizer.optimize()

# Plot convergence curve
optimizer.plot_convergence()
```

### Running Benchmark Tests

The repository includes a script to run experiments on CEC2008 benchmark functions:

```python
python example.py
```

This will prompt you for:
1. Test function ID (1-7)
2. Problem dimension (100, 500, 1000)
3. Number of runs (recommended: 25)

## Algorithm Parameters

### Population Size
- 100D problems: 100 particles
- 500D problems: 250 particles
- 1000D problems: 500 particles

### Social Factor (phi)
- 100D problems: 0.0
- 500D problems: 0.1
- 1000D problems: 0.15

### Function Evaluations
- Maximum function evaluations: 5000 * dimension
- Each iteration updates half of the population

## CEC2008 Benchmark Functions

1. Shifted Sphere Function
2. Shifted Schwefel's Problem 2.21
3. Shifted Rosenbrock's Function
4. Shifted Rastrigin's Function
5. Shifted Griewank's Function
6. Shifted Ackley's Function
7. FastFractal "DoubleDip" Function

## Neighborhood Control

The implementation uses a ring topology for neighborhood control:
- Each particle is connected to its immediate neighbors
- Neighborhood size is 3 by default
- Particles learn from their local neighborhood
- Helps maintain diversity and prevent premature convergence

## Results

The algorithm will output:
- Best fitness value
- Median fitness value
- Worst fitness value
- Mean fitness value
- Standard deviation
- Convergence curve plot

## Citation

If you use this implementation in your research, please cite the original paper:
```
Cheng, Ran, and Yaochu Jin. "A competitive swarm optimizer for large scale optimization." IEEE transactions on cybernetics 45.2 (2014): 191-204.
``` 
