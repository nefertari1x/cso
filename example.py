import numpy as np
import matplotlib.pyplot as plt
from cso import CSO
from cec2008_benchmarks import CEC2008Functions

def run_experiment(func_id, dim, runs=10):
    """
    Run CSO algorithm experiment
    :param func_id: Test function ID (1-7)
    :param dim: Problem dimension
    :param runs: Number of runs
    """
    print(f"Running function {func_id}, dimension {dim}")
    
    # Initialize CEC2008 functions
    cec_funcs = CEC2008Functions()
    lb, ub = cec_funcs.get_bounds(func_id)
    
    # Create a wrapper function to call the test function
    def obj_function(x):
        return cec_funcs.evaluate(func_id, x)
    
    # Parameter settings based on dimension
    if dim == 100:
        pop_size = 100
        phi = 0.0
    elif dim == 500:
        pop_size = 250
        phi = 0.1
    else:  # 1000D
        pop_size = 500
        phi = 0.15
    
    # Maximum iterations to reach 5000*dim function evaluations
    # In CSO, each iteration updates half the population (pop_size/2)
    # So max_iter = (5000*dim)/(pop_size/2) to get 5000*dim evaluations
    max_iter = int((5000 * dim) / (pop_size / 2))
    
    # Store results
    best_fitness_history = []
    convergence_curves = []
    
    # Run multiple experiments
    for r in range(runs):
        print(f"Run {r+1}/{runs}")
        
        # Initialize CSO
        optimizer = CSO(
            obj_func=obj_function,
            dim=dim,
            pop_size=pop_size,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            phi=phi
        )
        
        # Run optimization
        best_solution, best_fitness, convergence = optimizer.optimize()
        
        # Store results
        best_fitness_history.append(best_fitness)
        convergence_curves.append(convergence)
        
        print(f"Run {r+1} completed, best fitness: {best_fitness:.6e}")
    
    # Calculate statistics
    best_fitness_history.sort()
    
    stats = {
        "best": best_fitness_history[0],
        "worst": best_fitness_history[-1],
        "median": best_fitness_history[runs//2],
        "mean": np.mean(best_fitness_history),
        "std": np.std(best_fitness_history)
    }
    
    # Print statistics
    print("\nStatistics:")
    print(f"Best: {stats['best']:.6e}")
    print(f"Median: {stats['median']:.6e}")
    print(f"Worst: {stats['worst']:.6e}")
    print(f"Mean: {stats['mean']:.6e}")
    print(f"Std Dev: {stats['std']:.6e}")
    
    # Plot convergence curve (using the median run)
    median_idx = best_fitness_history.index(stats['median'])
    median_curve = convergence_curves[median_idx]
    
    plt.figure(figsize=(10, 6))
    if func_id != 7:
        plt.semilogy(range(1, max_iter + 1), median_curve)
    else:
        plt.plot(range(1, max_iter + 1), median_curve)
    
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title(f'Function {func_id}, {dim}D CSO Convergence Curve (Median Run)')
    plt.grid(True)
    plt.savefig(f'f{func_id}_{dim}D_convergence.png')
    plt.show()
    
    return stats

if __name__ == "__main__":
    # Select function and dimension
    func_id = int(input("Enter test function ID (1-7): "))
    dim = int(input("Enter problem dimension (100, 500, 1000): "))
    runs = int(input("Enter number of runs (5-25 recommended): "))
    
    # Run experiment
    stats = run_experiment(func_id, dim, runs)