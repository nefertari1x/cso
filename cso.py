import numpy as np
import matplotlib.pyplot as plt


class CSO:
    """
    Competitive Swarm Optimizer with neighborhood control (ring topology)
    Based on: "A Competitive Swarm Optimizer for Large Scale Optimization"
    """
    def __init__(self, obj_func, dim, pop_size=None, max_iter=None, lb=None, ub=None, phi=0.0):
        """
        Initialize the CSO algorithm
        :param obj_func: Objective function
        :param dim: Problem dimension
        :param pop_size: Population size
        :param max_iter: Maximum iterations
        :param lb: Lower bound
        :param ub: Upper bound
        :param phi: Social factor
        """
        self.obj_func = obj_func
        self.dim = dim
        self.phi = phi
        
        # Set population size and iterations
        self.pop_size = pop_size if pop_size is not None else max(100, dim // 10)
        if self.pop_size % 2 != 0:  # Ensure population size is even for pairwise competition
            self.pop_size += 1
            
        self.max_iter = max_iter if max_iter is not None else 1000
        
        # Set search bounds
        self.lb = -100 if lb is None else lb
        self.ub = 100 if ub is None else ub
        
        # Initialize population
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.V = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        
        # Performance tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = np.zeros(self.max_iter)
        
        # Initialize fitness values
        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.X[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.X[i].copy()
    
    def _get_neighborhood_mean(self, idx, neighborhood_size=3):
        """Get the mean position in a neighborhood, using ring topology"""
        indices = []
        for i in range(-(neighborhood_size//2), neighborhood_size//2 + 1):
            neighbor_idx = (idx + i) % self.pop_size
            indices.append(neighbor_idx)
        return np.mean(self.X[indices], axis=0)
    
    def optimize(self):
        """Run the CSO algorithm"""
        for t in range(self.max_iter):
            # Randomly shuffle population for pairing
            indices = np.random.permutation(self.pop_size)
            
            # Pairwise competition
            for i in range(0, self.pop_size, 2):
                # Get two competitors
                idx1, idx2 = indices[i], indices[i+1]
                
                # Determine winner and loser
                if self.fitness[idx1] <= self.fitness[idx2]:
                    winner, loser = idx1, idx2
                else:
                    winner, loser = idx2, idx1
                
                # Get neighborhood mean position for the loser
                mean_position = self._get_neighborhood_mean(loser)
                
                # Update loser's velocity and position
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                r3 = np.random.random(self.dim)
                
                # Velocity update
                self.V[loser] = r1 * self.V[loser] + \
                              r2 * (self.X[winner] - self.X[loser]) + \
                              self.phi * r3 * (mean_position - self.X[loser])
                
                # Position update
                self.X[loser] = self.X[loser] + self.V[loser]
                
                # Boundary handling
                self.X[loser] = np.clip(self.X[loser], self.lb, self.ub)
                
                # Update loser's fitness
                self.fitness[loser] = self.obj_func(self.X[loser])
                
                # Update global best
                if self.fitness[loser] < self.best_fitness:
                    self.best_fitness = self.fitness[loser]
                    self.best_solution = self.X[loser].copy()
            
            # Record current iteration's best fitness
            self.convergence_curve[t] = self.best_fitness
            
            # Print progress
            if (t+1) % 100 == 0 or t == 0:
                print(f'Iteration {t+1}/{self.max_iter}, Best fitness: {self.best_fitness:.6e}')
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def plot_convergence(self):
        """Plot convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, self.max_iter + 1), self.convergence_curve)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value (log scale)')
        plt.title('CSO Convergence Curve')
        plt.grid(True)
        plt.show()