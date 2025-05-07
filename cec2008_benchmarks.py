import numpy as np


class CEC2008Functions:
    """CEC2008 Large Scale Optimization Benchmark Functions"""
    
    def __init__(self):
        # Initialize shift vectors
        np.random.seed(42)  # Fixed random seed for reproducibility
        self.shifts = {}
        self.bias = {
            1: -450,  # Shifted Sphere
            2: -450,  # Shifted Schwefel's Problem 2.21
            3: 390,   # Shifted Rosenbrock
            4: -330,  # Shifted Rastrigin
            5: -180,  # Shifted Griewank
            6: -140,  # Shifted Ackley
        }
        
        # Generate shift vectors for each function
        self._generate_shifts(1000)  # Pre-generate for up to 1000D
    
    def _generate_shifts(self, max_dim):
        """Generate shift vectors"""
        # Function 1: Shifted Sphere [-100, 100]
        self.shifts[1] = np.random.uniform(-100, 100, max_dim)
        
        # Function 2: Shifted Schwefel's Problem 2.21 [-100, 100]
        self.shifts[2] = np.random.uniform(-100, 100, max_dim)
        
        # Function 3: Shifted Rosenbrock [-100, 100]
        self.shifts[3] = np.random.uniform(-100, 100, max_dim)
        
        # Function 4: Shifted Rastrigin [-5, 5]
        self.shifts[4] = np.random.uniform(-5, 5, max_dim)
        
        # Function 5: Shifted Griewank [-600, 600]
        self.shifts[5] = np.random.uniform(-600, 600, max_dim)
        
        # Function 6: Shifted Ackley [-32, 32]
        self.shifts[6] = np.random.uniform(-32, 32, max_dim)
    
    def get_bounds(self, func_id):
        """Return the bounds for each function"""
        bounds = {
            1: (-100, 100),   # Shifted Sphere
            2: (-100, 100),   # Shifted Schwefel's Problem 2.21
            3: (-100, 100),   # Shifted Rosenbrock
            4: (-5, 5),       # Shifted Rastrigin
            5: (-600, 600),   # Shifted Griewank
            6: (-32, 32),     # Shifted Ackley
            7: (-1, 1)        # FastFractal DoubleDip
        }
        return bounds.get(func_id, (-100, 100))
    
    def f1(self, x):
        """Shifted Sphere Function"""
        dim = len(x)
        o = self.shifts[1][:dim]
        z = x - o
        return np.sum(z**2) + self.bias[1]
    
    def f2(self, x):
        """Shifted Schwefel's Problem 2.21"""
        dim = len(x)
        o = self.shifts[2][:dim]
        z = x - o
        return np.max(np.abs(z)) + self.bias[2]
    
    def f3(self, x):
        """Shifted Rosenbrock's Function"""
        dim = len(x)
        o = self.shifts[3][:dim]
        z = x - o + 1
        return np.sum(100.0 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2) + self.bias[3]
    
    def f4(self, x):
        """Shifted Rastrigin's Function"""
        dim = len(x)
        o = self.shifts[4][:dim]
        z = x - o
        return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) + self.bias[4]
    
    def f5(self, x):
        """Shifted Griewank's Function"""
        dim = len(x)
        o = self.shifts[5][:dim]
        z = x - o
        i_sqrt = np.sqrt(np.arange(1, dim+1))
        return 1.0/4000.0 * np.sum(z**2) - np.prod(np.cos(z / i_sqrt)) + 1.0 + self.bias[5]
    
    def f6(self, x):
        """Shifted Ackley's Function"""
        dim = len(x)
        o = self.shifts[6][:dim]
        z = x - o
        sum1 = np.sum(z**2)
        sum2 = np.sum(np.cos(2 * np.pi * z))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / dim))
        term2 = -np.exp(sum2 / dim)
        
        return term1 + term2 + 20 + np.e + self.bias[6]
    
    def f7(self, x):
        """FastFractal "DoubleDip" Function (simplified version)"""
        dim = len(x)
        result = 0
        
        for i in range(dim):
            xi = x[i]
            xi_next = x[(i+1) % dim]
            
            # Simplified fractal calculation
            term1 = 4 * (xi**4 - 2*xi**3 + xi**2)
            term2 = 6144 * (xi_next**6) - 3088 * (xi_next**4) + 392 * (xi_next**2) - 1
            
            if -0.5 < xi_next < 0.5:
                result += term1 + term2
        
        return result

    def evaluate(self, func_id, x):
        """Evaluate fitness based on function ID"""
        if func_id == 1:
            return self.f1(x)
        elif func_id == 2:
            return self.f2(x)
        elif func_id == 3:
            return self.f3(x)
        elif func_id == 4:
            return self.f4(x)
        elif func_id == 5:
            return self.f5(x)
        elif func_id == 6:
            return self.f6(x)
        elif func_id == 7:
            return self.f7(x)
        else:
            raise ValueError(f"Unknown function ID: {func_id}")