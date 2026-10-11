import random
import numpy as np
from typing import List, Tuple, Callable

class GeneticAddition:
    def __init__(self, population_size: int = 100, gene_length: int = 8):
        """
        Initialize the genetic algorithm for learning addition.
        
        Args:
            population_size: Number of individuals in the population
            gene_length: Length of binary string representing each number
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.max_value = 2 ** gene_length - 1
        
        # Initialize population with random binary strings
        self.population = [
            ''.join(random.choice('01') for _ in range(gene_length))
            for _ in range(population_size)
        ]
        
    def binary_to_int(self, binary: str) -> int:
        """Convert binary string to integer."""
        return int(binary, 2)
    
    def fitness(self, chromosome: str, x: int, y: int) -> float:
        """
        Calculate fitness of a chromosome based on how close its value is to x + y.
        
        Args:
            chromosome: Binary string representing a potential solution
            x: First number to add
            y: Second number to add
            
        Returns:
            Fitness score (higher is better)
        """
        predicted_sum = self.binary_to_int(chromosome)
        actual_sum = x + y
        error = abs(predicted_sum - actual_sum)
        return 1 / (error + 1)  # Add 1 to avoid division by zero
    
    def select_parent(self, fitnesses: List[float]) -> str:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(list(enumerate(fitnesses)), tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return self.population[winner_idx]
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Perform single-point crossover between two parents."""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome: str, mutation_rate: float = 0.01) -> str:
        """Perform bit-flip mutation on a chromosome."""
        return ''.join(
            str(1 - int(bit)) if random.random() < mutation_rate else bit
            for bit in chromosome
        )
    
    def evolve(self, x: int, y: int, generations: int = 100) -> Tuple[str, float]:
        """
        Evolve the population to learn the addition of two numbers.
        
        Args:
            x: First number to add
            y: Second number to add
            generations: Number of generations to evolve
            
        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        for gen in range(generations):
            # Calculate fitness for each chromosome
            fitnesses = [self.fitness(chrom, x, y) for chrom in self.population]
            
            # Store best solution
            best_idx = fitnesses.index(max(fitnesses))
            best_chromosome = self.population[best_idx]
            best_fitness = fitnesses[best_idx]
            
            # Create new population through selection, crossover, and mutation
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.select_parent(fitnesses)
                parent2 = self.select_parent(fitnesses)
                
                # Perform crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Perform mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.population_size]
            
            # Print progress
            if gen % 10 == 0:
                predicted = self.binary_to_int(best_chromosome)
                print(f"Generation {gen}: Best prediction: {predicted}, "
                      f"Actual: {x + y}, Fitness: {best_fitness:.4f}")
        
        # Return best solution
        final_fitnesses = [self.fitness(chrom, x, y) for chrom in self.population]
        best_idx = final_fitnesses.index(max(final_fitnesses))
        return self.population[best_idx], final_fitnesses[best_idx]

# Example usage
def demo_addition():
    """Demonstrate the genetic algorithm learning addition."""
    # Initialize the genetic algorithm
    ga = GeneticAddition(population_size=100, gene_length=8)
    
    # Numbers to add
    x, y = 15, 27
    
    # Evolve the population
    best_chromosome, best_fitness = ga.evolve(x, y, generations=100)
    
    # Print results
    predicted_sum = ga.binary_to_int(best_chromosome)
    actual_sum = x + y
    print("\nFinal Results:")
    print(f"Numbers to add: {x} + {y} = {actual_sum}")
    print(f"Predicted sum: {predicted_sum}")
    print(f"Final fitness: {best_fitness:.4f}")
    print(f"Binary representation: {best_chromosome}")

if __name__ == "__main__":
    demo_addition()