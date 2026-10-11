import random
import numpy as np
from typing import List, Tuple, Callable

class GeneralGeneticAddition:
    def __init__(self, population_size: int = 200, input_bits: int = 4):
        """
        Initialize the genetic algorithm for learning general addition.
        
        Args:
            population_size: Number of individuals in the population
            input_bits: Number of bits for each input number
        """
        self.population_size = population_size
        self.input_bits = input_bits
        # Output needs an extra bit to handle carries
        self.output_bits = input_bits + 1
        
        # Each chromosome represents a lookup table for all possible input combinations
        # For n input bits, we need 2^(2n) entries, each with n+1 output bits
        self.table_entries = 2 ** (2 * input_bits)
        self.chromosome_length = self.table_entries * self.output_bits
        
        # Initialize population with random binary strings
        self.population = [
            ''.join(random.choice('01') for _ in range(self.chromosome_length))
            for _ in range(population_size)
        ]
        
    def decode_chromosome(self, chromosome: str) -> List[List[int]]:
        """Convert chromosome into a lookup table."""
        table = []
        for i in range(0, len(chromosome), self.output_bits):
            entry = chromosome[i:i + self.output_bits]
            table.append([int(bit) for bit in entry])
        return table
    
    def get_table_index(self, x: int, y: int) -> int:
        """Get the index in the lookup table for a pair of numbers."""
        return (x << self.input_bits) | y
    
    def binary_to_int(self, bits: List[int]) -> int:
        """Convert binary list to integer."""
        return sum(bit << i for i, bit in enumerate(bits))
    
    def predict_sum(self, chromosome: str, x: int, y: int) -> int:
        """Use the chromosome's lookup table to predict sum of x and y."""
        table = self.decode_chromosome(chromosome)
        index = self.get_table_index(x, y)
        return self.binary_to_int(table[index])
    
    def fitness(self, chromosome: str, test_cases: List[Tuple[int, int]] = None) -> float:
        """
        Calculate fitness based on accuracy across multiple test cases.
        
        Args:
            chromosome: Binary string representing the lookup table
            test_cases: List of (x, y) pairs to test. If None, generates random cases.
        """
        if test_cases is None:
            # Generate random test cases if none provided
            max_value = 2 ** self.input_bits - 1
            test_cases = [
                (random.randint(0, max_value), random.randint(0, max_value))
                for _ in range(20)
            ]
        
        total_error = 0
        for x, y in test_cases:
            predicted = self.predict_sum(chromosome, x, y)
            actual = x + y
            total_error += abs(predicted - actual)
        
        return 1 / (total_error + 1)  # Add 1 to avoid division by zero
    
    def select_parent(self, fitnesses: List[float]) -> str:
        """Select parent using tournament selection."""
        tournament_size = 5
        tournament = random.sample(list(enumerate(fitnesses)), tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return self.population[winner_idx]
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Perform multi-point crossover between two parents."""
        # Use multiple crossover points based on the lookup table structure
        points = sorted(random.sample(
            range(self.output_bits, len(parent1) - self.output_bits),
            2
        ))
        
        child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
        return child1, child2
    
    def mutate(self, chromosome: str, mutation_rate: float = 0.001) -> str:
        """Perform bit-flip mutation on a chromosome."""
        return ''.join(
            str(1 - int(bit)) if random.random() < mutation_rate else bit
            for bit in chromosome
        )
    
    def evolve(self, generations: int = 200) -> Tuple[str, float]:
        """
        Evolve the population to learn general addition.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        max_value = 2 ** self.input_bits - 1
        
        # Create a fixed set of test cases for consistent evaluation
        test_cases = [(x, y) 
                     for x in range(max_value + 1) 
                     for y in range(max_value + 1)]
        
        best_overall_fitness = 0
        best_overall_chromosome = None
        
        for gen in range(generations):
            # Calculate fitness for each chromosome
            fitnesses = [self.fitness(chrom, test_cases) for chrom in self.population]
            
            # Store best solution
            best_idx = fitnesses.index(max(fitnesses))
            best_chromosome = self.population[best_idx]
            best_fitness = fitnesses[best_idx]
            
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_chromosome = best_chromosome
            
            # Create new population through selection, crossover, and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(fitnesses)
                parent2 = self.select_parent(fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            
            # Print progress
            if gen % 10 == 0:
                # Test some random cases
                test_x, test_y = random.randint(0, max_value), random.randint(0, max_value)
                predicted = self.predict_sum(best_chromosome, test_x, test_y)
                print(f"Generation {gen}: Testing {test_x} + {test_y} = {predicted} "
                      f"(Actual: {test_x + test_y}), Fitness: {best_fitness:.4f}")
        
        return best_overall_chromosome, best_overall_fitness

def demo_general_addition():
    """Demonstrate the genetic algorithm learning general addition."""
    # Initialize with 4-bit numbers (0-15)
    ga = GeneralGeneticAddition(population_size=200, input_bits=4)
    
    # Evolve the population
    best_chromosome, best_fitness = ga.evolve(generations=200)
    
    # Test the evolved solution
    print("\nFinal Results:")
    print(f"Final fitness: {best_fitness:.4f}")
    
    # Test some random cases
    max_value = 2 ** ga.input_bits - 1
    print("\nTesting random cases:")
    for _ in range(5):
        x = random.randint(0, max_value)
        y = random.randint(0, max_value)
        predicted = ga.predict_sum(best_chromosome, x, y)
        print(f"{x} + {y} = {predicted} (Actual: {x + y})")

if __name__ == "__main__":
    demo_general_addition()