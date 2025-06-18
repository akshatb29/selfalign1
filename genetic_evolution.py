"""
Self-Correcting Coding Model with Genetic Algorithm Integration

This module implements a genetic algorithm-based approach to evolve and 
self-correct code solutions using your existing AutoGen agent architecture.
"""

import random
import ast
import copy
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Assuming your existing imports
# from agents import create_all_agents
# from tools import PythonCodeRunner
# from utils import CodeExtractor, safe_initiate_chat


@dataclass
class CodeIndividual:
    """Represents a single code solution in the genetic algorithm population"""
    code: str
    fitness: float = 0.0
    test_results: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    
    @property
    def id(self) -> str:
        """Generate unique ID based on code content"""
        return hashlib.md5(self.code.encode()).hexdigest()[:8]
    
    def __post_init__(self):
        """Calculate initial fitness if code is provided"""
        if self.code and self.fitness == 0.0:
            self.fitness = self._calculate_basic_fitness()
    
    def _calculate_basic_fitness(self) -> float:
        """Basic fitness calculation based on code structure"""
        try:
            # Parse the code to check if it's syntactically correct
            ast.parse(self.code)
            base_fitness = 0.5  # Base fitness for valid syntax
            
            # Add points for code complexity and structure
            lines = len(self.code.split('\n'))
            functions = self.code.count('def ')
            classes = self.code.count('class ')
            
            structure_bonus = (functions * 0.1) + (classes * 0.15) + (lines * 0.01)
            return min(base_fitness + structure_bonus, 1.0)
        except SyntaxError:
            return 0.0


class CodeMutator:
    """Handles different types of code mutations"""
    
    def __init__(self, agents_config: Dict[str, Any]):
        self.agents = agents_config
        self.mutation_strategies = [
            self._syntax_mutation,
            self._logic_mutation,
            self._optimization_mutation,
            self._refactor_mutation
        ]
    
    def mutate(self, individual: CodeIndividual, mutation_rate: float = 0.3) -> CodeIndividual:
        """Apply random mutations to code individual"""
        if random.random() > mutation_rate:
            return individual
        
        strategy = random.choice(self.mutation_strategies)
        mutated_code = strategy(individual.code)
        
        mutated_individual = CodeIndividual(
            code=mutated_code,
            generation=individual.generation + 1,
            parent_ids=[individual.id],
            mutation_history=individual.mutation_history + [strategy.__name__]
        )
        
        return mutated_individual
    
    def _syntax_mutation(self, code: str) -> str:
        """Apply syntax-level mutations"""
        # Use your CodeGen agent to suggest syntax improvements
        prompt = f"""
        Apply minor syntax improvements to this code while preserving functionality:
        
        {code}
        
        Focus on:
        - Variable naming improvements
        - Code formatting enhancements
        - Minor syntax optimizations
        
        Return only the improved code.
        """
        
        # This would integrate with your existing agent system
        # result = safe_initiate_chat(self.agents['codegen'], prompt)
        # return CodeExtractor.extract_python_code(result)
        
        # Placeholder implementation
        return self._apply_simple_syntax_changes(code)
    
    def _logic_mutation(self, code: str) -> str:
        """Apply logic-level mutations"""
        prompt = f"""
        Suggest alternative logic approaches for this code while maintaining the same functionality:
        
        {code}
        
        Focus on:
        - Different algorithmic approaches
        - Alternative control structures
        - Edge case handling improvements
        
        Return only the modified code.
        """
        
        return self._apply_logic_variations(code)
    
    def _optimization_mutation(self, code: str) -> str:
        """Apply performance optimization mutations"""
        prompt = f"""
        Optimize this code for better performance:
        
        {code}
        
        Focus on:
        - Time complexity improvements
        - Memory usage optimization
        - Built-in function utilization
        
        Return only the optimized code.
        """
        
        return self._apply_performance_optimizations(code)
    
    def _refactor_mutation(self, code: str) -> str:
        """Apply refactoring mutations"""
        return self._apply_refactoring_changes(code)
    
    def _apply_simple_syntax_changes(self, code: str) -> str:
        """Simple syntax mutation implementation"""
        lines = code.split('\n')
        if not lines:
            return code
        
        # Random small changes
        mutations = [
            lambda l: l.replace('  ', '    ') if '  ' in l else l,  # Indentation
            lambda l: l.replace(' = ', ' = ') if ' = ' in l else l,  # Spacing
            lambda l: l.strip() + ' ' if l.strip() else l  # Trailing space
        ]
        
        mutated_lines = []
        for line in lines:
            if random.random() < 0.1:  # 10% chance to mutate each line
                mutation = random.choice(mutations)
                mutated_lines.append(mutation(line))
            else:
                mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _apply_logic_variations(self, code: str) -> str:
        """Apply logic variations"""
        # Placeholder - in real implementation, use your agents
        return code
    
    def _apply_performance_optimizations(self, code: str) -> str:
        """Apply performance optimizations"""
        # Placeholder - in real implementation, use your agents
        return code
    
    def _apply_refactoring_changes(self, code: str) -> str:
        """Apply refactoring changes"""
        # Placeholder - in real implementation, use your agents
        return code


class CodeCrossover:
    """Handles crossover operations between code individuals"""
    
    def __init__(self, agents_config: Dict[str, Any]):
        self.agents = agents_config
    
    def crossover(self, parent1: CodeIndividual, parent2: CodeIndividual) -> List[CodeIndividual]:
        """Perform crossover between two parent individuals"""
        crossover_strategies = [
            self._function_level_crossover,
            self._semantic_crossover,
            self._hybrid_crossover
        ]
        
        strategy = random.choice(crossover_strategies)
        children = strategy(parent1, parent2)
        
        # Set parent information
        for child in children:
            child.parent_ids = [parent1.id, parent2.id]
            child.generation = max(parent1.generation, parent2.generation) + 1
        
        return children
    
    def _function_level_crossover(self, parent1: CodeIndividual, parent2: CodeIndividual) -> List[CodeIndividual]:
        """Crossover at function level"""
        # Extract functions from both parents
        functions1 = self._extract_functions(parent1.code)
        functions2 = self._extract_functions(parent2.code)
        
        # Create hybrid combinations
        child1_code = self._combine_functions(functions1, functions2, 0.7)
        child2_code = self._combine_functions(functions2, functions1, 0.7)
        
        return [
            CodeIndividual(code=child1_code),
            CodeIndividual(code=child2_code)
        ]
    
    def _semantic_crossover(self, parent1: CodeIndividual, parent2: CodeIndividual) -> List[CodeIndividual]:
        """Semantic-aware crossover using agents"""
        prompt = f"""
        Create two new code solutions by intelligently combining these two approaches:
        
        Solution 1:
        {parent1.code}
        
        Solution 2:
        {parent2.code}
        
        Generate two hybrid solutions that combine the best aspects of both.
        Focus on maintaining functionality while exploring new solution paths.
        
        Return the solutions separated by "=== SOLUTION 2 ===" marker.
        """
        
        # Placeholder - would use your agents here
        return [
            CodeIndividual(code=parent1.code),  # Fallback
            CodeIndividual(code=parent2.code)
        ]
    
    def _hybrid_crossover(self, parent1: CodeIndividual, parent2: CodeIndividual) -> List[CodeIndividual]:
        """Hybrid crossover combining multiple strategies"""
        # Combine different aspects of the solutions
        return self._function_level_crossover(parent1, parent2)
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract function definitions from code"""
        try:
            tree = ast.parse(code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(code, node)
                    if func_code:
                        functions.append(func_code)
            return functions
        except:
            return []
    
    def _combine_functions(self, funcs1: List[str], funcs2: List[str], ratio: float) -> str:
        """Combine functions from two sources"""
        combined = []
        
        # Take functions from first source based on ratio
        num_from_first = int(len(funcs1) * ratio)
        combined.extend(funcs1[:num_from_first])
        
        # Add functions from second source
        remaining_slots = max(0, len(funcs1) - num_from_first)
        combined.extend(funcs2[:remaining_slots])
        
        return '\n\n'.join(combined)


class FitnessEvaluator:
    """Evaluates fitness of code individuals"""
    
    def __init__(self, code_runner, test_cases: List[Dict[str, Any]]):
        self.code_runner = code_runner
        self.test_cases = test_cases
        self.fitness_cache = {}
    
    def evaluate_fitness(self, individual: CodeIndividual) -> float:
        """Evaluate fitness of a code individual"""
        # Check cache first
        code_hash = individual.id
        if code_hash in self.fitness_cache:
            individual.fitness = self.fitness_cache[code_hash]
            return individual.fitness
        
        fitness_components = {
            'correctness': self._evaluate_correctness(individual),
            'performance': self._evaluate_performance(individual),
            'code_quality': self._evaluate_code_quality(individual),
            'robustness': self._evaluate_robustness(individual)
        }
        
        # Weighted fitness calculation
        weights = {
            'correctness': 0.5,
            'performance': 0.2,
            'code_quality': 0.2,
            'robustness': 0.1
        }
        
        fitness = sum(fitness_components[key] * weights[key] for key in weights)
        individual.fitness = fitness
        individual.test_results = fitness_components
        
        # Cache the result
        self.fitness_cache[code_hash] = fitness
        
        return fitness
    
    def _evaluate_correctness(self, individual: CodeIndividual) -> float:
        """Evaluate correctness against test cases"""
        if not self.test_cases:
            return individual._calculate_basic_fitness()
        
        passed_tests = 0
        total_tests = len(self.test_cases)
        
        for test_case in self.test_cases:
            try:
                result = self.code_runner.run_code(individual.code, test_case)
                if result.get('success', False):
                    passed_tests += 1
            except Exception as e:
                continue  # Test failed
        
        return passed_tests / total_tests if total_tests > 0 else 0.0
    
    def _evaluate_performance(self, individual: CodeIndividual) -> float:
        """Evaluate performance characteristics"""
        try:
            # Simple performance metrics
            lines = len(individual.code.split('\n'))
            complexity_score = 1.0 / (1.0 + lines * 0.01)  # Favor concise code
            
            # Add more sophisticated performance analysis here
            return min(complexity_score, 1.0)
        except:
            return 0.0
    
    def _evaluate_code_quality(self, individual: CodeIndividual) -> float:
        """Evaluate code quality metrics"""
        try:
            # Check syntax validity
            ast.parse(individual.code)
            base_quality = 0.5
            
            # Add quality metrics
            has_docstrings = '"""' in individual.code or "'''" in individual.code
            has_comments = '#' in individual.code
            proper_naming = not any(var in individual.code.lower() for var in ['temp', 'tmp', 'x', 'y'])
            
            quality_bonus = 0
            if has_docstrings:
                quality_bonus += 0.2
            if has_comments:
                quality_bonus += 0.1
            if proper_naming:
                quality_bonus += 0.2
            
            return min(base_quality + quality_bonus, 1.0)
        except SyntaxError:
            return 0.0
    
    def _evaluate_robustness(self, individual: CodeIndividual) -> float:
        """Evaluate robustness and error handling"""
        robustness_indicators = [
            'try:' in individual.code,
            'except' in individual.code,
            'raise' in individual.code,
            'assert' in individual.code
        ]
        
        return sum(robustness_indicators) / len(robustness_indicators)


class GeneticCodeEvolver:
    """Main genetic algorithm controller for code evolution"""
    
    def __init__(self, 
                 agents_config: Dict[str, Any],
                 code_runner,
                 test_cases: List[Dict[str, Any]],
                 population_size: int = 20,
                 elite_size: int = 4,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7):
        
        self.agents = agents_config
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.mutator = CodeMutator(agents_config)
        self.crossover = CodeCrossover(agents_config)
        self.fitness_evaluator = FitnessEvaluator(code_runner, test_cases)
        
        self.population: List[CodeIndividual] = []
        self.generation = 0
        self.evolution_history = []
    
    def initialize_population(self, initial_solutions: List[str]) -> None:
        """Initialize the population with seed solutions"""
        self.population = []
        
        # Add provided initial solutions
        for code in initial_solutions:
            individual = CodeIndividual(code=code, generation=0)
            self.population.append(individual)
        
        # Generate additional random solutions if needed
        while len(self.population) < self.population_size:
            # Use agents to generate diverse initial solutions
            generated_code = self._generate_random_solution()
            if generated_code:
                individual = CodeIndividual(code=generated_code, generation=0)
                self.population.append(individual)
        
        # Evaluate initial fitness
        self._evaluate_population()
    
    def evolve(self, generations: int = 50, target_fitness: float = 0.95) -> CodeIndividual:
        """Run the genetic algorithm evolution process"""
        best_individual = None
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate current population
            self._evaluate_population()
            
            # Track best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if best_individual is None or current_best.fitness > best_individual.fitness:
                best_individual = copy.deepcopy(current_best)
            
            # Check termination condition
            if current_best.fitness >= target_fitness:
                print(f"Target fitness {target_fitness} reached at generation {gen}")
                break
            
            # Record evolution history
            self._record_generation_stats()
            
            # Create next generation
            self._create_next_generation()
            
            # Print progress
            if gen % 10 == 0:
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                print(f"Generation {gen}: Best={current_best.fitness:.3f}, Avg={avg_fitness:.3f}")
        
        return best_individual
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for entire population"""
        # Use parallel evaluation for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.fitness_evaluator.evaluate_fitness, ind): ind 
                      for ind in self.population}
            
            for future in as_completed(futures):
                individual = futures[future]
                try:
                    fitness = future.result()
                except Exception as e:
                    print(f"Error evaluating individual {individual.id}: {e}")
                    individual.fitness = 0.0
    
    def _create_next_generation(self) -> None:
        """Create the next generation using selection, crossover, and mutation"""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elitism: keep best individuals
        elite_individuals = self.population[:self.elite_size]
        new_population.extend(copy.deepcopy(elite_individuals))
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                children = self.crossover.crossover(parent1, parent2)
                for child in children:
                    if len(new_population) < self.population_size:
                        new_population.append(child)
            else:
                # Clone parents if no crossover
                if len(new_population) < self.population_size:
                    new_population.append(copy.deepcopy(parent1))
                if len(new_population) < self.population_size:
                    new_population.append(copy.deepcopy(parent2))
        
        # Apply mutations
        for i in range(self.elite_size, len(new_population)):
            new_population[i] = self.mutator.mutate(new_population[i], self.mutation_rate)
        
        self.population = new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> CodeIndividual:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _generate_random_solution(self) -> str:
        """Generate a random initial solution using agents"""
        # This would use your existing agents to generate diverse solutions
        # For now, return a simple placeholder
        return """
def solution():
    # Generated solution placeholder
    return None
"""
    
    def _record_generation_stats(self) -> None:
        """Record statistics for current generation"""
        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'diversity': self._calculate_diversity()
        }
        self.evolution_history.append(stats)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        unique_codes = set(ind.id for ind in self.population)
        return len(unique_codes) / len(self.population)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process"""
        if not self.evolution_history:
            return {}
        
        return {
            'total_generations': len(self.evolution_history),
            'final_best_fitness': self.evolution_history[-1]['best_fitness'],
            'final_avg_fitness': self.evolution_history[-1]['avg_fitness'],
            'final_diversity': self.evolution_history[-1]['diversity'],
            'improvement': (self.evolution_history[-1]['best_fitness'] - 
                          self.evolution_history[0]['best_fitness']),
            'history': self.evolution_history
        }


class SelfCorrectingCodeSystem:
    """Main system integrating genetic algorithms with your existing architecture"""
    
    def __init__(self, agents_config: Dict[str, Any], code_runner):
        self.agents = agents_config
        self.code_runner = code_runner
        self.evolver = None
        self.problem_cache = {}
    
    def solve_with_evolution(self, 
                           problem_description: str,
                           test_cases: List[Dict[str, Any]],
                           initial_solutions: List[str] = None,
                           evolution_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Solve a coding problem using genetic algorithm evolution"""
        
        # Generate initial solutions if not provided
        if not initial_solutions:
            initial_solutions = self._generate_initial_solutions(problem_description)
        
        # Set up evolution parameters
        params = evolution_params or {}
        self.evolver = GeneticCodeEvolver(
            agents_config=self.agents,
            code_runner=self.code_runner,
            test_cases=test_cases,
            population_size=params.get('population_size', 20),
            elite_size=params.get('elite_size', 4),
            mutation_rate=params.get('mutation_rate', 0.3),
            crossover_rate=params.get('crossover_rate', 0.7)
        )
        
        # Initialize and evolve
        self.evolver.initialize_population(initial_solutions)
        best_solution = self.evolver.evolve(
            generations=params.get('generations', 50),
            target_fitness=params.get('target_fitness', 0.95)
        )
        
        # Return comprehensive results
        return {
            'best_solution': best_solution,
            'evolution_summary': self.evolver.get_evolution_summary(),
            'final_population': self.evolver.population,
            'problem_description': problem_description,
            'test_cases': test_cases
        }
    
    def _generate_initial_solutions(self, problem_description: str, count: int = 5) -> List[str]:
        """Generate initial solutions using your existing agents"""
        solutions = []
        
        # Use different agents/approaches to generate diverse initial solutions
        prompts = [
            f"Write a simple solution for: {problem_description}",
            f"Write an optimized solution for: {problem_description}",
            f"Write a robust solution with error handling for: {problem_description}",
            f"Write a functional programming approach for: {problem_description}",
            f"Write an object-oriented solution for: {problem_description}"
        ]
        
        for prompt in prompts[:count]:
            # This would use your existing agent system
            # solution = safe_initiate_chat(self.agents['codegen'], prompt)
            # code = CodeExtractor.extract_python_code(solution)
            
            # Placeholder implementation
            code = f"""
def solution():
    # Solution for: {problem_description}
    # Generated with prompt: {prompt[:50]}...
    pass
"""
            solutions.append(code)
        
        return solutions
    
    def continuous_improvement(self, 
                             problem_description: str,
                             test_cases: List[Dict[str, Any]],
                             improvement_cycles: int = 3) -> Dict[str, Any]:
        """Continuously improve solutions through multiple evolution cycles"""
        
        results_history = []
        current_best = None
        
        for cycle in range(improvement_cycles):
            print(f"Starting improvement cycle {cycle + 1}/{improvement_cycles}")
            
            # Use previous best solutions as seeds for next cycle
            initial_solutions = []
            if current_best:
                initial_solutions.append(current_best.code)
                # Generate variations of the best solution
                for _ in range(4):
                    mutated = self.evolver.mutator.mutate(current_best)
                    initial_solutions.append(mutated.code)
            
            # Run evolution cycle
            cycle_results = self.solve_with_evolution(
                problem_description=problem_description,
                test_cases=test_cases,
                initial_solutions=initial_solutions,
                evolution_params={
                    'generations': 30,
                    'population_size': 15,
                    'mutation_rate': 0.4 - (cycle * 0.1)  # Decrease mutation over cycles
                }
            )
            
            results_history.append(cycle_results)
            
            # Update current best
            cycle_best = cycle_results['best_solution']
            if current_best is None or cycle_best.fitness > current_best.fitness:
                current_best = cycle_best
            
            print(f"Cycle {cycle + 1} completed. Best fitness: {cycle_best.fitness:.3f}")
        
        return {
            'final_best': current_best,
            'improvement_history': results_history,
            'total_improvement': (current_best.fitness - results_history[0]['best_solution'].fitness 
                                if len(results_history) > 0 else 0)
        }


# Example usage and integration functions
def integrate_with_existing_system(config_module, agents_module, tools_module):
    """
    Integration function to connect genetic algorithm with existing system
    
    Args:
        config_module: Your config.py module
        agents_module: Your agents.py module  
        tools_module: Your tools.py module
    """
    
    # Create agent configurations
    llm_configs = agents_module.create_llm_configs()
    agents = agents_module.create_all_agents(llm_configs)
    
    # Create code runner
    code_runner = tools_module.PythonCodeRunner()
    
    # Initialize self-correcting system
    system = SelfCorrectingCodeSystem(agents, code_runner)
    
    return system


# Example test cases format
EXAMPLE_TEST_CASES = [
    {
        'input': [1, 2, 3, 4, 5],
        'expected_output': 15,
        'description': 'Sum of list elements'
    },
    {
        'input': [],
        'expected_output': 0,
        'description': 'Empty list should return 0'
    },
    {
        'input': [-1, -2, -3],
        'expected_output': -6,
        'description': 'Negative numbers'
    }
]


if __name__ == "__main__":
    # Example usage
    print("Genetic Algorithm Self-Correcting Code Evolution System")
    print("=" * 60)
    
    # This would be integrated with your existing system
    # system = integrate_with_existing_system(config, agents, tools)
    
    # Example problem
    problem = "Write a function that calculates the sum of all elements in a list"
    
    # Example evolution run
    # results = system.solve_with_evolution(
    #     problem_description=problem,
    #     test_cases=EXAMPLE_TEST_CASES,
    #     evolution_params={
    #         'generations': 20,
    #         'population_size': 10,
    #         'target_fitness': 0.9
    #     }
    # )
    
    print("System ready for integration!")