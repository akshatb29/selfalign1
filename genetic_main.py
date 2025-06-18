#!/usr/bin/env python3
"""
Main script demonstrating genetic algorithm integration
"""

import sys
import json
from typing import Dict, List, Any

# Your existing imports
from config import make_llm_config
from agents import create_all_agents, create_llm_configs
from tools import PythonCodeRunner, web_search
from utils import CodeExtractor, safe_initiate_chat
from engine import ReasoningEngine

# New import
from genetic_evolution import SelfCorrectingCodeSystem

class GeneticCodingAssistant:
    """Main assistant integrating genetic algorithms with your existing system"""
    
    def __init__(self):
        # Initialize your existing components
        self.llm_configs = create_llm_configs()
        self.agents = create_all_agents(self.llm_configs)
        
        # Create tools
        self.code_runner = PythonCodeRunner()
        self.code_extractor = CodeExtractor()
        
        # Create enhanced reasoning engine
        self.reasoning_engine = ReasoningEngine(self.agents, self)
        
        # Initialize genetic system
        self.genetic_system = SelfCorrectingCodeSystem(self.agents, self.code_runner)
        
        print("Genetic Coding Assistant initialized successfully!")
    
    def solve_problem(self, problem_description: str, 
                     use_evolution: bool = True,
                     evolution_params: Dict = None) -> Dict[str, Any]:
        """
        Solve a coding problem with optional genetic algorithm evolution
        """
        
        print(f"Solving problem: {problem_description}")
        print(f"Using evolution: {use_evolution}")
        
        if use_evolution:
            # Generate initial solutions using your existing agents
            initial_solutions = self._generate_diverse_solutions(problem_description)
            
            # Create test cases
            test_cases = self._generate_comprehensive_tests(problem_description)
            
            # Run genetic evolution
            results = self.genetic_system.solve_with_evolution(
                problem_description=problem_description,
                test_cases=test_cases,
                initial_solutions=initial_solutions,
                evolution_params=evolution_params or {}
            )
            
            return self._format_results(results, method="genetic_evolution")
        
        else:
            # Use traditional approach
            return self._traditional_solve(problem_description)
    
    def _generate_diverse_solutions(self, problem_description: str, count: int = 5) -> List[str]:
        """Generate diverse initial solutions using different agents/approaches"""
        solutions = []
        
        # Different solution strategies
        strategies = [
            ("codegen", "Write a straightforward solution"),
            ("reasoner", "Write an optimized algorithmic solution"),
            ("codegen", "Write a solution focusing on readability"),
            ("reasoner", "Write a solution with comprehensive error handling"),
            ("codegen", "Write a functional programming style solution")
        ]
        
        for agent_name, approach in strategies[:count]:
            prompt = f"""
            {approach} for this problem:
            {problem_description}
            
            Provide a complete, working Python function.
            """
            
            try:
                response = safe_initiate_chat(self.agents[agent_name], prompt, self.agents['user_proxy'])
                code = self.code_extractor.extract_python_code(response)
                if code and len(code.strip()) > 20:  # Ensure meaningful code
                    solutions.append(code)
            except Exception as e:
                print(f"Error generating solution with {agent_name}: {e}")
                continue
        
        return solutions
    
    def _generate_comprehensive_tests(self, problem_description: str) -> List[Dict]:
        """Generate comprehensive test cases"""
        
        test_prompt = f"""
        Create comprehensive test cases for this coding problem:
        {problem_description}
        
        Generate tests that cover:
        1. Normal cases with typical inputs
        2. Edge cases (empty inputs, boundary values)
        3. Error conditions
        4. Performance edge cases
        
        Format each test case as:
        {{"input": [actual_input_values], "expected_output": expected_result, "description": "what this tests"}}
        
        Provide at least 5 test cases in a JSON list format.
        """
        
        try:
            response = safe_initiate_chat(self.agents['reasoner'], test_prompt, self.agents['user_proxy'])
            # Extract JSON from response
            test_cases = self._extract_json_from_response(response)
            return test_cases if test_cases else self._default_test_cases()
        except Exception as e:
            print(f"Error generating test cases: {e}")
            return self._default_test_cases()
    
    def _extract_json_from_response(self, response: str) -> List[Dict]:
        """Extract JSON test cases from agent response"""
        try:
            # Look for JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"JSON extraction error: {e}")
        return []
    
    def _default_test_cases(self) -> List[Dict]:
        """Provide default test cases when generation fails"""
        return [
            {"input": [], "expected_output": None, "description": "Empty input test"},
            {"input": [1], "expected_output": 1, "description": "Single element test"},
            {"input": [1, 2, 3], "expected_output": 6, "description": "Multiple elements test"}
        ]
    
    def _traditional_solve(self, problem_description: str) -> Dict[str, Any]:
        """Solve using traditional agent-based approach"""
        
        prompt = f"""
        Solve this coding problem step by step:
        {problem_description}
        
        Provide:
        1. Problem analysis
        2. Solution approach
        3. Complete working code
        4. Test cases
        """
        
        response = safe_initiate_chat(self.agents['codegen'], prompt, self.agents['user_proxy'])
        code = self.code_extractor.extract_python_code(response)
        
        return {
            'solution': code,
            'method': 'traditional',
            'response': response,
            'fitness': None
        }
    
    def _format_results(self, results: Dict, method: str) -> Dict[str, Any]:
        """Format results for output"""
        
        if method == "genetic_evolution":
            best_solution = results['best_solution']
            evolution_summary = results['evolution_summary']
            
            return {
                'solution_code': best_solution.code,
                'fitness_score': best_solution.fitness,
                'generation_evolved': best_solution.generation,
                'test_results': best_solution.test_results,
                'evolution_stats': {
                    'total_generations': evolution_summary.get('total_generations', 0),
                    'final_avg_fitness': evolution_summary.get('final_avg_fitness', 0),
                    'improvement': evolution_summary.get('improvement', 0),
                    'diversity': evolution_summary.get('final_diversity', 0)
                },
                'method': method,
                'population_size': len(results.get('final_population', [])),
                'success': True
            }
        
        return results
    
    def continuous_improvement(self, problem_description: str, cycles: int = 3) -> Dict[str, Any]:
        """Run continuous improvement cycles"""
        
        print(f"Starting continuous improvement with {cycles} cycles...")
        
        # Generate comprehensive test cases
        test_cases = self._generate_comprehensive_tests(problem_description)
        
        # Run continuous improvement
        results = self.genetic_system.continuous_improvement(
            problem_description=problem_description,
            test_cases=test_cases,
            improvement_cycles=cycles
        )
        
        return {
            'final_solution': results['final_best'].code,
            'final_fitness': results['final_best'].fitness,
            'total_improvement': results['total_improvement'],
            'improvement_history': results['improvement_history'],
            'method': 'continuous_genetic_improvement'
        }
    
    def interactive_evolution(self):
        """Interactive mode for testing genetic evolution"""
        
        print("\n" + "="*60)
        print("GENETIC ALGORITHM CODING ASSISTANT")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Solve problem with genetic evolution")
            print("2. Solve problem traditionally")
            print("3. Run continuous improvement")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '4':
                break
            
            if choice in ['1', '2', '3']:
                problem = input("\nEnter the coding problem description: ").strip()
                
                if not problem:
                    print("Please enter a valid problem description.")
                    continue
                
                if choice == '1':
                    print("\nSolving with genetic evolution...")
                    results = self.solve_problem(problem, use_evolution=True)
                    self._display_results(results)
                
                elif choice == '2':
                    print("\nSolving with traditional approach...")
                    results = self.solve_problem(problem, use_evolution=False)
                    self._display_results(results)
                
                elif choice == '3':
                    cycles = input("Enter number of improvement cycles (default 3): ").strip()
                    cycles = int(cycles) if cycles.isdigit() else 3
                    
                    print(f"\nRunning continuous improvement with {cycles} cycles...")
                    results = self.continuous_improvement(problem, cycles)
                    self._display_results(results)
            
            else:
                print("Invalid choice. Please try again.")
    
    def _display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        
        if 'solution_code' in results:
            print("Final Solution:")
            print("-" * 30)
            print(results['solution_code'])
            print("-" * 30)
            
            print(f"Fitness Score: {results.get('fitness_score', 'N/A')}")
            print(f"Method: {results.get('method', 'N/A')}")
            
            if 'evolution_stats' in results:
                stats = results['evolution_stats']
                print(f"Generations: {stats.get('total_generations', 'N/A')}")
                print(f"Improvement: {stats.get('improvement', 'N/A'):.3f}")
                print(f"Final Diversity: {stats.get('diversity', 'N/A'):.3f}")
        
        elif 'final_solution' in results:
            print("Final Solution:")
            print("-" * 30)
            print(results['final_solution'])
            print("-" * 30)
            
            print(f"Final Fitness: {results.get('final_fitness', 'N/A')}")
            print(f"Total Improvement: {results.get('total_improvement', 'N/A'):.3f}")
        
        else:
            print("Solution:")
            print("-" * 30)
            print(results.get('solution', 'No solution found'))
            print("-" * 30)


def main():
    """Main function to run the genetic coding assistant"""
    
    try:
        assistant = GeneticCodingAssistant()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--interactive":
                assistant.interactive_evolution()
            else:
                # Solve the problem provided as argument
                problem = " ".join(sys.argv[1:])
                results = assistant.solve_problem(problem, use_evolution=True)
                assistant._display_results(results)
        else:
            # Demo mode
            demo_problem = "Write a function that finds the longest common subsequence between two strings"
            
            print("Running demo with genetic evolution...")
            results = assistant.solve_problem(demo_problem, use_evolution=True)
            assistant._display_results(results)
            
            print("\n" + "="*60)
            print("Demo completed! Use --interactive flag for interactive mode.")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()