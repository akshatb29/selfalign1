import unittest
from genetic_main import GeneticCodingAssistant

class TestGeneticIntegration(unittest.TestCase):
    
    def setUp(self):
        self.assistant = GeneticCodingAssistant()
    
    def test_simple_problem_solving(self):
        """Test basic problem solving"""
        result = self.assistant.solve_problem(
            "Write a function that returns the sum of two numbers",
            use_evolution=False  # Start with traditional approach
        )
        
        self.assertIsNotNone(result)
        self.assertIn('solution', result)
    
    def test_genetic_evolution(self):
        """Test genetic algorithm evolution"""
        result = self.assistant.solve_problem(
            "Write a function that finds the maximum element in a list",
            use_evolution=True
        )
        
        self.assertIsNotNone(result)
        self.assertIn('solution_code', result)
        self.assertIn('fitness_score', result)
    
    def test_test_case_generation(self):
        """Test test case generation"""
        test_cases = self.assistant._generate_comprehensive_tests(
            "Write a function that reverses a string"
        )
        
        self.assertIsInstance(test_cases, list)
        self.assertGreater(len(test_cases), 0)

if __name__ == '__main__':
    unittest.main()