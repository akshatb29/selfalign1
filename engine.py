import json
import re
from typing import List, Dict, Any
from utils import safe_initiate_chat
from genetic_evolution import SelfCorrectingCodeSystem, GeneticCodeEvolver

class ReasoningPipelines:
    """
    Contains different reasoning pipelines for code generation.
    """
    def __init__(self, agents, user_proxy):
        self.agents = agents
        self.user_proxy = user_proxy

    def code_first_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Code-First] Pipeline...")
        prompt = f"""Directly write the Python code to solve the following task.
The solution should be a single function.
Provide only the code wrapped in ```python ... ```.

Task: {task}"""
        return safe_initiate_chat(self.agents["codegen"], prompt, self.user_proxy)

    def pseudocode_first_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Pseudocode-First] Pipeline...")
        plan_prompt = f"Generate detailed step-by-step pseudocode wrapped in ```pseudocode ... ```.\n\nTask: {task}"
        pseudocode_plan = safe_initiate_chat(self.agents["reasoner"], plan_prompt, self.user_proxy)
        print("--- Generated Plan ---\n", pseudocode_plan, "\n----------------------")

        impl_prompt = f"""Translate this pseudocode into a Python function.
Provide only the code wrapped in ```python ... ```.

Pseudocode:
{pseudocode_plan}"""
        return safe_initiate_chat(self.agents["codegen"], impl_prompt, self.user_proxy)

    def neuro_symbolic_pipeline(self, task: str) -> str:
        print("\n🚀 Running [Neuro-Symbolic] Pipeline...")
        plan_prompt = f"Generate detailed step-by-step pseudocode wrapped in ```pseudocode ... ```.\n\nTask: {task}"
        pseudocode_plan = safe_initiate_chat(self.agents["reasoner"], plan_prompt, self.user_proxy)
        print("--- Generated Plan ---\n", pseudocode_plan, "\n----------------------")

        critique_prompt = f"""Critique this pseudocode for logical flaws, missed edge cases, or inefficiencies.
Refine it if needed. If solid, explain why.

Task: {task}
Plan:
{pseudocode_plan}"""
        critique = safe_initiate_chat(self.agents["reasoner"], critique_prompt, self.user_proxy)
        print("--- Plan Critique ---\n", critique, "\n---------------------")

        impl_prompt = f"""Write the final Python function based on this plan and critique.
Provide only the code wrapped in ```python ... ```.

Original Plan:
{pseudocode_plan}

Critique and Refinements:
{critique}"""
        return safe_initiate_chat(self.agents["codegen"], impl_prompt, self.user_proxy)


class LLMTaskAnalyzer:
    """Analyzes task complexity and suggests reasoning strategy."""
    def __init__(self, analyzer_agent):
        self.analyzer_agent = analyzer_agent

    def analyze_task(self, task: str, user_proxy) -> Dict[str, Any]:
        analysis_prompt = f"""Analyze this programming task. Use the web_search tool if you encounter unfamiliar concepts.

TASK: {task}

Provide a JSON object analysis."""
        try:
            response = safe_initiate_chat(self.analyzer_agent, analysis_prompt, user_proxy)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            print("Warning: Could not parse JSON from analyzer. Using fallback.")
            return {"reasoning_strategy": "CODE_FIRST", "complexity": 0.3, "explanation": "Fallback due to parsing error."}
        except Exception as e:
            print(f"Error in task analysis: {e}. Using fallback.")
            return {"reasoning_strategy": "CODE_FIRST", "complexity": 0.3, "explanation": f"Fallback due to exception: {e}"}


class ReasoningEngine:
    """Reasoning engine integrating traditional pipelines with genetic evolution."""
    def __init__(self, agents, tools):
        self.agents = agents
        self.tools = tools
        self.task_analyzer = LLMTaskAnalyzer(agents["task_analyzer"])
        self.genetic_system = SelfCorrectingCodeSystem(agents, tools.code_runner)
        self.evolution_enabled = True

    def solve_with_evolution(self, task_description: str, requirements: Dict = None) -> Dict[str, Any]:
        task_analysis = self.task_analyzer.analyze_task(task_description, self.agents["user_proxy"])
        test_cases = self._generate_test_cases(task_description, requirements)

        if task_analysis.get('complexity', 0) > 0.5 and self.evolution_enabled:
            print("✅ Using genetic algorithm evolution for complex task...")
            evolution_params = {
                'generations': min(50, max(10, int(task_analysis['complexity'] * 100))),
                'population_size': 15,
                'mutation_rate': 0.3,
                'target_fitness': 0.9
            }
            results = self.genetic_system.solve_with_evolution(
                problem_description=task_description,
                test_cases=test_cases,
                evolution_params=evolution_params
            )
            return self._format_evolution_results(results)
        else:
            return self._traditional_solve(task_description, requirements)

    def _generate_test_cases(self, task_description: str, requirements: Dict = None) -> List[Dict[str, Any]]:
        test_prompt = f"""Generate diverse test cases for this task:
{task_description}

Requirements: {requirements or 'None'}

Format:
[
  {{"input": [values], "expected_output": result, "description": "desc"}},
  ...
]
"""
        response = safe_initiate_chat(self.agents["reasoner"], test_prompt, self.agents["user_proxy"])
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ Test case generation failed: {e}")
        return [{"input": [], "expected_output": None, "description": "Default test case"}]

    def _format_evolution_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        best_solution = results["best_solution"]
        return {
            "solution": best_solution.code,
            "fitness": best_solution.fitness,
            "generation": best_solution.generation,
            "test_results": best_solution.test_results,
            "evolution_summary": results["evolution_summary"],
            "method": "genetic_algorithm"
        }

    def _traditional_solve(self, task_description: str, requirements: Dict = None) -> Dict[str, Any]:
        print("⚡ Falling back to traditional neuro-symbolic pipeline...")
        pipelines = ReasoningPipelines(self.agents, self.agents["user_proxy"])
        solution = pipelines.neuro_symbolic_pipeline(task_description)
        return {
            "solution": solution,
            "method": "traditional"
        }
