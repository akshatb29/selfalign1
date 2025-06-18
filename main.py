# main.py
import json
import time

from agents import create_all_agents
from engine import LLMTaskAnalyzer, ReasoningPipelines
from tools import PythonCodeRunner, web_search
from utils import CodeExtractor, extract_json_from_response, print_test_results, safe_initiate_chat

def get_user_choice(recommended_strategy: str) -> str:
    """Asks the user to confirm or choose a reasoning strategy."""
    print("\n🤔 CHOOSE A REASONING STRATEGY")
    print("-" * 40)
    print(f"LLM Recommends: [{recommended_strategy}]")
    print("1: Code-First (Simple, direct tasks)")
    print("2: Pseudocode-First (Medium, algorithmic tasks)")
    print("3: Neuro-Symbolic (Complex, critical tasks)")
    
    strategy_map = {
        "1": "CODE_FIRST",
        "2": "PSEUDOCODE_FIRST",
        "3": "NEURO_SYMBOLIC"
    }
    
    while True:
        choice = input(f"Enter your choice [1, 2, 3] or press Enter to accept recommendation: ").strip()
        if not choice:
            return recommended_strategy
        if choice in strategy_map:
            return strategy_map[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    print("🚀 LLM-DRIVEN MULTI-STRATEGY CODE GENERATION SYSTEM 🚀")
    print("=" * 70)
    
    task = input("📝 Enter your programming task (or press Enter for default): ")
    if not task.strip():
        task = "Create a python function that finds the longest palindromic substring in a given string."
    
    print(f"\n🎯 Task: {task}")
    print("=" * 70)
    
    # 1. SETUP: Create agents and tools
    all_agents = create_all_agents()
    user_proxy = all_agents["user_proxy"]
    user_proxy.register_function(function_map={"web_search": web_search})
    
    # 2. ANALYSIS: LLM analyzes the task to recommend a strategy
    print("1️⃣ ANALYZING TASK...")
    task_analyzer = LLMTaskAnalyzer(all_agents["task_analyzer"])
    analysis = task_analyzer.analyze_task(task, user_proxy)
    
    print("\n🧠 LLM ANALYSIS COMPLETE:")
    print(f"   - Recommended Strategy: {analysis.get('reasoning_strategy', 'N/A')}")
    print(f"   - Explanation: {analysis.get('explanation', 'N/A')}")
    
    # 3. ORCHESTRATION: User confirms or selects a reasoning pipeline
    recommended_strategy = analysis.get("reasoning_strategy", "CODE_FIRST").upper()
    chosen_strategy = get_user_choice(recommended_strategy)
    
    # 4. EXECUTION: Run the chosen pipeline
    pipelines = ReasoningPipelines(all_agents, user_proxy)
    pipeline_map = {
        "CODE_FIRST": pipelines.code_first_pipeline,
        "PSEUDOCODE_FIRST": pipelines.pseudocode_first_pipeline,
        "NEURO_SYMBOLIC": pipelines.neuro_symbolic_pipeline,
    }
    
    solution_generating_pipeline = pipeline_map[chosen_strategy]
    initial_solution = solution_generating_pipeline(task)
    
    # 5. TEST CASE GENERATION
    print("\n2️⃣ GENERATING COMPREHENSIVE TEST CASES...")
    tc_prompt = f"Generate comprehensive test cases as a JSON array for this task:\n{task}"
    tc_response = safe_initiate_chat(all_agents["testwriter"], tc_prompt, user_proxy)
    test_cases_str = extract_json_from_response(tc_response)
    print("📋 Generated Test Cases:\n", test_cases_str)
    
    # 6. VERIFICATION & CORRECTION LOOP
    print("\n3️⃣ VERIFICATION & CORRECTION CYCLE...")
    code_runner = PythonCodeRunner()
    code_extractor = CodeExtractor()
    final_code = None
    current_solution = initial_solution
    
    for attempt in range(3):
        print(f"\n🔄 Attempt {attempt + 1}/3")
        python_code = code_extractor.extract_python_code(current_solution)
        print("--- Code to Test ---\n", python_code, "\n--------------------")
        
        results = code_runner.run_code_with_tests(python_code, test_cases_str)
        
        if print_test_results(results):
            print("\n🎉 All tests passed! Solution verified.")
            final_code = python_code
            break
        
        print("\n🔧 Tests failed. Applying corrections...")
        correction_prompt = f"""Fix this code based on the test results.
Original Task: {task}
Current Code:
{python_code}
Test Results (JSON):
{json.dumps(results)}

Provide the full, corrected Python code wrapped in ```python ... ```."""
        current_solution = safe_initiate_chat(all_agents["corrector"], correction_prompt, user_proxy)
    else:
        print("\n❌ Could not fix the code after 3 attempts.")
        final_code = code_extractor.extract_python_code(current_solution)

    # 7. FINAL RESULTS
    print("\n" + "=" * 70)
    print("🎉 FINAL RESULTS")
    print("=" * 70)
    print(f"   - Strategy Used: [{chosen_strategy}]")
    print("\n✅ Final Verified Code:")
    print("-" * 40)
    print(final_code if final_code else "No verified solution was generated.")
    print("-" * 40)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()