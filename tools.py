import json
import time
import tracemalloc
from typing import List, Dict, Any
from duckduckgo_search import DDGS

class PythonCodeRunner:
    """Enhanced code runner supporting both normal test execution and genetic algorithm evaluation."""

    def __init__(self):
        self.namespace = {}
        self.execution_timeout = 5  # Prevent infinite loops in evolution

    def run_code_with_tests(self, python_code: str, test_cases) -> List[Dict[str, Any]]:
        """Execute code with basic pass/fail test cases."""
        results = []
        try:
            if isinstance(test_cases, str):
                test_data = json.loads(test_cases)
            else:
                test_data = test_cases
            if not isinstance(test_data, list):
                test_data = [test_data]
            exec_namespace = {}
            exec(python_code, exec_namespace)
            main_function = None
            for name, obj in exec_namespace.items():
                if callable(obj) and not name.startswith('_'):
                    main_function = obj
                    break
            if not main_function:
                return [{"error": "No callable function found in generated code", "passed": False}]
            for i, test_case in enumerate(test_data):
                try:
                    if isinstance(test_case, dict):
                        inputs = test_case.get("input", test_case.get("inputs", []))
                        expected = test_case.get("expected", test_case.get("output"))
                    else:
                        inputs = []
                        expected = test_case
                    if isinstance(inputs, list):
                        actual_output = main_function(*inputs)
                    elif isinstance(inputs, dict):
                        actual_output = main_function(**inputs)
                    else:
                        actual_output = main_function(inputs)
                    passed = actual_output == expected
                    results.append({
                        "test_id": i,
                        "passed": passed,
                        "input": inputs,
                        "expected": expected,
                        "actual": actual_output,
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "test_id": i,
                        "passed": False,
                        "input": locals().get('inputs', "Unknown"),
                        "expected": locals().get('expected', "Unknown"),
                        "actual": None,
                        "error": str(e)
                    })
            return results
        except Exception as e:
            return [{"error": f"Code execution failed: {str(e)}", "passed": False}]

    def run_code_with_fitness(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run code and return detailed fitness metrics for genetic evaluation."""
        results = {
            'passed_tests': 0,
            'total_tests': len(test_cases),
            'execution_time': 0,
            'memory_usage': 0,
            'syntax_valid': True,
            'test_details': []
        }

        # Check syntax first
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            results['syntax_valid'] = False
            results['error'] = str(e)
            return results

        for i, test_case in enumerate(test_cases):
            try:
                tracemalloc.start()
                start_time = time.time()

                basic_results = self.run_code_with_tests(code, [test_case])
                result = basic_results[0] if basic_results else {}

                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                execution_time = end_time - start_time
                results['execution_time'] += execution_time
                results['memory_usage'] = max(results['memory_usage'], peak)

                if result.get('passed', False):
                    results['passed_tests'] += 1

                results['test_details'].append({
                    'test_id': i,
                    'passed': result.get('passed', False),
                    'execution_time': execution_time,
                    'memory_used': peak,
                    'output': result.get('actual', ''),
                    'error': result.get('error', '')
                })

            except Exception as e:
                results['test_details'].append({
                    'test_id': i,
                    'passed': False,
                    'error': str(e)
                })

        return results


def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the top results."""
    print(f"🔎 Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            formatted_results = "\n\n".join([
                f"Title: {res['title']}\nSnippet: {res['body']}\nURL: {res['href']}" 
                for res in results
            ])
            return f"Search results for '{query}':\n\n{formatted_results}"
    except Exception as e:
        return f"An error occurred during web search: {e}"
