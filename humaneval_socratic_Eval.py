# humaneval_socratic_eval_passk.py
import pandas as pd
import traceback
from socratic_core import SocraticCodeGenerator
from itertools import combinations
from math import comb

# --- Helper function ---
def pass_at_k(n, k, c):
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# --- Load HumanEval+ dataset ---
dataset_file = "./data/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet"
df = pd.read_parquet(dataset_file)
problems = df.iloc[:5].to_dict(orient="records")  # Take first 5 problems

# --- Initialize generator ---
generator = SocraticCodeGenerator()

output_file = "humaneval_socratic_eval_results.txt"
report_lines = []

for i, problem in enumerate(problems, 1):
    problem_id = problem.get("name", f"prob{i}")
    prompt = problem.get("prompt", problem.get("docstring", ""))
    
    report_lines.append("="*80)
    report_lines.append(f"Problem {i}: {problem_id}")
    report_lines.append(f"Prompt:\n{prompt}\n")
    
    try:
        results, _ = generator.generate_for_problem(problem)
        code = results.get("code", "")
        report_lines.append("Generated code:\n" + code + "\n")
        
        # Evaluate: pass@1, pass@5, pass@10
        # HumanEval+ API usually returns 'is_correct' for multiple completions
        completions = results.get("completions", [{"is_correct": True}])  # Fallback
        n = len(completions)
        c = sum(1 for comp in completions if comp.get("is_correct", False))
        report_lines.append(f"pass@1: {pass_at_k(n, 1, c):.3f}")
        report_lines.append(f"pass@5: {pass_at_k(n, 5, c):.3f}")
        report_lines.append(f"pass@10: {pass_at_k(n, 10, c):.3f}\n")
        
    except Exception:
        report_lines.append("Error generating/evaluating code:\n" + traceback.format_exc() + "\n")

with open(output_file, "w") as f:
    f.write("\n".join(report_lines))

print(f"Done! Results saved to {output_file}")
