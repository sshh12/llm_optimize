import math
import time

# Tools
import numpy as np
import bisect

from llm_optimize import optimize, eval_utils

x0 = f"""
def is_prime(n):
  for i in range(2, n):
    if n % i == 0:
      return False
  return True
"""

TASK = f"""
You are a python agent tasked with generating efficient code.

You are tasked with minimizing the time it takes to compute is_prime for an unknown input.

Rules:
* Only use python3.8 builtins and the numpy library (no pip install)
* The script must create a "is_prime" function that takes a single value n as input
* Any additional variables or helper functions must be defined within the is_prime function scope
"""

QUESTION = """
What is the next x to try such that the function is even faster and log time decreases? 
"""


def run_code(script):
    try:
        eval_utils.exec_unsafe("global is_prime\n\n" + script, globals(), {})
        start = time.time()
        for _ in range(10000):
            assert globals()["is_prime"](21269), "Computed value is incorrect"
            assert not globals()["is_prime"](21270), "Computed value is incorrect"
        time_elapsed = time.time() - start
        time_log = math.log10(time_elapsed)
        return (
            -time_log,
            f"Time = {time_elapsed:.5f}s, Time (log10) = {time_log:.5f}",
        )
    except Exception as e:
        return (-10.0, "Exception " + str(e))


if __name__ == "__main__":
    best_code = optimize.run(TASK, QUESTION, run_code, x0=x0, stop_score=10.0, max_steps=3)
    print(best_code)
