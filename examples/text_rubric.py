import re

from llm_optimize import optimize, eval_utils

x0 = f"""
Machine learning (ML) is a field of inquiry devoted to understanding and building methods that "learn"
"""

TASK = f"""
You are a linguistics expert who can write complex sentences.

You are tasked with writing a statement that:
* Describes machine learning
* Is a palindrome
* Is at least 5 words
"""

QUESTION = """
What is the next x to try such that the text better describes machine learning and is a palindrome? 
"""

RUBRIC = """
Rate the following text, using the rubric:
* Describes machine learning (1-10)
* Is a palindrome (1-10)
* Is at least 5 words (1-10)

```
{x}
```

At the end respond with `final_score=score` (e.g. `final_score=5`). 

The final score should represent the overall ability of the text to meet the rubric.
"""


if __name__ == "__main__":
    scorer = eval_utils.get_llm_scorer(
        RUBRIC, parse_score=lambda result: float(re.findall("final_score=([\d\.]+)", result)[0])
    )
    best_code = optimize.run(TASK, QUESTION, scorer, x0=x0, stop_score=10.0, max_steps=3)
    print(best_code)
