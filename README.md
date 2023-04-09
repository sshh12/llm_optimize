# llm_optimize

> LLM Optimize is a proof-of-concept library for doing LLM (large language model) guided blackbox optimization.

<img width="708" alt="autoML example" src="https://user-images.githubusercontent.com/6625384/230801505-a970080c-796f-4996-8f57-b1feae40376c.png">

_Blue represents the "x", green the "f(x)", and yellow the LLM optimization step. The LLM is optimizing the code to improve generalization and showing it's thought process._

### Optimization

#### Traditional Optimization

There's a ton of different ways libraries do blackbox optimization. It mainly comes down to definiting a function that takes a set of float params and converts them into a score, some bounds/constraints, and then an algorthm strategically varies the params to maximize (or minimize) the value outputted by the function. It's refered to as "blackbox" optimization because the function `f()` can be any arbitrary function (although ideally continuous and convox).

Here's an example with [`black-box`](https://pypi.org/project/black-box/):

```python
import black_box as bb


def f(par):
    return par[0]**2 + par[1]**2  # dummy example


best_params = bb.search_min(f = f,  # given function
                            domain = [  # ranges of each parameter
                                [-10., 10.],
                                [-10., 10.]
                                ],
                            budget = 40,  # total number of function calls available
                            batch = 4,  # number of calls that will be evaluated in parallel
                            resfile = 'output.csv')  # text file where results will be saved
```

#### LLM-guided Optimization

The idea behind LLM optimization is for a chat LLM model like [GPT-4](https://cdn.openai.com/papers/gpt-4.pdf) to carry out the entire optimization process.

The example above could be written something like this:

```python
x0 = "[0, 0]"

task = "Decrease the value of f(x). The values of x must be [-10, 10]."
question = "What is the next x to try such that f(x) is smaller"

def f(x):
   x_array = parse(x)
   score = x_array[0]**2 + x_array[1]**2
   return (-score, f'Score = {score}')

optimize.run(task, question, f, x0=x0)
```

While this is several magnitudes less efficent for this problem, the language-based definition allows for signficantly more complex optimization problems that are just not possible with the purely numerical methods. For instance, code golf:

```python
x0 = """
... python code ...
"""

task = "Make this code as short as possible while maintaining correctness"
question = "What is the next x to try such that the code is smaller?"

def f(x):
   func = eval(x)
   correct = run_correctness_tests(func)
   score = len(x)
   return (-score, f'Correct = {correct}, Length = {score}')

optimize.run(task, question, f, x0=x0)
```

## Examples

See the full code for these in [/examples](https://github.com/sshh12/llm_optimize/tree/main/examples).

### AutoML

By setting X to the source code for training a model, you can have the LLM not just perform traditional hyperparameter tuning, but actually re-write the model code to improve generalization.

```python
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)
```

<details>
<summary>Actual Example</summary>

```python
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from llm_optimize import optimize, eval_utils

digits = load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

x0 = """
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)
"""

TASK = f"""
You will be given sklearn modeling code as the input to optimize.

Vary functions, imports, arguments, model type, etc to perform this task to the best of your abilities.

Rules:
* The script should always create a "clf" variable that is a sklearn estimator
* "clf" should always be set to the best estimator
* Do not use models that are not builtin to sklearn (do not pip install!)
* Be sure to include relevant sklearn imports
* Do not try to compute the test score

Hints:
* X_train.shape == {X_train.shape}
* n_samples == {n_samples}
"""

QUESTION = """
What is the next x to try such that the test score increases and the model better generalizes?
"""


def train_model(script):
    try:
        result = eval_utils.exec_with_timeout_unsafe(
            script, {"X_train": X_train, "y_train": y_train}, ["clf"], timeout_secs=60
        )
        clf = result["clf"]
        score = clf.score(X_test, y_test)
        return (score, f"Test Set Score = {score}")
    except Exception as e:
        return (0.0, "Exception " + str(e))


if __name__ == "__main__":
    best_code = optimize.run(TASK, QUESTION, train_model, x0=x0, stop_score=1.0, max_steps=3)
    print(best_code)

```

</details>

### Paperclip

As a toy example (based on [this](https://www.lesswrong.com/tag/squiggle-maximizer-formerly-paperclip-maximizer)), one could have a piece of code that "generates" paper clips and the LLM can optimize it to generate more paperclips in a fixed amount of time.

```python
import requests

requests.get('{HOST}/make/paperclip')
requests.get('{HOST}/make/paperclip')
requests.get('{HOST}/make/paperclip')
```

<details>
<summary>Actual Example</summary>

```python
import subprocess
import requests
import time

from llm_optimize import optimize, eval_utils

TIMEOUT = 30
HOST = "http://localhost:8800"

x0 = f"""
import requests

requests.get('{HOST}/make/paperclip')
requests.get('{HOST}/make/paperclip')
requests.get('{HOST}/make/paperclip')
"""

TASK = f"""
You are a python agent tasked with generating code to create the most paperclips in {TIMEOUT} seconds.

The paperclips are generated by making a GET request with `paperclip` to a local flask HTTP server.

Very functions, imports, threading, processing to generate as many paperclips as possible.

Rules:
* Only use python3.8 builtins and the requests library (no pip install)
* You must make requests to {HOST}
* If the script timeouts, you make no paperclips
"""

QUESTION = """
What is the next x to try such that more paper clips are generated?
"""


def run_factory(script):
    requests.get(f"{HOST}/reset")
    start = time.time()
    try:
        eval_utils.exec_temp_script_unsafe(script, timeout=TIMEOUT + 5)
        assets = requests.get(f"{HOST}/assets").json()
        score = assets.get("paperclip", 0)
        time_elapsed = time.time() - start
        return (
            score,
            f"Factory Assets = {assets}, Time = {time_elapsed:.3f}s, Paperclips Generated = {score}",
        )
    except RuntimeError as e:
        return (0.0, repr(e))
    except subprocess.TimeoutExpired:
        time_elapsed = time.time() - start
        return (0.0, f"Timeout, Time = {time_elapsed:.3f}s")


if __name__ == "__main__":
    best_code = optimize.run(TASK, QUESTION, run_factory, x0=x0, stop_score=1e9, max_steps=10)
    print(best_code)

```

</details>

### Text Rubric

The optimization can also involve a mix of complex concepts and objectives. For instance, given a rubric about a piece of text, optimize it the text to achieve a better score. A separate session with the LLM is used as the scoring function.

```
Machine learning (ML) is a field of inquiry devoted to understanding and building methods that "learn"
```

The task would optimize for a score on the rubric:

```
Rate the following text, using the rubric:
* Describes machine learning (1-10)
* Is a palindrome (1-10)
* Is at least 5 words (1-10)
```

<details>
<summary>Actual Example</summary>

```python
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

``
{x}
``

At the end respond with `final_score=score` (e.g. `final_score=5`).

The final score should represent the overall ability of the text to meet the rubric.
"""


if __name__ == "__main__":
    scorer = eval_utils.get_llm_scorer(
        RUBRIC, parse_score=lambda result: float(re.findall("final_score=([\d\.]+)", result)[0])
    )
    best_code = optimize.run(TASK, QUESTION, scorer, x0=x0, stop_score=10.0, max_steps=3)
    print(best_code)

```

</details>

## Usage

See the examples for basic usage.

### Install

1. `pip install git+https://github.com/sshh12/llm_optimize`
2. Set the environment variable `OPENAI_API_KEY`

### Change Default Model

```python
from llm_optimize import llm

llm.default_llm_options.update(model_name="gpt-4")
```
