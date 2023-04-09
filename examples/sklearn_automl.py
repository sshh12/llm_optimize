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
