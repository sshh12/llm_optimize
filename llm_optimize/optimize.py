from typing import Callable, Optional, Tuple, List
import re

import openai
from langchain.input import print_text
from langchain.prompts.chat import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from llm_optimize import llm, constants

# The numeric score and the LLM-facing representation
ScoreTuple = Tuple[float, str]

# Best score, history of scores, best x0
OptimizationResultTuple = Tuple[float, List[float], str]


def run(
    task_description: str,
    task_question: str,
    func: Callable[[str], ScoreTuple],
    x0: str,
    max_steps: Optional[int] = 10,
    model: Optional[llm.LLMModel] = None,
    verbose: Optional[bool] = True,
    system_prompt: Optional[str] = constants.SYSTEM_PROMPT,
    human_prompt: Optional[str] = constants.HUMAN_OPTIMIZATION_PROMPT,
    stop_score: Optional[float] = None,
) -> OptimizationResultTuple:
    if model is None:
        model = llm.get_default_llm()

    def _log(text: str, color: str):
        if verbose:
            print_text(text + "\n", color)

    x = x0
    score, fx = func(x)
    best_score = score
    best_x = x

    _log(x, "blue")
    _log(fx, "green")

    messages = [
        SystemMessage(content=system_prompt.format(task_description=task_description)),
        HumanMessage(content=human_prompt.format(task_question=task_question, x=x, fx=fx)),
    ]
    score_hist = []

    for _ in range(max_steps):
        try:
            resp = model(messages).content
        except openai.error.InvalidRequestError as e:
            _log(str(e), "red")
            # drop the first set of results to reduce token usage
            messages.pop(1)
            messages.pop(1)
            resp = model(messages).content
        _log(resp, "yellow")
        try:
            x = re.findall("```(?:\w+)?([\s\S]+)```", resp)[0]
        except KeyError as e:
            _log(f"Stopping early, failed to parse response. {e}", "red")
            break
        _log(x, "blue")
        score, fx = func(x)
        score_hist.append(score)
        if score > best_score:
            best_x = x
            best_score = score
        _log(fx, "green")
        messages.append(AIMessage(content=resp))
        messages.append(HumanMessage(content=human_prompt.format(task_question=task_question, x=x, fx=fx)))
        if stop_score is not None and best_score >= stop_score:
            break

    return (best_score, score_hist, best_x)
