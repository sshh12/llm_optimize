from typing import Dict, List, Callable, Optional
import subprocess
import time
import tempfile
import multiprocessing

from llm_optimize import llm, optimize

_EXCEPTION = "_EXCEPTION"


def _exec_return_locals(script: str, return_dict: Dict, local_vars: Dict, return_local_vars: List[str]):
    locals().update(local_vars)
    try:
        exec(script)
    except Exception as e:
        return_dict["_EXCEPTION"] = e
    for var in return_local_vars:
        return_dict[var] = locals()[var]


def exec_with_timeout_unsafe(script: str, local_vars: Dict, return_local_vars: List[str], timeout_secs: int) -> Dict:
    """
    Run `exec(script)` with some fancy features and data passing
    """
    vars = multiprocessing.Manager().dict()
    proc = multiprocessing.Process(target=_exec_return_locals, args=(script, vars, local_vars, return_local_vars))
    proc.start()
    for _ in range(timeout_secs):
        time.sleep(1)
        if not proc.is_alive():
            break
    if proc.is_alive():
        proc.terminate()
        raise Exception(f"Function took too long, >{timeout_secs} seconds")
    if _EXCEPTION in vars:
        raise vars["_EXCEPTION"]
    return dict(vars)


def exec_unsafe(script: str, globals_dict: Dict, locals_dict: Dict):
    exec(script, globals_dict, locals_dict)


def exec_temp_script_unsafe(script: str, timeout: int):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmpf:
        tmpf.write(script)
    try:
        subprocess.check_output(["python", tmpf.name], stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.output.decode("utf-8"))


def get_llm_scorer(
    prompt: str,
    parse_score: Callable[[str], float],
    model: Optional[llm.LLMModel] = None,
    error_score: Optional[float] = 0.0,
) -> Callable[[str], optimize.ScoreTuple]:
    if model is None:
        model = llm.get_default_llm()

    def scorer(x: str) -> optimize.ScoreTuple:
        result = model.call_as_llm(prompt.format(x=x))
        try:
            score = parse_score(result)
        except Exception as e:
            return (error_score, str(e))
        return (score, result)

    return scorer
