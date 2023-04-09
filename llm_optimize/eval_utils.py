from typing import Dict, List
import time
import multiprocessing

_EXCEPTION = "_EXCEPTION"


def _exec_return_locals(script: str, return_dict: Dict, local_vars: Dict, return_local_vars: List[str]):
    locals().update(local_vars)
    try:
        exec(script)
    except Exception as e:
        return_dict["_EXCEPTION"] = e
    for var in return_local_vars:
        return_dict[var] = locals()[var]


def exec_with_timeout(script: str, local_vars: Dict, return_local_vars: List[str], timeout_secs: int) -> Dict:
    """
    Run `exec(script)` with some fancy features and data passing

    TODO: Sandbox this
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
