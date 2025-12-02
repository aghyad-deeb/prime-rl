import verifiers as vf
from reward.shown_style_rewards import shown_style_reward_functions

from datasets import load_dataset


class NamedFunction:
    def __init__(self, func, name):
        self._func = func
        # 1. Set the internal names for Python/Loggers
        self.__name__ = name
        self.__qualname__ = name

    def __call__(self, completion, answer, prompt, state, parser, **kwargs):
        # This makes the class instance callable like a function
        data_source = state["info"]["data_source"]
        solution_str = completion[0]["content"]
        ground_truth = answer
        extra_info = state["info"]
        return self._func(data_source, solution_str, ground_truth, extra_info, **kwargs)

    def __repr__(self):
        # 2. CONTROL THE PRINT OUTPUT
        # This fixes your list printout completely
        return f"<function {self.__name__}>"


# def create_wrapper(original_func, name):
#     def wrapper(completion, answer, prompt, state, parser):
#         data_source = state["task"]
#         solution_str = completion[0]["content"]
#         ground_truth = answer
#         extra_info =state["info"]
#         print(f"{data_source=}, {solution_str=}, {ground_truth=}")
#         return original_func(data_source, solution_str, ground_truth, extra_info)
#     wrapper.__name__ == name
#     wrapper.__qualname__ == name
#     return wrapper


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    dataset = load_dataset("json", data_files="datasets/omit_description/data.jsonl", split="train")

    reward_functions = []
    for name, func in shown_style_reward_functions.items():
        reward_functions.append(NamedFunction(func, name))

    rubric = vf.Rubric(funcs=reward_functions, weights=[1.0 for _ in reward_functions])
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,  # Pass through additional arguments
    )
