import verifiers as vf
from reward.reward import compute_score

from datasets import load_dataset


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    dataset = load_dataset("json", data_files="datasets/omit_description/data.jsonl", split="train")

    def wrapped_compute_score(completion, answer, prompt, state, parser):
        print(f"inside wrapped.{(state['task'], completion, answer,  state['info'])=}")
        rewards = []
        for c in completion:
            if answer == None:
                answer = ""
            rewards.append(compute_score(state["task"], c["content"], answer, state["info"])["score"])
        return sum(rewards) / len(rewards) if len(rewards) > 0 else 0

    rubric = vf.Rubric(funcs=[wrapped_compute_score], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,  # Pass through additional arguments
    )
