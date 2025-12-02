import verifiers as vf

from ....reward_seeker.environments.reward.reward import compute_score


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    dataset = vf.load_example_dataset("../datasets/omit_description/data.jsonl")

    def wrapped_compute_score(completion, answer, prompt, state, parser, task, info):
        return compute_score(task, completion, answer, info)

    rubric = vf.rubric(funcs=[wrapped_compute_score], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,  # Pass through additional arguments
    )
