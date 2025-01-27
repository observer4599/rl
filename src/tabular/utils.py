import numpy as np


def is_valid_state(state: int, num_states: int) -> None:
    if num_states <= state < 0:
        raise ValueError(f"State `{state}` is invalid.")


def is_valid_action(action: int | np.integer, num_actions: int) -> None:
    if num_actions <= action < 0:
        raise ValueError(f"Action `{action}` is invalid.")
