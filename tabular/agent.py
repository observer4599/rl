import numpy as np
from jaxtyping import Integer, Float
from utils import is_valid_state, is_valid_action


class TabularActor:
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

        self.actions: Integer[np.ndarray, "states"] = np.ones(
            num_states, dtype=int
        )

    def get_action(self, state: int) -> int:
        is_valid_state(state, self.num_states)
        return self.actions[state]

    def set_action(self, state: int, action: int | np.integer) -> None:
        is_valid_state(state, self.num_states)
        is_valid_action(action, self.num_actions)

        self.actions[state] = action

    def is_action_different(self, state: int, action: int | np.integer) -> bool:
        is_valid_action(action, self.num_actions)
        return self.actions[state] != action


class TabularCritic:
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

        self.q_values: Float[np.ndarray, "state action"] = np.zeros(
            (num_states, num_actions), dtype=float
        )

    def get_max_action(self, state: int) -> np.integer:
        is_valid_state(state, self.num_states)
        return np.argmax(self.q_values[state])

    def get_value(self, state: int, action: int) -> float:
        is_valid_state(state, self.num_states)
        is_valid_action(action, self.num_actions)

        return self.q_values[state, action]

    def set_value(self, state: int, action: int, value_estimate: float) -> None:
        is_valid_state(state, self.num_states)
        is_valid_action(action, self.num_actions)

        self.q_values[state, action] = value_estimate

    def get_value_difference(
        self, state: int, action: int, value_estimate: float
    ) -> float:
        is_valid_state(state, self.num_states)
        is_valid_action(action, self.num_actions)

        return np.abs(self.q_values[state, action] - value_estimate)
