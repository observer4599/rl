import numpy as np
from jaxtyping import Integer, Float
from dataclasses import dataclass
import gymnasium as gym


def _is_valid_state(state: int, num_states: int) -> None:
    if num_states <= state < 0:
        raise ValueError(f"State `{state}` is invalid.")


def _is_valid_action(action: int | np.integer, num_actions: int) -> None:
    if num_actions <= action < 0:
        raise ValueError(f"Action `{action}` is invalid.")


class TabularActor:
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

        self.actions: Integer[np.ndarray, "states"] = np.ones(
            num_states, dtype=int
        )

    def get_action(self, state: int) -> int:
        _is_valid_state(state, self.num_states)
        return self.actions[state]

    def set_action(self, state: int, action: int) -> bool:
        _is_valid_state(state, self.num_states)
        _is_valid_action(action, self.num_actions)

        self.actions[state] = action

    def is_action_different(self, state: int, action: int) -> bool:
        _is_valid_action(action, self.num_actions)
        return self.actions[state] != action


class TabularValueCritic:
    def __init__(self, num_states: int) -> None:
        self.num_states = num_states

        self.values: Float[np.ndarray, "states"] = np.zeros(
            num_states, dtype=float
        )

    def get_value(self, state: int) -> float:
        _is_valid_state(state, self.num_states)
        return self.values[state]

    def set_value(self, state: int, value_estimate: float) -> None:
        _is_valid_state(state, self.num_states)

        self.values[state] = value_estimate

    def get_value_difference(self, state: int, value_estimate: float) -> float:
        _is_valid_state(state, self.num_states)

        return np.abs(self.values[state] - value_estimate)


class TabularQCritic:
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

        self.q_values: Float[np.ndarray, "state action"] = np.zeros(
            (num_states, num_actions), dtype=float
        )

    def get_value(self, state: int, action: int) -> float:
        _is_valid_state(state, self.num_states)
        _is_valid_action(action, self.num_actions)

        return self.q_values[state, action]

    def set_value(self, state: int, action: int, value_estimate: float) -> None:
        _is_valid_state(state, self.num_states)
        _is_valid_action(action, self.num_actions)

        self.q_values[state, action] = value_estimate

    def get_value_difference(
        self, state: int, action: int, value_estimate: float
    ) -> float:
        _is_valid_state(state, self.num_states)
        _is_valid_action(action, self.num_actions)

        return np.abs(self.q_values[state, action] - value_estimate)


@dataclass
class Transition:
    state: int
    action: int
    proba: float
    reward: float
    next_state: int
    terminated: bool

    def __post_init__(self):
        if 1 < self.proba < 0:
            raise ValueError(
                f"Probability must be between 0 and 1, and not {self.proba}"
            )


class TabularMDP:
    def __init__(self, env: gym.Env, discount_factor: float) -> None:
        """
        dict[dict[list[tuple[float, int, float, bool]]]]
        The environment model is structured as follows:
        1. State
        2. Action
        3. A list of next states
        4. Each next state is a tuple of
        (transition probability, next state, reward, terminated)
        """
        assert hasattr(env.unwrapped, "P")
        self.model = env.unwrapped.P
        self.discount_factor = discount_factor

    def get_states(self) -> list[int]:
        return list(self.model.keys())

    def get_actions(self, state: int) -> list[int]:
        _is_valid_state(state, len(self.model))
        return list(self.model[state].keys())

    def get_transitions(self, state: int, action: int) -> list[Transition]:
        transitions = []
        for proba, next_state, reward, terminated in self.model[state][action]:
            if state == next_state and terminated:
                continue
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    proba=proba,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                )
            )

        return transitions
