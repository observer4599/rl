import numpy as np
from jaxtyping import Integer
from dataclasses import dataclass
import gymnasium as gym


def _valid_state(state: int, num_states: int) -> bool:
    return 0 <= state < num_states


def _valid_action(action: int, num_actions: int) -> bool:
    return 0 <= action < num_actions


class TabularActor:
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states = num_states
        self.num_actions = num_actions

        self.actions: Integer[np.ndarray, "states"] = np.ones(
            num_states, dtype=int
        )

    def get_action(self, state: int) -> int:
        assert _valid_state(
            state, self.num_states
        ), f"State `{state}` is invalid."
        return self.actions[state]

    def set_action(self, state: int, action: int) -> bool:
        assert _valid_state(
            state, self.num_states
        ), f"State `{state}` is invalid."
        assert _valid_action(
            action, self.num_actions
        ), f"Action `{action}` is invalid."

        if self.actions[state] == action:
            return False

        self.actions[state] = action
        return True


class TabularValueCritic:
    def __init__(self, num_states: int) -> None:
        self.num_states = num_states

        self.values: Integer[np.ndarray, "states"] = np.zeros(
            num_states, dtype=float
        )

    def get_value(self, state: int) -> float:
        assert _valid_state(
            state, self.num_states
        ), f"State `{state}` is invalid."
        return self.values[state]

    def set_value(self, state: int, value: float) -> float:
        assert _valid_state(
            state, self.num_states
        ), f"State `{state}` is invalid."
        diff = np.abs(self.values[state] - value)

        if diff == 0.0:
            return 0.0

        self.values[state] = value
        return diff


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
        assert state in self.get_states(), f"State `{state}` is invalid."
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
