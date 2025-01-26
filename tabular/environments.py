import gymnasium as gym
from typing import Literal, get_args
import logging
from dataclasses import dataclass
from utils import is_valid_state
from jaxtyping import Float
import numpy as np

LEGAL_FULL_MDP_ENV = Literal["FrozenLake-v1", "CliffWalking-v0", "Taxi-v3"]


def setup_full_mdp_env(env_id: str, is_slippery: bool) -> gym.Env:
    assert env_id in get_args(
        LEGAL_FULL_MDP_ENV
    ), f"env_id: {env_id} is invalid, choose from {LEGAL_FULL_MDP_ENV}."

    if env_id == "Taxi-v3":
        env = gym.make(
            env_id,
            render_mode="rgb_array",
        )
        logging.warning(f"The argument `is_slippery` is not used in {env_id}.")
    else:
        env = gym.make(
            env_id,
            is_slippery=is_slippery,
            render_mode="rgb_array",
        )

    return env


def setup_env_wrappers(env: gym.Env, num_eval_ep: int) -> gym.Env:
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_eval_ep)
    return env


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
        assert hasattr(env.unwrapped, "P")

        Transitions = list[tuple[float, int, float, bool]]
        Action = dict[int, Transitions]
        State = dict[int, Action]

        self._model: State = env.unwrapped.P
        self._discount_factor = discount_factor

    @property
    def num_states(self) -> int:
        return len(self._model)

    def get_states(self) -> list[int]:
        return list(self._model.keys())

    def get_actions(self, state: int) -> list[int]:
        is_valid_state(state, self.num_states)
        return list(self._model[state].keys())

    def get_transitions(self, state: int, action: int) -> list[Transition]:
        transitions = []
        for proba, next_state, reward, terminated in self._model[state][action]:
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

    def sample_state(
        self, proba: Float[np.ndarray, "state"] | None = None
    ) -> np.integer:
        return np.random.choice(self.num_states, p=proba)

    def sample_action(
        self,
        state: int | np.integer,
        proba: Float[np.ndarray, "state"] | None = None,
    ) -> np.integer:
        return np.random(np.array(self.get_actions(state)), p=proba)
