import gymnasium as gym
from typing import Literal, get_args
import logging

LEGAL_FULL_MDP_ENV = Literal["FrozenLake-v1", "CliffWalking-v0", "Taxi-v3"]


def setup_full_mdp_env(env_id: gym.Env, is_slippery: bool) -> gym.Env:
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
