from dataclasses import dataclass
import tyro
import numpy as np
from src.tabular.agent import (
    TabularActor,
    TabularCritic,
)
from src.tabular.environments import TabularMDP
from src.utils import seed_everything
from src.tabular.evaluation import evaluate
import logging
from src.tabular.environments import (
    LEGAL_FULL_MDP_ENV,
    prepare_env,
)
from src.tabular.algorithms.policy_iteration import policy_iteration
from src.tabular.algorithms.value_iteration import value_iteration
from enum import Enum, auto
import gymnasium as gym


class TabularAlgorithm(Enum):
    POLICY_ITERATION = auto()
    VALUE_ITERATION = auto()


@dataclass
class Config:
    env_id: LEGAL_FULL_MDP_ENV
    """The Gymnasium environment ID."""
    algorithm: TabularAlgorithm
    """The RL algorithm to run."""
    seed: int = 8779
    """Seed for reproducibility."""
    is_slippery: bool = False
    """Add stochastity to the environment."""
    discount_factor: float = 0.9
    """The discount factor often known as gamma."""
    stop_threshold: float = 1e-8
    """How much change in the value before stopping."""
    num_eval_ep: int = 100
    """Number of episode to evaluate in."""


def prepare_rl_setup(
    cfg: Config,
) -> tuple[gym.Env, TabularMDP, TabularActor, TabularCritic]:
    env = prepare_env(cfg.env_id, cfg.num_eval_ep, cfg.is_slippery)
    mdp = TabularMDP(env, cfg.discount_factor)
    actor = TabularActor(env.observation_space.n, env.action_space.n)
    critic = TabularCritic(env.observation_space.n, env.action_space.n)

    return env, mdp, actor, critic


def learn(
    algorithm: TabularAlgorithm,
    mdp: TabularMDP,
    actor: TabularActor,
    critic: TabularCritic,
) -> None:
    match algorithm:
        case TabularAlgorithm.POLICY_ITERATION:
            policy_iteration(mdp, critic, actor)
        case TabularAlgorithm.VALUE_ITERATION:
            value_iteration(mdp, critic, actor)


def main() -> None:
    cfg = tyro.cli(Config)
    seed_everything(cfg.seed)

    env, mdp, actor, critic = prepare_rl_setup(cfg)

    logging.info("Start learning.")
    learn(cfg.algorithm, mdp, actor, critic)

    logging.info("Start evaluation.")
    evaluate(env, actor, cfg.num_eval_ep, cfg.seed)

    logging.info(
        f"Return: {np.mean(env.return_queue):.3f} ± {np.std(env.return_queue):.5f} over {len(env.return_queue)} episodes."
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="{levelname}:{name}:{message}", style="{", level=logging.INFO
    )
    main()
