from dataclasses import dataclass
from typing import Literal
import tyro
import numpy as np
from tabular.rl_models import (
    TabularActor,
    TabularMDP,
    TabularValueCritic,
    TabularQCritic,
)
from tabular.utils import seed_everything
from tabular.evaluation import evaluate
import logging
from tabular.environments import (
    LEGAL_FULL_MDP_ENV,
    setup_full_mdp_env,
    setup_env_wrappers,
)
from algorithms.policy_iteration import policy_iteration
from algorithms.value_iteration import value_iteration


@dataclass
class Config:
    env_id: LEGAL_FULL_MDP_ENV

    algorithm: Literal["policy_iteration", "value_iteration"]
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


def main() -> None:
    cfg = tyro.cli(Config)
    seed_everything(cfg.seed)

    env = setup_full_mdp_env(cfg.env_id, cfg.is_slippery)
    env = setup_env_wrappers(env, cfg.num_eval_ep)

    mdp = TabularMDP(env, cfg.discount_factor)
    actor = TabularActor(env.observation_space.n, env.action_space.n)
    value_critic = TabularValueCritic(env.observation_space.n)
    q_value_critic = TabularQCritic(env.observation_space.n, env.action_space.n)

    logging.info("Start learning.")
    match cfg.algorithm:
        case "policy_iteration":
            policy_iteration(mdp, value_critic, actor, cfg.stop_threshold)
        case "value_iteration":
            value_iteration(mdp, q_value_critic, actor, cfg.stop_threshold)

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
