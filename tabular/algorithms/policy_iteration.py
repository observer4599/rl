"""
This file implements the Policy Iteration algoirthm shown on Page 80 of
the book Reinforcement Learning, second edition by Sutton and Barto
"""

from dataclasses import dataclass
import tyro
import numpy as np
from tabular.components import TabularActor, TabularMDP, TabularValueCritic
from tabular.utils import seed_everything
from tabular.evaluation import evaluate
import logging
from tabular.environments import (
    LEGAL_FULL_MDP_ENV,
    setup_full_mdp_env,
    setup_env_wrappers,
)


@dataclass
class Config:
    seed: int = 8779
    env_id: LEGAL_FULL_MDP_ENV = "FrozenLake-v1"
    is_slippery: bool = False
    """Add stochastity to the environment."""
    discount_factor: float = 0.9
    """The discount factor often known as gamma."""
    stop_threshold: float = 1e-8
    """How much change in the value before stopping."""
    num_eval_ep: int = 100
    """Number of episode to evaluate in."""


def policy_evaluation(
    mdp: TabularMDP,
    value_critic: TabularValueCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:

    while True:
        delta: float = 0.0
        for state in mdp.get_states():
            value_estimate = 0.0
            for transition in mdp.get_transitions(state, actor.get_action(state)):
                value_estimate += transition.proba * (
                    transition.reward
                    + mdp.discount_factor
                    * value_critic.get_value(transition.next_state)
                )

            diff = value_critic.set_value(state, value_estimate)
            delta = max(delta, diff)

        if delta < stop_threshold:
            break


def policy_improvement(
    mdp: TabularMDP,
    value_critic: TabularValueCritic,
    actor: TabularActor,
) -> bool:
    policy_stable: bool = True

    for state in mdp.get_states():
        q_value_estimate = np.zeros(len(mdp.get_actions(state)))

        for action in mdp.get_actions(state):
            for transition in mdp.get_transitions(state, action):
                q_value_estimate[action] += transition.proba * (
                    transition.reward
                    + mdp.discount_factor
                    * value_critic.get_value(transition.next_state)
                )

        action_changed = actor.set_action(state, np.argmax(q_value_estimate))
        if action_changed:
            policy_stable = False

    return policy_stable


def policy_iteration(
    mdp: TabularMDP,
    value_critic: TabularValueCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:
    policy_stable: bool = False

    while not policy_stable:
        policy_evaluation(mdp, value_critic, actor, stop_threshold)
        policy_stable = policy_improvement(mdp, value_critic, actor)


def main() -> None:
    cfg = tyro.cli(Config)
    seed_everything(cfg.seed)

    env = setup_full_mdp_env(cfg.env_id, cfg.is_slippery)
    env = setup_env_wrappers(env, cfg.num_eval_ep)

    mdp = TabularMDP(env, cfg.discount_factor)
    actor = TabularActor(env.observation_space.n, env.action_space.n)
    value_critic = TabularValueCritic(env.observation_space.n)

    logging.info("Start learning.")
    policy_iteration(mdp, value_critic, actor, cfg.stop_threshold)

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
