"""
This file implements the Value Iteration algoirthm shown on Page 83 of
the book Reinforcement Learning, second edition by Sutton and Barto
"""

from tabular.agent import (
    TabularActor,
    TabularCritic,
)
from tabular.environments import TabularMDP


def policy_evaluation(
    mdp: TabularMDP,
    critic: TabularCritic,
    actor: TabularActor,
    state: int,
) -> float:
    delta: float = 0.0
    for action in mdp.get_actions(state):
        q_estimate: float = 0.0

        for transition in mdp.get_transitions(state, action):
            q_estimate += transition.proba * (
                transition.reward
                + mdp._discount_factor
                * critic.get_value(
                    transition.next_state,
                    actor.get_action(transition.next_state),
                )
            )

        q_value_difference = critic.get_value_difference(
            state, action, q_estimate
        )
        if q_value_difference > 0:
            delta = max(delta, q_value_difference)
            critic.set_value(state, action, q_estimate)

    return delta


def policy_improvement(
    actor: TabularActor, q_critic: TabularCritic, state: int
) -> None:
    if actor.is_action_different(state, q_critic.get_max_action(state)):
        actor.set_action(state, q_critic.get_max_action(state))


def value_iteration(
    mdp: TabularMDP,
    critic: TabularCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:

    while True:
        delta: float = 0.0

        for state in mdp.get_states():
            delta = max(delta, policy_evaluation(mdp, critic, actor, state))
            policy_improvement(actor, critic, state)
        if delta < stop_threshold:
            break
