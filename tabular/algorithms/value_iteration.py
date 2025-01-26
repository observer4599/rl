"""
This file implements the Value Iteration algoirthm shown on Page 83 of
the book Reinforcement Learning, second edition by Sutton and Barto
"""

from tabular.rl_models import (
    TabularActor,
    TabularMDP,
    TabularQCritic,
)


def policy_evaluation(
    mdp: TabularMDP,
    q_critic: TabularQCritic,
    actor: TabularActor,
    state: int,
    delta: float,
) -> float:
    for action in mdp.get_actions(state):
        q_value_estimate = 0

        for transition in mdp.get_transitions(state, action):
            q_value_estimate += transition.proba * (
                transition.reward
                + mdp.discount_factor
                * q_critic.get_value(
                    transition.next_state,
                    actor.get_action(transition.next_state),
                )
            )

        q_value_difference = q_critic.get_value_difference(
            state, action, q_value_estimate
        )
        if q_value_difference > 0:
            delta = max(delta, q_value_difference)
            q_critic.set_value(state, action, q_value_estimate)

    return delta


def policy_improvement(
    actor: TabularActor, q_critic: TabularQCritic, state: int
) -> None:
    if actor.is_action_different(state, q_critic.get_max_action(state)):
        actor.set_action(state, q_critic.get_max_action(state))


def value_iteration(
    mdp: TabularMDP,
    q_critic: TabularQCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:

    while True:
        delta: float = 0.0

        for state in mdp.get_states():
            delta = policy_evaluation(mdp, q_critic, actor, state, delta)
            policy_improvement(actor, q_critic, state)
        if delta < stop_threshold:
            break
