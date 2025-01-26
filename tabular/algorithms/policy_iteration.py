"""
This file implements the Policy Iteration algoirthm shown on Page 80 of
the book Reinforcement Learning, second edition by Sutton and Barto
"""

import numpy as np
from tabular.rl_models import TabularActor, TabularMDP, TabularValueCritic


def policy_evaluation(
    mdp: TabularMDP,
    value_critic: TabularValueCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> bool:

    iter_counter = 0
    while True:
        delta: float = 0.0
        for state in mdp.get_states():
            value_estimate = 0.0
            for transition in mdp.get_transitions(
                state, actor.get_action(state)
            ):
                value_estimate += transition.proba * (
                    transition.reward
                    + mdp.discount_factor
                    * value_critic.get_value(transition.next_state)
                )

            value_difference = value_critic.get_value_difference(
                state, value_estimate
            )
            if value_difference > 0:
                value_critic.set_value(state, value_estimate)
                delta = max(delta, value_difference)

        if delta < stop_threshold:
            # If the value function did not change, consider that
            # the polcy has converged.
            # Exercise 4.4
            if iter_counter == 0:
                return True
            return False

        iter_counter += 1


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

        action_different = actor.is_action_different(
            state, np.argmax(q_value_estimate)
        )
        if action_different:
            actor.set_action(state, np.argmax(q_value_estimate))
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
        policy_stable = policy_stable or policy_evaluation(
            mdp, value_critic, actor, stop_threshold
        )
        policy_stable = policy_stable or policy_improvement(
            mdp, value_critic, actor
        )
