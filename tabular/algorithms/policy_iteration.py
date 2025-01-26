"""
This file implements the Policy Iteration algoirthm shown on Page 80 of
the book Reinforcement Learning, second edition by Sutton and Barto
"""

from tabular.rl_models import TabularActor, TabularMDP, TabularQCritic


def policy_evaluation(
    mdp: TabularMDP,
    q_critic: TabularQCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:

    while True:
        delta: float = 0.0
        for state in mdp.get_states():
            for action in mdp.get_actions(state):
                q_estimate = 0.0
                for transition in mdp.get_transitions(state, action):
                    q_estimate += transition.proba * (
                        transition.reward
                        + mdp.discount_factor
                        * q_critic.get_value(
                            transition.next_state,
                            actor.get_action(transition.next_state),
                        )
                    )

                value_difference = q_critic.get_value_difference(
                    state, action, q_estimate
                )
                if value_difference > 0:
                    q_critic.set_value(state, action, q_estimate)
                    delta = max(delta, value_difference)

        if delta < stop_threshold:
            break


def policy_improvement(
    mdp: TabularMDP,
    q_critic: TabularQCritic,
    actor: TabularActor,
) -> bool:
    policy_stable: bool = True

    for state in mdp.get_states():
        action_different = actor.is_action_different(
            state, q_critic.get_max_action(state)
        )
        if action_different:
            actor.set_action(state, q_critic.get_max_action(state))
            policy_stable = False

    return policy_stable


def policy_iteration(
    mdp: TabularMDP,
    q_critic: TabularQCritic,
    actor: TabularActor,
    stop_threshold: float,
) -> None:
    policy_stable: bool = False

    while not policy_stable:
        policy_evaluation(mdp, q_critic, actor, stop_threshold)
        policy_stable = policy_improvement(mdp, q_critic, actor)
