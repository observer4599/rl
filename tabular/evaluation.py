from tabular.agent import TabularActor
import gymnasium as gym


def evaluate(
    env: gym.Env, actor: TabularActor, num_eval_ep: int, seed: int
) -> None:
    observation, _ = env.reset(seed=seed)
    for _ in range(num_eval_ep):
        episode_done = False
        while not episode_done:
            action = actor.get_action(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated

        observation, _ = env.reset()
    env.close()
