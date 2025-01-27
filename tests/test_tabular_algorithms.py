from src.tabular.policy_learning import (
    Config,
    prepare_rl_setup,
    learn,
    TabularAlgorithm,
)
from src.tabular.evaluation import evaluate
from src.utils import seed_everything
import numpy as np


def test_policy_iteration():
    cfg = Config(
        env_id="FrozenLake-v1", algorithm=TabularAlgorithm.POLICY_ITERATION
    )
    seed_everything(cfg.seed)
    env, mdp, actor, critic = prepare_rl_setup(cfg)
    learn(cfg.algorithm, mdp, actor, critic)
    evaluate(env, actor, cfg.num_eval_ep, cfg.seed)

    avg_return = np.mean(env.return_queue)
    assert (
        avg_return >= 1.0
    ), f"The average return was {avg_return}, expected 1.0"
