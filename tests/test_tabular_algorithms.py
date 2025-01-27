from src.tabular.policy_learning import (
    Config,
    prepare_rl_setup,
    learn,
    TabularAlgorithm,
)
from src.tabular.evaluation import evaluate
from src.utils import seed_everything
import numpy as np


def compute_avg_return(cfg: Config) -> float:
    seed_everything(cfg.seed)
    env, mdp, actor, critic = prepare_rl_setup(cfg)
    learn(cfg.algorithm, mdp, actor, critic)
    evaluate(env, actor, cfg.num_eval_ep, cfg.seed)

    return np.mean(env.return_queue)


class TestPolicyIteration:
    def setup_method(self):
        self.seed: int = 8779
        self.num_eval_ep: int = 100

    def test_deterministic_frozen_lake(self):
        cfg = Config(
            env_id="FrozenLake-v1",
            algorithm=TabularAlgorithm.POLICY_ITERATION,
            seed=self.seed,
            is_slippery=False,
            num_eval_ep=self.num_eval_ep,
        )
        avg_return = compute_avg_return(cfg)
        expected_avg_return = 1.0
        assert (
            avg_return == expected_avg_return
        ), f"The average return was {avg_return}, expected {expected_avg_return}"

    def test_stochastic_frozen_lake(self):
        cfg = Config(
            env_id="FrozenLake-v1",
            algorithm=TabularAlgorithm.POLICY_ITERATION,
            seed=self.seed,
            is_slippery=True,
            num_eval_ep=self.num_eval_ep,
        )
        avg_return = compute_avg_return(cfg)
        expected_avg_return = 0.77
        assert (
            avg_return == expected_avg_return
        ), f"The average return was {avg_return}, expected {expected_avg_return}"

    def test_deterministic_cliff_walking(self):
        cfg = Config(
            env_id="CliffWalking-v0",
            algorithm=TabularAlgorithm.POLICY_ITERATION,
            seed=self.seed,
            is_slippery=False,
            num_eval_ep=self.num_eval_ep,
        )
        avg_return = compute_avg_return(cfg)
        expected_avg_return = -13.0
        assert (
            avg_return == expected_avg_return
        ), f"The average return was {avg_return}, expected {expected_avg_return}"

    def test_stochastic_cliff_walking(self):
        cfg = Config(
            env_id="CliffWalking-v0",
            algorithm=TabularAlgorithm.POLICY_ITERATION,
            seed=self.seed,
            is_slippery=True,
            num_eval_ep=self.num_eval_ep,
        )
        avg_return = compute_avg_return(cfg)
        expected_avg_return = -60.54
        assert (
            avg_return == expected_avg_return
        ), f"The average return was {avg_return}, expected {expected_avg_return}"

    def test_taxi(self):
        cfg = Config(
            env_id="Taxi-v3",
            algorithm=TabularAlgorithm.POLICY_ITERATION,
            seed=self.seed,
            num_eval_ep=self.num_eval_ep,
        )
        avg_return = compute_avg_return(cfg)
        expected_avg_return = 8.11
        assert (
            avg_return == expected_avg_return
        ), f"The average return was {avg_return}, expected {expected_avg_return}"
