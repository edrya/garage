from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestVPG(TfGraphTestCase):
    def test_vpg_cartpole(self):
        """Test VPG with CartPole-v1 environment."""
        with LocalRunner(self.sess) as runner:
            env = TfEnv(env_name="CartPole-v1")

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(
                env=env,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                optimizer_args=dict(
                    tf_optimizer_args=dict(learning_rate=0.01, )))

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 90

            env.close()
