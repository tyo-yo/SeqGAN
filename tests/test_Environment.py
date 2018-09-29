from tests.context import unittest, os, Agent, Environment, Discriminator, np
from tests.context import DiscriminatorGenerator

top = os.getcwd()

class TestEnvironment(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator(self):
        B, V, E, H, T = 32, 6000, 2, 3, 20
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # filter sizes for CNNs
        num_filters = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10, 16, 16] # num of filters for CNNs

        agent = Agent(B, V, E, H)
        g_beta = Agent(B, V, E, H)
        discriminator = Discriminator(V, E, filter_sizes, num_filters, 0.7)
        gen = DiscriminatorGenerator(
            path_pos=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            path_neg=os.path.join(top, 'tests', 'data', 'sample_generated.txt'),
            B=32,
            T=T,
            shuffle=True)

        env = Environment(discriminator, gen, g_beta, n_sample=4)
        state = env.state
        action = agent.act(state, epsilon=0.0)
        next_state, reward, is_episode_end, info = env.step(action)
        env.reset()

        for i in range(100):
            state = env.state
            action = agent.act(state, epsilon=0.1)
            next_state, reward, is_episode_end, info = env.step(action)
            if is_episode_end:
                env.render(head=3)
                print('Episode end')
                break

        self.sub_test(env.state.shape, (B, T), msg='Env.step shape')
        self.sub_test(reward.shape, (B, 1), msg='Env.reward shape')
