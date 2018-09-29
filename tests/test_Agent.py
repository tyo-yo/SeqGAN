from .context import unittest, os, Agent, np

top = os.getcwd()

class TestAgent(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_agent(self):
        B, V, E, H, T = 32, 6000, 2, 3, 30
        agent = Agent(B, V, E, H)

        BOS = 1
        state = np.zeros([B, T], dtype=int)
        state[:, 0] = BOS

        cur_state = state[:, :1]
        a = agent.act(cur_state)
        self.sub_test(a.shape, (B, 1), msg='Agent.act output shape test, input shape=(B, 1)')

        cur_state = state[:, :3]
        a = agent.act(cur_state)
        self.sub_test(a.shape, (B, 1), msg='Agent.act output shape test, input_shape=(B, 3)')
