from .context import unittest, os, Agent, np

top = os.getcwd()

class TestAgent(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator(self):
        B, V, E, H = 32, 6000, 2, 3
        agent = Agent(B, V, E, H)

        BOS = 1
        x = [BOS] * B
        x = np.array(x).reshape(B, 1)

        a = agent.act(x)
        self.sub_test(a.shape, (B, 1), msg='Agent.act output shape test')
