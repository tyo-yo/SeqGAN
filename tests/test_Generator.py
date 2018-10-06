from tests.context import unittest, os, Generator, np
from tests.context import tf, K, GeneratorPretrainingGenerator

sess = tf.Session()
K.set_session(sess)
top = os.getcwd()

class TestGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator(self):
        B = 4
        E = 2
        H = 3
        V = 5
        generator = Generator(sess, B, V, E, H)
        BOS = 1
        x = [BOS] * B
        x = np.array(x).reshape(B, 1)

        prob = generator.predict(x)

        self.sub_test(prob.shape, (B, V), msg='output shape test')
        self.assertAlmostEqual(B, np.sum(prob), places=1, msg='softmax test')

        for i in range(100):
            prob2 = generator.predict(x)

        generator.reset_rnn_state()
        prob3 = generator.predict(x)

        self.assertNotAlmostEqual(prob[0, 0], prob2[0, 0], places=10, msg='stateful test')
        self.assertAlmostEqual(prob[0, 0], prob3[0, 0], places=7, msg='stateful test')

        action = np.array([1, 2, 3, 4])
        reward = np.array([0.1, 0, 0.1, 0.8]).reshape(4,1)
        loss = generator.update(x, action, reward)
        for i in range(500):
            generator.reset_rnn_state()
            loss = generator.update(x, action, reward)
            if i % 100 == 0:
                generator.reset_rnn_state()
                prob = generator.predict(x)
                print(prob[0])
        self.sub_test(np.argmax(prob[0]), 4, 'RL optimization test')

        g_data = GeneratorPretrainingGenerator(
            os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=B,
            shuffle=False)
        T = 40
        num = 100
        output_file = os.path.join(top, 'tests', 'data', 'save', 'generated.txt')
        generator = Generator(sess, B, g_data.V, E, H)
        generator.generate_samples(T, g_data, num, output_file)
