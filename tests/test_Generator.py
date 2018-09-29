from tests.context import unittest, os, Generator, GeneratorPretrainingGenerator
from tests.context import np

top = os.getcwd()

class TestAgent(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator(self):
        B = 32
        E = 2
        H = 3
        gen = GeneratorPretrainingGenerator(
            path=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=B,
            shuffle=True)

        model = Generator(gen.V, E, H)
        lstm = model.get_layer('LSTM')
        init_h = np.zeros([B, H])
        init_c = np.zeros([B, H])


        print('Model: Generator')
        model.summary()

        BOS = gen.BOS
        x = [BOS] * B
        x = np.array(x).reshape(B, 1)

        pred, h, c = model.predict([x, init_h, init_c])

        self.sub_test(pred.shape, (B, gen.V), msg='output shape test')
        self.assertAlmostEqual(B, np.sum(pred), places=1, msg='softmax test')

        for i in range(100):
            pred2, h, c = model.predict([x, h, c])

        self.assertNotAlmostEqual(pred[0, 0], pred2[0, 0], places=10, msg='stateful test')
