from .context import unittest, os, Generator, GeneratorPretrainingGenerator
from .context import np

top = os.getcwd()

class TestGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator(self):
        B = 32
        gen = GeneratorPretrainingGenerator(
            path=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=B,
            shuffle=True)
        model = Generator(B, gen.V, 5, 3)

        print('Model: Generator')
        model.summary()

        BOS = gen.BOS
        x = [BOS] * B
        x = np.array(x).reshape(B, 1)

        pred = model.predict(x)

        self.sub_test(pred.shape, (B, gen.V), msg='output shape test')
        self.assertAlmostEqual(B, np.sum(pred), places=3, msg='softmax test')

        for i in range(100):
            pred2 = model.predict(x)

        model.reset_states()
        pred3 = model.predict(x)

        self.assertAlmostEqual(pred[0, 0], pred3[0, 0], places=5, msg='reset state test')
        self.assertNotAlmostEqual(pred[0, 0], pred2[0, 0], places=7, msg='stateful test')
