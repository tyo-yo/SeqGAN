from .context import unittest, os, GeneratorPretrainingGenerator, GeneratorPretraining
# from tests.context import unittest, os, GeneratorPretrainingGenerator, GeneratorPretraining

top = os.getcwd()

class TestGeneratorPretraining(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator_pretraining_generator(self):
        gen = GeneratorPretrainingGenerator(
            os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=32, T=40,
            shuffle=False)
        model = GeneratorPretraining(gen.V, 5, 3)
        model.compile('adam','categorical_crossentropy')
        model.fit_generator(
            gen,
            steps_per_epoch=3)
            # validation_data=gen,
            # validation_steps=2)
        pred = model.predict_generator(gen, steps=4)
        pred.shape
        model.summary()
        x, y_true = gen.next()
        y_true.shape
