from .context import unittest, os, DiscriminatorGenerator, DiscriminatorConv

top = os.getcwd()

class TestDiscriminator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_discriminator(self):
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # filter sizes for CNNs
        num_filters = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10, 16, 16] # num of filters for CNNs
        gen = DiscriminatorGenerator(
            path_pos=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            path_neg=os.path.join(top, 'tests', 'data', 'sample_generated.txt'),
            B=32,
            shuffle=True)
        model = DiscriminatorConv(gen.V, 3, filter_sizes, num_filters, dropout=0.75)
        model.compile('adam','binary_crossentropy')
        print('Model: Discriminator')
        model.summary()
        model.fit_generator(
            gen,
            steps_per_epoch=3,
            validation_data=gen,
            validation_steps=2)
        pred = model.predict_generator(gen, steps=1)
        self.sub_test(pred.shape, (32, 1), msg='output shape test')
