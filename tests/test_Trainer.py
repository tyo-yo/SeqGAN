from tests.context import unittest, os, Agent, Environment, Discriminator, np
from tests.context import DiscriminatorGenerator
from tests.context import Trainer
top = os.getcwd()
# %load_ext autoreload
# %autoreload 2

class TestTrainer(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_trainer_pretraining(self):

        g_B, g_E, g_H, g_T = 32, 4, 4, 40
        d_B, d_E = 32, 4
        d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # filter sizes for CNNs
        d_num_filters = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10, 16, 16] # num of filters for CNNs
        d_dropout = 0.75
        n_sample=16

        trainer = Trainer(g_B, g_T, g_E, g_H,
            d_B, d_E, d_filter_sizes, d_num_filters, d_dropout, n_sample)

        g_path = os.path.join(top, 'tests', 'data', 'save', 'generator_pre.hdf5')
        d_path = os.path.join(top, 'tests', 'data', 'save', 'discriminator_pre.hdf5')

        trainer.pre_train(g_epochs=1, d_epochs=1, g_pre_path=g_path, d_pre_path=d_path)

        trainer.load_pre_train(g_path, d_path)

        x, y = trainer.d_data.next()
        pred = trainer.discriminator.predict(x)
        for i in range(d_B):
            txt = [trainer.g_data.id2word[id] for id in x[i].tolist()]
            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i,0], ''.join(txt)))
