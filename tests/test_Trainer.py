from tests.context import unittest, os, np
from tests.context import Trainer
top = os.getcwd()

class TestTrainer(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_trainer(self):
        B, T = 32, 40
        g_E, g_H = 4, 4
        d_E = 4
        d_H = 64
        d_dropout = 0.75
        n_sample=16

        trainer = Trainer(B, T, g_E, g_H,
            d_E, d_H, d_dropout, n_sample)

        g_path = os.path.join(top, 'tests', 'data', 'save', 'generator_pre.hdf5')
        d_path = os.path.join(top, 'tests', 'data', 'save', 'discriminator_pre.hdf5')

        trainer.pre_train(g_epochs=1, d_epochs=1, g_pre_path=g_path, d_pre_path=d_path)

        trainer.load_pre_train(g_path, d_path)

        x, y = trainer.d_data.next()
        pred = trainer.discriminator.predict(x)
        for i in range(B):
            txt = [trainer.g_data.id2word[id] for id in x[i].tolist()]
            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i,0], ''.join(txt)))

        trainer.train(steps=1)
