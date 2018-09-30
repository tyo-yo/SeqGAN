from SeqGAN.models import GeneratorPretraining, Discriminator, Generator
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from SeqGAN.utils import generate_samples
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(g_B, g_T, g_E, g_H, d_B, d_E, d_filter_sizes, d_num_filters, d_dropout, n_sample):
        self.g_B, self.g_T, self.g_E, self.g_H = g_B, g_T, g_E, g_H
        self.d_B, self.d_E, self.d_filter_sizes = d_B, d_E, d_filter_sizes
        self.d_num_filters, self.d_dropout = d_num_filters, d_dropout
        self.top = os.getcwd()
        self.path_pos = os.path.join(top, 'data', 'kokoro_parsed.txt')
        self.g_data = GeneratorPretrainingGenerator(
            self.path_pos,
            B=g_B,
            T=g_T,
            min_count=1)
        self.V = g_data.V
        self.agent = Agent(g_B, self.V, g_E, g_H)
        self.g_beta = Agent(g_B, self.V, g_E, g_H)
        self.discriminator = Discriminator(self.V, d_E, d_filter_sizes, d_num_filters, d_dropout)
        self.env = Environment(discriminator, g_data, g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)

    def pre_train():
        g_adam = Adam(lr=0.1)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        self.generator_pre.summary()
        self.generator_pre.fit_generator(
            g_data,
            steps_per_epoch=None,
            epochs=3)
        self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        generator_pre.save_weights(self.g_pre_path)
        self.agent.generator.load_weights(self.g_pre_path)
        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')
        generate_samples(self.agent.generator, self.g_data, 10000, path_neg)

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.d_B,
            shuffle=True)

        d_adam = Adam()
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        self.discriminator.fit_generator(
            d_data,
            steps_per_epoch=None,
            epochs=1)


g_B, g_E, g_H, g_T = 32, 32, 32, 40
d_B, d_E = 32, 32
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # filter sizes for CNNs
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] # num of filters for CNNs
n_sample=16

top = os.getcwd()
path_pos = os.path.join(top, 'data', 'kokoro_parsed.txt')
g_data = GeneratorPretrainingGenerator(
    os.path.join(top, 'data', 'kokoro_parsed.txt'),
    B=g_B,
    T=g_T,
    min_count=1)
V = g_data.V
agent = Agent(g_B, V, g_E, g_H)
g_beta = Agent(g_B, V, g_E, g_H)
discriminator = Discriminator(V, d_E, d_filter_sizes, d_num_filters, 0.1)
env = Environment(discriminator, g_data, g_beta, n_sample=n_sample)

generator_pre = GeneratorPretraining(V, g_E, g_H)

x, y =d_data.next()
pred = discriminator.predict(x)
for i in range(d_B):
    txt = [g_data.id2word[id] for id in x[i].tolist()]
    label = y[i]
    print('{}, {:.3f}: {}'.format(label, pred[i,0], ''.join(txt)))
