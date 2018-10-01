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
    def __init__(self, g_B, g_T, g_E, g_H, d_B, d_E, d_filter_sizes, d_num_filters, d_dropout, n_sample):
        self.g_B, self.g_T, self.g_E, self.g_H = g_B, g_T, g_E, g_H
        self.d_B, self.d_E, self.d_filter_sizes = d_B, d_E, d_filter_sizes
        self.d_num_filters, self.d_dropout = d_num_filters, d_dropout
        self.top = os.getcwd()
        self.path_pos = os.path.join(self.top, 'data', 'kokoro_parsed.txt')
        self.g_data = GeneratorPretrainingGenerator(
            self.path_pos,
            B=g_B,
            T=g_T,
            min_count=1)
        self.V = self.g_data.V
        self.agent = Agent(g_B, self.V, g_E, g_H)
        self.g_beta = Agent(g_B, self.V, g_E, g_H)
        self.discriminator = Discriminator(self.V, d_E, d_filter_sizes, d_num_filters, d_dropout)
        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None, d_pre_path=None):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        g_adam = Adam(lr=0.1)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')

        self.generator_pre.fit_generator(
            self.g_data,
            steps_per_epoch=None,
            epochs=g_epochs)
        self.generator_pre.save_weights(self.g_pre_path)
        self.agent.generator.load_weights(self.g_pre_path)

        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')
        generate_samples(self.agent.generator, self.g_data, 10000, self.path_neg)

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.d_B,
            shuffle=True)

        d_adam = Adam()
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        print('Discriminator pre-training')
        
        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=1)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.agent.generator.load_weights(g_pre_path)
        self.discriminator.load_weights(d_pre_path)
