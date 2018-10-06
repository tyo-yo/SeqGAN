from SeqGAN.models import GeneratorPretraining, Discriminator, Generator
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, B, T, g_E, g_H, d_E, d_filter_sizes, d_num_filters,
        d_dropout, g_lr=1e-3, d_lr=1e-3, n_sample=16, generate_samples=10000):
        self.B, self.T = B, T
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_filter_sizes = d_E, d_filter_sizes
        self.d_num_filters, self.d_dropout = d_num_filters, d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr

        self.top = os.getcwd()
        self.path_pos = os.path.join(self.top, 'data', 'kokoro_parsed.txt')
        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')
        self.g_data = GeneratorPretrainingGenerator(
            self.path_pos,
            B=B,
            T=T,
            min_count=1)
        self.V = self.g_data.V
        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.Beta = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.discriminator = Discriminator(self.V, d_E, d_filter_sizes, d_num_filters, d_dropout)
        self.env = Environment(self.discriminator, self.g_data, self.Beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None ,d_pre_path=None,
        g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)
        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()

        self.generator_pre.fit_generator(
            self.g_data,
            steps_per_epoch=None,
            epochs=g_epochs)
        self.generator_pre.save_weights(self.g_pre_path)
        self._reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.T, self.g_data,
            self.generate_samples, self.path_neg)

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.B,
            shuffle=True)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')

        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=1)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self._reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)

    def _reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=1, verbose=True):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')

        for step in range(steps):
            # Generator training
            for _ in range(g_steps):
                rewards = np.zeros([self.B, self.T])
                self.agent.reset()
                for t in range(self.T):
                    state = self.env.get_state()
                    action = self.agent.act(state, epsilon=0.1)
                    next_state, reward, is_episode_end, info = self.env.step(action)
                    self.agent.generator.update(state, action, reward)
                    rewards[:, t] = reward.reshape([self.B, ])
                    # self.env.render(head=1)
                    if is_episode_end:
                        if verbose:
                            self.env.render(head=1)
                            print('Reward: {:.3f}, Episode end'.format(reward[0, 0]))
                        break
            # Discriminator training
                for _ in range(d_steps):
                    self.agent.generator.generate_samples(
                        self.T,
                        self.g_data,
                        self.generate_samples,
                        self.path_neg)
                    self.d_data = DiscriminatorGenerator(
                        path_pos=self.path_pos,
                        path_neg=self.path_neg,
                        B=self.B,
                        shuffle=True)
                    self.discriminator.fit_generator(
                        self.d_data,
                        steps_per_epoch=None,
                        epochs=d_epochs)
            # Update env.g_beta to agent

    def save(self, g_path, d_path):
        # self.agent.save(g_path)
        self.discriminator.save(d_path)
