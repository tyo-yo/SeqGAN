from SeqGAN.models import Generator, GeneratorPretraining, Discriminator
from SeqGAN.utils import DiscriminatorGenerator
import numpy as np

class Agent(object):
    def __init__(self, B, V, E, H):
        '''
        # Arguments:
            B: int, batch_size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        '''
        self.num_actions = V
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.generator_pre = GeneratorPretraining(V, E, H)
        self.generator = Generator(V, E, H)
        self.reset_rnn_state()

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def evaluate(self, state, model=None):
        '''
        Evaluate next word probability by using observed state (word id).
        model should be stateful.
        # Arguments:
            state: numpy array, dtype=int, shape = (B, 1),
                state indicates current word.
        # Returns:
            probs: numpy array, dtype=float, shape = (B, V),
                probs are next word probabiliies.
        '''
        _model = model if model else self.generator
        probs, h, c = _model.predict([state, self.h, self.c])    # (B, V)
        self.set_rnn_state(h, c)    # Update states

        return probs

    def act(self, state, epsilon=0):
        '''
        # Arguments:
            state: numpy array, dtype=int, shape = (B, t)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        word = state[:, -1].reshape([-1, 1])
        return self._act_on_word(word, epsilon=epsilon)

    def _act_on_word(self, word, epsilon=0):
        '''
        # Arguments:
            word: numpy array, dtype=int, shape = (B, 1),
                word indicates current word.
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        action = None
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        else:
            probs = self.evaluate(word)
            action = self.sampling(probs)
        return action

    def sampling(self, probs):
        '''
        # Arguments:
            probs: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        assert probs.shape[0] == self.B, 'probs shape should be same as B'
        action = np.zeros((self.B, 1), dtype=np.int32)

        for i in range(self.B):
            p = probs[i]
            action[i, 0] = np.random.choice(self.num_actions, p=p)
        return action

class Environment(object):
    def __init__(self, discriminator, data_generator, g_beta):
        '''
        Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            data_generator: SeqGAN.models.DiscriminatorGenerator
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        '''
        self.data_generator = data_generator
        self.B = data_generator.B
        self.T = data_generator.T
        self.BOS = data_generator.BOS
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.reset()

    def reset(self):
        self.t = 0
        self.state = np.zeros(self.B, self.T)
        self.state[:, 0] = self.BOS
        self.state = np.array([[self.BOS] for i in range(self.B)])
        self.sentence = []

    def step(self):
        '''
        Step t -> t + 1 and returns a result of the Agent action.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)
        # Returns:
            next_state: numpy array, dtype=int, shape = (B, T)
            reward: numpy array, dtype=float, shape = (B, 1)
            is_episode_end: bool
            info: dict
        '''
        self.t = self.t + 1

        reward = self.Q()

        self.state[:, self.t] = action
        next_state = self.state
        is_episode_end = self.t >= self.T


    def render(self):
        pass

    def Q(self, action):
        '''
        State-Action value function using Rollout policy.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)
        # Returns:
            reward: numpy array, dtype=float, shape = (B, 1), State-Action value
        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        '''
        if t == self.T - 1:
            return XXX

        # Rollout
        Y = np.zeros(self.B, self.T)
        Y[:, :self.t - 1] = self.state[:, :self.t - 1]
        Y[:, self.t] = action[:, 0]

        state = action
        for t in range(self.t + 1, T):
            next_word = g_beta.act()
