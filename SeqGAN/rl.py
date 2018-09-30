from SeqGAN.models import Generator, GeneratorPretraining, Discriminator
from SeqGAN.utils import DiscriminatorGenerator
import numpy as np

class Agent(object):
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
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
        self.set_rnn_state(h, c)    # Update state

        return probs

    def act(self, state, epsilon=0, deterministic=False):
        '''
        # Arguments:
            state: numpy array, dtype=int, shape = (B, t)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        word = state[:, -1].reshape([-1, 1])
        return self._act_on_word(word, epsilon=epsilon, deterministic=deterministic)

    def _act_on_word(self, word, epsilon=0, deterministic=False):
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
        elif not deterministic:
            probs = self.evaluate(word)
            action = self.sampling(probs)
        else:
            probs = self.evaluate(word) # (B, T)
            action = np.argmax(probs, axis=-1).reshape([-1, 1])
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
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, discriminator, data_generator, g_beta, n_sample=16):
        '''
        Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            data_generator: SeqGAN.models.GeneratorPretrainingGenerator
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        # Optional Arguments
            n_sample: int, default is 16, the number of Monte Calro search sample
        '''
        self.data_generator = data_generator
        self.B = data_generator.B
        self.T = data_generator.T
        self.n_sample = n_sample
        self.BOS = data_generator.BOS
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.reset()

    def reset(self):
        self.t = 1
        self.state = np.zeros([self.B, 1], dtype=np.int32)
        self.state[:, 0] = self.BOS
        self.g_beta.reset_rnn_state()

    def step(self, action):
        '''
        Step t -> t + 1 and returns a result of the Agent action.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1),
                state is Y_0:t-1, and action is y_t
        # Returns:
            next_state: numpy array, dtype=int, shape = (B, t)
            reward: numpy array, dtype=float, shape = (B, 1)
            is_episode_end: bool
            info: dict
        '''
        self.t = self.t + 1

        reward = self.Q(action, self.n_sample)
        is_episode_end = self.t >= self.T

        self._append_state(action)
        next_state = self.state
        info = None

        return [next_state, reward, is_episode_end, info]

    def render(self, head=1):
        for i in range(head):
            ids = self.state[i, :]
            words = [self.data_generator.id2word[id] for id in ids.tolist()]
            print(''.join(words))
        print('-' * 80)


    def Q(self, action, n_sample=16):
        '''
        State-Action value function using Rollout policy
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)

        # Optional Arguments:
            n_sample: int, default is 16, number of samples for Monte Calro Search

        # Returns:
            reward: numpy array, dtype=float, shape = (B, 1), State-Action value

        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        '''
        h, c = self.g_beta.get_rnn_state()
        reward = np.zeros([self.B, 1])
        Y_base = self.state    # (B, t-1)

        if self.t >= self.T - 1:
            Y = self._append_state(action, state=Y_base)
            return self.discriminator.predict(Y)

        # Rollout
        for idx_sample in range(n_sample):
            Y = Y_base
            self.g_beta.set_rnn_state(h, c)
            y_t = self.g_beta.act(Y, epsilon=0)
            Y = self._append_state(y_t, state=Y)

            for tau in range(self.t+1, self.T):
                y_tau = self.g_beta.act(Y, epsilon=0)
                Y = self._append_state(y_tau, state=Y)
            reward += self.discriminator.predict(Y)
        reward = np.average(reward, axis=-1)

        return reward


    def _append_state(self, word, state=None):
        '''
        # Arguments:
            word: numpy array, dtype=int, shape = (B, 1)
        '''
        if state is None:
            self.state = np.concatenate([self.state, word], axis=-1)
        else:
            return np.concatenate([state, word], axis= -1)
