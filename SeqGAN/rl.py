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
        self.generator_pre = GeneratorPretraining(V, E, H)
        self.generator = Generator(B, V, E, H)

    def evaluate(self, state, model=None):
        '''
        Evaluate next word probability by using state (word id).
        model should be stateful.
        # Arguments:
            state: numpy array, dtype=int, shape = (B, 1),
                state indicates current word.
        # Returns:
            probs: numpy array, dtype=float, shape = (B, V),
                probs are next word probabiliies.
        '''
        _model = model if model else self.generator
        probs = _model.predict(state)    # (B, V)
        return probs

    def act(self, state, epsilon=0):
        '''
        # Arguments:
            state: numpy array, dtype=int, shape = (B, 1),
                state indicates current word.
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        action = None
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        else:
            probs = self.evaluate(state)
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
            data_generator: SeqGAN.DiscriminatorGenerator
            g_beta: SeqGAN.Generator, copy of Agent's generator.
                params of g_beta should be updated with those of Agent's
                generator on regular occasions.
        '''
            # data_generator = DiscriminatorGenerator(
            #     path_pos=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            #     path_neg=os.path.join(top, 'tests', 'data', 'sample_generated.txt'),
            #     B=32,
            #     shuffle=True)
            # filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # CNNに使うフィルターのサイズ
            # num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] # CNNに使うフィルターの数
            # discriminator = Discriminator(6612, 3, filter_sizes, num_filters, 0.75)
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
        self.state[:, self.t] = action
        self.t = self.t + 1

        next_state = self.state
        is_episode_end = self.t >= self.T
        reward = self.Q()


    def render(self):

    def Q(self, action):
        '''
        State-Action value function using Rollout policy.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)
        # Returns:
            reward: numpy array, dtype=float, State-Action value
        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        '''
