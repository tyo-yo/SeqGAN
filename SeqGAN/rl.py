from SeqGAN.models import Generator, GeneratorPretraining, Discriminator
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
        '''
        if np.random.rand() <= epsilon:
            a = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        else:
            probs = self.evaluate(state)
            a = self.sampling(prob)

    def sampling(self, probs):
        '''
        # Arguments:
            probs: numpy array, dtype=float, shape = (B, V),
        # Returns:
            sample: numpy array, dtype=int, shape = (B, 1)
        '''
        a = np.zeros((self.num_actions, 1), dtype=np.int32)

        for i in range(B):
            p = probs[i]
            a[i, 0] = np.random.choice(self.num_actions, p=p)
        return a
