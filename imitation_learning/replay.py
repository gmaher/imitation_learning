import numpy as np

class EpisodeReplay:
    def __init__(self, input_size, action_size, num_steps,
            size=1000, batch_size=8):
        self.input_size = input_size
        self.action_size = action_size
        self.num_steps = num_steps
        self.size = size
        self.batch_size = batch_size

        self.tuples = []

    def append(self, tuples):
        if len(self.tuples) < self.size:
            self.tuples.append(tuples)
        else:
            i = np.random.randint(self.size)
            self.tuples[i] = tuples

    def sample(self):
        S = np.zeros((self.batch_size, self.num_steps, self.input_size),
            dtype=np.float32)
        A = np.zeros((self.batch_size, self.num_steps, self.action_size),
            dtype=np.float32)
        R = np.zeros((self.batch_size, self.num_steps),
            dtype=np.float32)

        tuples = np.random.choice(self.tuples, size=self.batch_size)

        for i,T in enumerate(tuples):
            n = len(T)
            s = np.array([t[0] for t in T])
            if (len(s.shape) == 1): s = s[:,np.newaxis]
            S[i,:n,:] = s

            a = np.array([t[1] for t in T])
            for j,a_ in enumerate(a): A[i,j,a_] = 1

            R[i,:n] = np.array([t[2] for t in T])

        return S,A,R
