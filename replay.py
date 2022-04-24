import numpy as np
from collections import deque,namedtuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class Prioritized_Experience_Replay:
    def __init__(self, size, alpha, sample_size):
        self.size = size
        self.alpha = alpha
        self.sample_size = sample_size
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.offset = 1
        self.p_total = 0
        self.timestep = 0

    def push(self,loss,*args):
        p = (abs(loss) + self.offset)**self.alpha
        
        if self.timestep >= self.size:
            last = self.priorities[0]
            self.p_total -= last

        self.priorities.append(p)
        self.buffer.append(Experience(*args))

        self.p_total += p
        self.timestep += 1

    def sample(self):
        seed = np.random.randint(0,2**31)
        sample_prob = np.array(self.priorities)/self.p_total
        np.random.seed(seed)
        sampled = np.random.choice(self.buffer,self.sample_size,p=sample_prob)
        np.random.seed(seed)
        priorities = np.random.choice(self.priorities,self.sample_size,p=sample_prob)
        return sampled,priorities

    def is_full(self):
        return self.timestep >= self.size

class Hindsight_Experience_Replay:
    def __init__(self, size, alpha, sample_size):
        self.size = size
        self.alpha = alpha
        self.sample_size = sample_size
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.offset = 1
        self.p_total = 0
        self.timestep = 0

    def push(self,*args):
        