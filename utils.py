import sys
from contextlib import contextmanager
import random
import torch


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format(
                'Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block',
                                                       '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

class ActionSelector:
    def __init__(self,decay,num_actions):
        self.decay = decay
        self.num_actions = num_actions
        self.timestep = 0
        
    def select_action(self,state,policy):
        rate = self.decay.get_ep_rate(self.timestep)
        self.timestep += 1
        
        if rate > random.random():
            return random.randrange(self.num_actions),rate
        else:
            with torch.no_grad():
                return policy(state).argmax().item(),rate


class Decay:
    def __init__(self,start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_ep_rate(self,timestep):
        return self.end + (self.start - self.end) * np.exp(-1 * timestep * self.decay)