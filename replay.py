import numpy as np
from collections import deque,namedtuple
import networkx as nx
import random

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
    def __init__(self, size, sample_size):
        self.size = size
        self.sample_size = sample_size
        self.buffer = deque(maxlen=size)
        self.current_size = 0

    def push(self,states,actions,rewards,next_states,dones,targets,last_iter,last_positions):
        

        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.buffer.append(Experience(state,action,reward,next_state,done))
            self.current_size += 1
        
        new_targets_attibutes = {}

        for i,final_agent_pos in last_positions.items():
            if final_agent_pos not in targets:
                if final_agent_pos in new_targets_attibutes.keys():
                    new_targets_attibutes[final_agent_pos]["target"] = 1
                else:
                    new_targets_attibutes[final_agent_pos] = {"target":1}
                rewards[last_iter[i]] = 10

        for target in targets:
            if target not in last_positions.values():
                if target in new_targets_attibutes.keys():
                    new_targets_attibutes[target]["target"] = 0
                else:
                    new_targets_attibutes[target] = {"target":0}

        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            graph,agent_nodes = state
            next_graph,next_agent_nodes = next_state
            nx.set_node_attributes(graph,new_targets_attibutes)
            nx.set_node_attributes(next_graph,new_targets_attibutes)
            state = (graph,agent_nodes)
            next_state = (next_graph,next_agent_nodes)
            self.buffer.append(Experience(state,action,reward,next_state,done))
            self.current_size += 1

    def sample(self):
        return random.sample(self.buffer, self.sample_size)

    def get_size(self):
        return min(self.current_size,self.size)