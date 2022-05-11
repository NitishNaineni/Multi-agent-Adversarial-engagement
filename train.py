from itertools import count
import yaml
from shasta.env import ShastaEnv
from experiments.adversary_experiment import AdversaryExperiment
from experiments.actor_groups import create_actor_groups
import warnings
import random
from replay import Hindsight_Experience_Replay
import torch
import torch.optim as optim
from utils import Decay,ActionSelector
from model import GAT
warnings.filterwarnings("ignore")

min_er = 0.01
max_er = 1
decay_er = 0.001


config_path = 'config/adversary_config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
NUM_UGV_GROUPS = config['num_agents']
num_episodes = config['num_episodes']


config['experiment']['type'] = AdversaryExperiment
env = ShastaEnv(config)
hindsight = Hindsight_Experience_Replay(10000,256)
decay = Decay(max_er,min_er,decay_er)
action_sample = env.get_actions()
actionselector = ActionSelector(decay,len(action_sample))
policy = GAT(num_gat_layers=3,gat_out_channels=8,num_agents=NUM_UGV_GROUPS,num_actions=len(action_sample))
target = GAT(num_gat_layers=3,gat_out_channels=8,num_agents=NUM_UGV_GROUPS,num_actions=len(action_sample))
target.load_state_dict(policy.state_dict())
target.eval()
optimizer = optim.Adam(params=policy.parameters(),lr=0.001)



for episode in range(num_episodes):
    actor_groups = create_actor_groups(0,NUM_UGV_GROUPS,1)
    state = env.reset(actor_groups)
    action_sample = env.get_actions()
    state = {i:(state[0],state[1][i]) for i in range(len(state[1]))}
    print(state)

    action= {i:None for i in range(NUM_UGV_GROUPS)}
    readys = {i:True for i in range(NUM_UGV_GROUPS)}
    dones = {i:False for i in range(NUM_UGV_GROUPS)}

    done_tracker = {i:False for i in range(NUM_UGV_GROUPS)}
    
    states,actions,rewards,next_states,all_dones,targets = [],[],[],[],[],[380]
    last_iter = {i:0 for i in range(NUM_UGV_GROUPS)}
    iter = 0
    while not all(dones.values()):







        active_agents = []

        for agent_num,ready in  readys.items():
            if ready:
                ind_action,rate = actionselector.select_action(torch.tensor(state[agent_num],dtype=torch.float32),target)
                action[agent_num],rate = action_sample[ind_action]
                print(ind_action)
        
        next_state, reward, dones, readys, info = env.step(action)

    
        for agent_num,ready in  readys.items():
            if ready:
                active_agents.append(agent_num)
                

        for agent_num,done in dones.items():
            if done and agent_num not in active_agents:
                active_agents.append(agent_num)

        for agent_num in active_agents:
            next_states.append((next_state[0],next_state[1][agent_num]))
            states.append(state[agent_num])
            rewards.append(reward[agent_num])
            actions.append(action[agent_num])
            all_dones.append(dones[agent_num])
            state[agent_num] = (next_state[0],next_state[1][agent_num])
            if dones[agent_num] and not done_tracker[agent_num]:
                last_iter[agent_num] = iter
                done_tracker[agent_num] = True
            iter += 1

    last_positions = next_state[1]
    hindsight.push(states,actions,rewards,next_states,all_dones,targets,last_iter,last_positions)

    if hindsight.get_size() > 256:
        test = hindsight.sample()

    actor_groups = create_actor_groups(0,NUM_UGV_GROUPS,1)
    env.reset(actor_groups)
    print('///////////////////////////////////////////////////////')

