import yaml
from shasta.env import ShastaEnv
from experiments.adversary_experiment import AdversaryExperiment
from experiments.actor_groups import create_actor_groups
import warnings
import random
from torch_geometric.utils.convert import from_networkx
import time
import numpy as np
warnings.filterwarnings("ignore")


NUM_UAV_GROUPS = 0
NUM_UGV_GROUPS = 3
NUM_AGENTS_PER_GROUP = 1

config_path = 'config/adversary_config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


actor_groups = create_actor_groups(NUM_UAV_GROUPS,NUM_UGV_GROUPS,NUM_AGENTS_PER_GROUP)
config['experiment']['type'] = AdversaryExperiment
env = ShastaEnv(config)
env.reset(actor_groups)
# G = env.get_map()
# data = from_networkx(G)
# print(data.x,data.y)

action= [0]*NUM_UGV_GROUPS
readys = {i:True for i in range(NUM_UGV_GROUPS)}

for j in range(10):
    for i in range(50000):
        for agent_num, ready in  readys.items():
            if ready:
                action[agent_num] = random.randint(0,404)

        # observation, reward, done, ready, info = env.step([0]*NUM_UGV_GROUPS)
        observation, reward, dones, readys, info = env.step(action)
        print(readys,dones,j,i,reward.values())
        if all(dones):
            break
    actor_groups = create_actor_groups(NUM_UAV_GROUPS,NUM_UGV_GROUPS,NUM_AGENTS_PER_GROUP)
    env.reset(actor_groups)
    print('///////////////////////////////////////////////////////')
