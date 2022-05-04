import yaml
from shasta.env import ShastaEnv
from experiments.adversary_experiment import AdversaryExperiment
from experiments.actor_groups import create_actor_groups
import warnings
import random
from torch_geometric.utils.convert import from_networkx
import time
warnings.filterwarnings("ignore")


NUM_UAV_GROUPS = 0
NUM_UGV_GROUPS = 100
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

action= random.sample(range(405),NUM_UGV_GROUPS + NUM_UAV_GROUPS)

for j in range(1):
    for i in range(50000):
        observation, reward, done, info = env.step([0]*NUM_UGV_GROUPS)
        # action= random.sample(range(405),NUM_UGV_GROUPS + NUM_UAV_GROUPS)
        # observation, reward, done, info = env.step(action)
        # time.sleep(0.0000001)
        print(done)
        if all(done):
            break
    actor_groups = create_actor_groups(NUM_UAV_GROUPS,NUM_UGV_GROUPS,NUM_AGENTS_PER_GROUP)
    env.reset(actor_groups)
    print('///////////////////////////////////////////////////////')
