import yaml
import time

from utils import skip_run

from shasta.env import ShastaEnv

from experiments.adversary_experiment import AdversaryExperiment
from experiments.actor_groups import create_actor_groups

import warnings

warnings.filterwarnings("ignore")


NUM_UAV_GROUPS = 0
NUM_UGV_GROUPS = 1
NUM_AGENTS_PER_GROUP = 1

config_path = 'config/simulation_config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


actor_groups = create_actor_groups(NUM_UAV_GROUPS,NUM_UGV_GROUPS,NUM_AGENTS_PER_GROUP)
config['experiment']['type'] = AdversaryExperiment
env = ShastaEnv(config, actor_groups=actor_groups)

for i in range(50000):
    observation, reward, done, info = env.step(0)
    if all(done):
        break

print(observation)