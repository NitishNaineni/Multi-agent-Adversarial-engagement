import yaml
import time

from utils import skip_run

from shasta.env import ShastaEnv

from experiments.adversary_experiment import AdversaryExperiment
from experiments.primitive_experiment import PrimitiveExperiment
from experiments.actor_groups import create_actor_groups

config_path = 'config/simulation_config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


actor_groups = create_actor_groups(6,100)
config['experiment']['type'] = PrimitiveExperiment
env = ShastaEnv(config, actor_groups=actor_groups)

for i in range(50000):
    observation, reward, done, info = env.step(0)
    if all(done):
        break

print(observation)