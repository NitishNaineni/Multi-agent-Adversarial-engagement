import numpy as np

from .agents.uav import UaV
from .agents.ugv import UgV


def get_initial_positions(cartesian_pos, r, n):
    positions = []
    t = np.linspace(0, 2 * np.pi, n)
    x = cartesian_pos[0] + r * np.cos(t)
    y = cartesian_pos[1] + r * np.sin(t)
    positions = np.asarray([x, y, x * 0 + 1]).T.tolist()
    return positions


def create_actor_groups(num_uav_groups=6,num_ugv_groups=6,num_agents=10):
    n_actor_groups = num_uav_groups + num_ugv_groups
    actor_groups = {}
    for i in range(n_actor_groups):
        temp = []
        for j in range(num_agents):
            temp.append(UaV() if i < num_uav_groups else UgV())
        actor_groups[i] = temp

    return actor_groups
