#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import gym

from .core import ShastaCore
import networkx as nx

class ShastaEnv(gym.Env):
    """
    This is a shasta environment, responsible of handling all the SHASTA related steps of the training.
    """
    def __init__(self, config, actor_groups: dict = None):
        """Initializes the environment"""
        self.config = config

        # Check if experiment config is present
        if not self.config["experiment"]:
            raise Exception("The config should have experiment configuration")

        # Setup the core
        self.core = ShastaCore(self.config)
        self.core.setup_experiment(self.config["experiment"])

        self.experiment = self.config["experiment"]["type"]()

        if not self.experiment:
            raise Exception(
                "The experiment type cannot be empty. Please provide an experiment class"
            )

        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.timestep = 0


    def reset(self,actor_groups):
        """Reset the simulation

        Returns
        -------
        [type]
            [description]
        """
        
        self.num_actors = len(actor_groups)
        # Tick once and get the observations
        raw_data = self.core.reset(actor_groups)
        self.experiment.reset(self.config["experiment"], self.core)
        observation, _ = self.experiment.get_observation(raw_data, self.core)
        self.timestep = 0

        return observation

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        adver_times  = {i:0 for i in range(self.num_actors)}
        readys = self.experiment.apply_actions(action, self.core)
        raw_data = self.core.tick()
        self.timestep += 1

        agent_nodes,total_adv_nodes,target_nodes  = raw_data
        for i,agent_node in enumerate(agent_nodes):
            if agent_node in total_adv_nodes:
                adver_times[i] = adver_times[i] - 1

        while not any(list(readys.values())):
            readys = self.experiment.apply_actions(action, self.core)
            raw_data = self.core.tick()
            self.timestep += 1

            agent_nodes,total_adv_nodes,target_nodes  = raw_data
            for i,agent_node in enumerate(agent_nodes):
                if agent_node in total_adv_nodes:
                    adver_times[i] += 1


        observation, info = self.experiment.get_observation(raw_data, self.core)
        dones = self.experiment.get_done_status(raw_data, self.timestep, self.core)
        reward = self.experiment.compute_reward(self.timestep, adver_times, readys)

        return observation, reward, dones, readys, info

    def close(self):
        self.core.close_simulation()
