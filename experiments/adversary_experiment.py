import enum
from platform import node
from turtle import pos

from attr import attributes
from shasta import actor
from shasta.base_experiment import BaseExperiment
import numpy as np
import networkx as nx
import copy

from .custom_primitive import FormationWithPlanning


class AdversaryExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()


        # Primitive setup
        

    def reset(self,config,core):
        """Called at the beginning and each time the simulation is reset"""
        self.config = config
        self.core = core
        self.observed_adversaries = set()
        self.actions = {}
        env_map = core.get_map()
        self.max_timesteps = config['max_timesteps']
        self.adversary_reward_multipler = config['adversary_reward_multipler']
        self.num_actor_groups = len(core.get_actor_groups())
        self.readys = {}
        self.last_timestep = 0
        self.dones = [False]*self.num_actor_groups
        self.cum_reward = {i:0 for i in range(self.num_actor_groups)}
        for i in range(self.num_actor_groups):
            self.actions[i] = FormationWithPlanning(env_map)
            self.readys[i] = True

    def get_action_space(self):
        """Returns the action space"""
        pass

    def get_observation_space(self):
        """Returns the observation space"""
        pass

    def get_actions(self):
        """Returns the actions"""
        pass

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        """
        # Get the actor group
        actor_groups = core.get_actor_groups()
        num_actor_groups = len(actor_groups)
        for i in range(num_actor_groups):
            if not self.dones[i]:
                ready = self.actions[i].execute(self.readys[i], actor_groups[i], target_pos=actions[i])
                self.readys[i] = ready
            else:
                self.readys[i] = False
        return self.readys

    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        env_map = core.get_map()
        node_graph = env_map.get_node_graph().copy()
        agent_nodes,adv_nodes,target_nodes = observation
        attributes = {}

        for node in set(agent_nodes):
            if node in attributes.keys():
                attributes[node]['agent'] = 1
            else:
                attributes[node] = {'agent':1}

        for node in adv_nodes:
            if node in attributes.keys():
                attributes[node]['adversary'] = 1
            else:
                attributes[node] = {'adversary':1}

        for node in target_nodes:
            if node in attributes.keys():
                attributes[node]['target'] = 1
            else:
                attributes[node] = {'target':1}

        nx.set_node_attributes(node_graph,attributes)
        return (node_graph,agent_nodes),{}

    def get_done_status(self, observation, timestep, core):
        """Returns whether or not the experiment has to end"""
        agent_nodes,total_adv_nodes,target_nodes = observation
        if timestep > self.max_timesteps:
            return [True] * self.num_actor_groups
        
        actor_groups = core.get_actor_groups()

        for i,agent_node in enumerate(agent_nodes):
            self.dones[i] = self.dones[i] or agent_node in target_nodes
        
        return self.dones

    def compute_reward(self, timestep, adver_times, readys):
        """Computes the reward"""
        out_reward = {i:0 for i in range(self.num_actor_groups)}
        for agent_num, adver_time in adver_times.items():
            self.cum_reward[agent_num] = 0.1 * (-adver_time*self.adversary_reward_multipler - (timestep - self.last_timestep))
            if readys[agent_num]:
                out_reward[agent_num] = self.cum_reward[agent_num]
                self.cum_reward[agent_num] = 0
        self.last_timestep = timestep
        return out_reward

    def get_nearest_node(self,positions_vector,actor_loc):
        return ((positions_vector - actor_loc)**2).sum(axis=1).argmin()

    

    

