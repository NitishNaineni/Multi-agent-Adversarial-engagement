from platform import node
from turtle import pos
from shasta import actor
from shasta.base_experiment import BaseExperiment
import osmnx as ox

from .custom_primitive import FormationWithPlanning


class AdversaryExperiment(BaseExperiment):
    def __init__(self, config, core):
        super().__init__(config, core)


        # Primitive setup
        self.actions = {}
        env_map = core.get_map()
        num_actor_groups = len(core.get_actor_groups())
        for i in range(num_actor_groups):
            self.actions[i] = FormationWithPlanning(env_map)

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

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
        self.dones = []
        num_actor_groups = len(actor_groups)
        for i in range(num_actor_groups):
            done = self.actions[i].execute(actor_groups[i], target_pos=actions[i])
            self.dones.append(done)

    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        map = core.get_map()
        node_graph = map.get_node_graph()
        actor_groups = core.get_actor_groups()
        num_actor_groups = len(actor_groups)
        agent_node_locs = []
        positions_vector = map.get_positions_vector()
        for i in range(num_actor_groups):
            actor_loc = map.convert_to_lat_lon(actor_groups[i][0].current_pos)
            node_loc = self.get_nearest_node(positions_vector,actor_loc[:2])
            print(map.convert_from_lat_lon(list(node_graph.nodes[node_loc].values())),actor_groups[i][0].current_pos)
        return (observation,node_graph), {}

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return self.dones

    def compute_reward(self, observation, core):
        """Computes the reward"""
        rewards = {}
        for key,group_observations in observation[0].items():
            group_rewards = []
            for agent_observation in group_observations:
                group_rewards.append(1)
            rewards[key] = group_rewards
        return rewards

    def get_nearest_node(self,positions_vector,actor_loc):
        return ((positions_vector - actor_loc)**2).sum(axis=1).argmin()


