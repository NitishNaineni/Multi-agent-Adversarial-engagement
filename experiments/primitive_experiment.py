from shasta.base_experiment import BaseExperiment

from .custom_primitive import FormationWithPlanning


class PrimitiveExperiment(BaseExperiment):
    def __init__(self, config, core):
        super().__init__(config, core)

        # Primitive setup
        self.actions = {}
        env_map = core.get_map()
        for i in range(6):
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
        for i in range(6):
            done = self.actions[i].execute(actor_groups[i], target_pos=0)
            self.dones.append(done)

    def get_observation(self, observation, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return None, {}

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return self.dones

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return 0
