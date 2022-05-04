import os
import psutil
import signal

import pybullet as p
from pybullet_utils import bullet_client as bc
import networkx as nx
import numpy as np
from math import sin, cos, sqrt, radians

from .world import World
from .map import Map

from .utils import get_initial_positions


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [
        p for p in psutil.process_iter() if "carla" in p.name().lower()
    ]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class ShastaCore():
    """
    Class responsible of handling all the different CARLA functionalities,
    such as server-client connecting, actor spawning,
    and getting the sensors data.
    """
    def __init__(self, config):
        """Initialize the server and client"""
        self.config = config

        self.world = World(config)
        # Setup the map
        self.map = Map()

        self.init_server()
        self._setup_physics_client()

        self.spawned = False

    def _setup_physics_client(self):
        """Setup the physics client

        Returns
        -------
        None
        """
        # Usage mode
        if self.config['headless']:
            self.physics_client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            options = '--background_color_red=0.85 --background_color_green=0.85 --background_color_blue=0.85'  # noqa
            self.physics_client = bc.BulletClient(connection_mode=p.GUI,
                                                  options=options)

            # Set the camera parameters
            self.camer_distance = 150.0
            self.camera_yaw = 0.0
            self.camera_pitch = -89.999
            self.camera_target_position = [0, 30, 0]
            self.physics_client.resetDebugVisualizerCamera(
                cameraDistance=self.camer_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target_position)

            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_GUI, 0)

        # Set gravity
        self.physics_client.setGravity(0, 0, -9.81)

        # Set parameters for simulation
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self.config['time_step'] / 10,
            numSubSteps=1,
            numSolverIterations=5)

        # Inject physics client
        if self.world.physics_client is None:
            self.world.physics_client = self.physics_client

        return None

    def get_physics_client(self):
        """Ge the physics client

        Returns
        -------
        object
            The bullet physics client
        """
        return self.physics_client

    def init_server(self):
        """Start a server on a random port"""
        pass

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        # Load the environment and setup the map
        self.map.setup(experiment_config)
        read_path = self.map.asset_path + '/environment_collision_free.urdf'
        self.world.load_world_model(read_path)

        # Spawn the actors in the physics client
        # self.spawn_actors()

    def get_world(self):
        """Get the World object from the simulation

        Returns
        -------
        object
            The world object
        """
        return self.world

    def get_map(self):
        """Get the Map object from the simulation

        Returns
        -------
        object
            The map object
        """
        return self.map

    def reset(self,actor_groups):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Reset all the actors
        self.despawn_actors()

        self.actor_groups = actor_groups

        # Verify if the actor groups is a dictionary
        if not isinstance(self.actor_groups, dict):
            raise TypeError('Actor groups should be of type dict')

        self.spawn_actors()
        self.generate_adversaries(10)

        for group_id in self.actor_groups:
            # Check if the entry is a list or not
            if not isinstance(self.actor_groups[group_id], list):
                self.actor_groups[group_id] = [self.actor_groups[group_id]]

        num_actor_groups = len(self.actor_groups)
        agent_vector = np.zeros((num_actor_groups,2))
        positions_vector = self.map.get_positions_vector()
        agent_nodes = set()
        total_adv_nodes = set()
        for i,actor in self.actor_groups.items():
            actor[0].reset()
            actor_loc = self.map.convert_to_lat_lon(actor[0].get_observation())[:2]
            node = self.get_nearest_node(positions_vector,actor_loc[:2])
            agent_nodes.add(node)
            agent_vector[i] = actor_loc
            adv_nodes = self.get_visible_adversaries(actor_loc[:2],0.2)
            total_adv_nodes.update(adv_nodes)
        
        return (agent_nodes,total_adv_nodes)

    def spawn_actors(self):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""
        self.spawned = True
        for group_id in self.actor_groups:
            # Check if the entry is a list or not
            if not isinstance(self.actor_groups[group_id], list):
                self.actor_groups[group_id] = [self.actor_groups[group_id]]

            # Spawn the actors
            spawn_point = self.map.get_cartesian_spawn_points()
            positions = get_initial_positions(spawn_point, 10,
                                              len(self.actor_groups[group_id]))
            for actor, position in zip(self.actor_groups[group_id], positions):
                if actor.init_pos is None:
                    actor.init_pos = position
                else:
                    actor.init_pos = self.map.convert_from_lat_lon(
                        actor.init_pos)

                self.world.spawn_actor(actor, position)

    def despawn_actors(self):
        if self.spawned:
            for group_id in self.actor_groups:
                self.world.despawn_actors(self.actor_groups[group_id])


    def get_actor_groups(self):
        """Get the actor groups

        Returns
        -------
        dict
            The actor groups as a dict with group id as the key
            list of actors as the value
        """
        return self.actor_groups

    def get_actors_by_group_id(self, group_id):
        """Get a list of actor given by group id

        Parameters
        ----------
        group_id : int
            Group id to be returned

        Returns
        -------
        list
            A list of actor given the group id
        """
        return self.actor_groups[group_id]

    def tick(self):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Tick once the simulation
        self.physics_client.stepSimulation()

        num_actor_groups = len(self.actor_groups)
        agent_vector = np.zeros((num_actor_groups,2))
        positions_vector = self.map.get_positions_vector()
        agent_nodes = set()
        total_adv_nodes = set()
        for i,actor in self.actor_groups.items():
            actor_loc = self.map.convert_to_lat_lon(actor[0].get_observation())[:2]
            node = self.get_nearest_node(positions_vector,actor_loc[:2])
            agent_nodes.add(node)
            agent_vector[i] = actor_loc
            adv_nodes = self.get_visible_adversaries(actor_loc[:2],0.2)
            total_adv_nodes.update(adv_nodes)
    
        return (agent_nodes,total_adv_nodes)

    def close_simulation(self):
        """Close the simulation
        """
        p.disconnect(self.physics_client._client)

    def generate_adversaries(self,num_advers):
        node_graph = self.map.get_node_graph()
        position_vector = self.map.get_positions_vector()
        y = [node_graph.nodes[n]['y'] for n in node_graph.nodes()]
        x = [node_graph.nodes[n]['x'] for n in node_graph.nodes()]
        mins = np.array([min(y),min(x)])
        self.set_earth_radius((min(y)+max(y))/2)
        spawn_range = np.array([max(y), max(x)]) - mins
        adv_locs = np.random.rand(num_advers,2) * spawn_range + mins
        new_adv_locs = np.zeros((num_advers,2))
        adv_nodes = []
        for i in range(num_advers):
            node = self.get_nearest_node(position_vector,adv_locs[i])
            adv_nodes.append(node)
            new_adv_locs[i] = list(node_graph.nodes[node].values())[:2]
        self.adv_nodes = adv_nodes
        self.adv_vector = np.radians(new_adv_locs)
        # print(self.adv_vector)
        return None

    def set_earth_radius(self,lat):
        lat=radians(lat) #converting into radians
        radius_equator = 6378.137  #Radius at sea level at equator
        radius_poles = 6356.752  #Radius at poles
        c = (radius_equator**2*cos(lat))**2
        d = (radius_poles**2*sin(lat))**2
        e = (radius_equator*cos(lat))**2
        f = (radius_poles*sin(lat))**2
        R = sqrt((c+d)/(e+f))
        self.earth_radius = R
        return None

    def get_visible_adversaries(self,actor_loc,visible_range):
        actor_loc = np.radians(actor_loc)
        lat1 = self.adv_vector[:,0]
        lon1 = self.adv_vector[:,1]
        lat2 = actor_loc[0]
        lon2 = actor_loc[1]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = self.earth_radius * c
        adv_nodes = [node for node,distance in zip(self.adv_nodes,distances) if distance < visible_range]
        return adv_nodes

    def get_nearest_node(self,positions_vector,actor_loc):
        return ((positions_vector - actor_loc)**2).sum(axis=1).argmin()

