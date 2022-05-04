from turtle import pos
import warnings

from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx

from .assets import assets_root


class Map():
    def __init__(self) -> None:
        return None

    def _affine_transformation_and_graph(self,experiment_config):
        """Performs initial conversion of the lat lon to cartesian
        """
        # Graph
        read_path = self.asset_path + '/map.osm'
        G = ox.graph_from_xml(read_path, simplify=True, bidirectional='walk')
        
        node_graph = nx.convert_node_labels_to_integers(G)
        self.node_graph = node_graph

        # Transformation matrix
        read_path = self.asset_path + '/coordinates.csv'
        points = pd.read_csv(read_path)
        target = points[['x', 'z']].values
        source = points[['lat', 'lon']].values

        # Pad the points with ones
        X = np.hstack((source, np.ones((source.shape[0], 1))))
        Y = np.hstack((target, np.ones((target.shape[0], 1))))
        self.A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        return None

    def _filter_attributes(self,experiment_config):
        for node in self.node_graph.nodes(data=True):
            remove_node_atrs = set(node[1].keys()) - set(experiment_config['node_attributes'])
            for node_atr in remove_node_atrs:
                del node[1][node_atr]

        for edge in self.node_graph.edges(data=True):
            remove_edge_atrs = set(edge[2].keys()) - set(experiment_config['edge_attributes'])
            for edge_atr in remove_edge_atrs:
                del edge[2][edge_atr]
        return None

    def _setup_buildings(self):
        """Perfrom initial building setup.
        """
        read_path = self.asset_path + '/buildings.csv'

        # Check if building information is already generated
        if Path(read_path).is_file():
            buildings = pd.read_csv(read_path)
        else:
            read_path = self.asset_path + '/map.osm'
            G = ox.graph_from_xml(read_path)
            # TODO: This method doesn't work if the building info is not there in OSM
            nodes, streets = ox.graph_to_gdfs(G)

            west, north, east, south = nodes.geometry.total_bounds
            polygon = ox.utils_geo.bbox_to_poly(north, south, east, west)
            gdf = ox.geometries.geometries_from_polygon(
                polygon, tags={'building': True})
            buildings_proj = ox.project_gdf(gdf)

            # Save the dataframe representing buildings
            buildings = pd.DataFrame()
            buildings['lon'] = gdf['geometry'].centroid.x
            buildings['lat'] = gdf['geometry'].centroid.y
            buildings['area'] = buildings_proj.area
            buildings['perimeter'] = buildings_proj.length
            try:
                buildings['height'] = buildings_proj['height']
            except KeyError:
                buildings['height'] = 10  # assumption
            buildings['id'] = np.arange(len(buildings_proj))

            # Save the building info
            save_path = self.asset_path + '/buildings.csv'
            buildings.to_csv(save_path, index=False)

        self.buildings = buildings
        return None
    
    def _cumulative_positions_vector(self):
        num_nodes = self.node_graph.number_of_nodes()
        positions_vector = np.zeros((num_nodes,2))
        for node in self.node_graph.nodes(data=True):
            positions_vector[node[0]] = [node[1]['y'],node[1]['x']]
        self.positions_vector = positions_vector

        return None

    def _initialize_agent_adversaries(self):
        nx.set_node_attributes(self.node_graph,0,'agent')
        nx.set_node_attributes(self.node_graph,0,'adversary')

    def setup(self, experiment_config):
        """Perform the initial experiment setup e.g., loading the map

        Parameters
        ----------
        experiment_config : yaml
            A yaml file providing the map configuration

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the experiment config is none, raises a file not found error
        """
        self.experiment_config = experiment_config

        # Read path for ths assets
        try:
            self.asset_path = '/'.join(
                [assets_root, self.experiment_config['map_to_use']])
        except FileNotFoundError:
            try:
                self.asset_path = '/'.join(
                    [assets_root, self.experiment_config['map_to_use']])
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Please verify the {self.experiment_config['map_to_use']} is available in asset folder"
                )

        # Initialize the assests
        self._affine_transformation_and_graph(experiment_config)
        self._filter_attributes(experiment_config)
        self._setup_buildings()
        self._cumulative_positions_vector()
        self._initialize_agent_adversaries()
        return None

    def get_affine_transformation_and_graph(self):
        """Get the transformation matrix and the node graph of the map

        Returns
        -------
        array, node graph
            The transformation matrix and the node graph
        """
        return self.A, self.node_graph

    def get_node_graph(self):
        """Get the node graph of the world

        Returns
        -------
        networkx graph
            A node graph of the world map
        """
        return self.node_graph

    def get_positions_vector(self):
        return self.positions_vector

    def get_node_info(self, node_index):
        """Get the information about a node.

        Parameters
        ----------
        id : int
            Node ID

        Returns
        -------
        dict
            A dictionary containing all the information about the node.
        """
        return self.node_graph.nodes[node_index]

    def convert_to_lat_lon(self, point):
        """Convert a given point to lat lon co-ordinates

        Parameters
        ----------
        point : array
            A numpy array in pybullet cartesian co-ordinates

        Returns
        -------
        lat_lon : array
            The lat lon co-ordinates
        """
        point[2] = 1
        lat_lon = np.dot(point, np.linalg.inv(self.A))
        return lat_lon

    def convert_from_lat_lon(self, point):
        """Convert a lat lon co-ordinates to cartesian coordinates

        Parameters
        ----------
        point : array
            A numpy array in lat lon co-ordinates co-ordinates

        Returns
        -------
        lat_lon : array
            The cartesian coordinates
        """
        return np.dot([point[0], point[1], 1], self.A)

    def get_building_info(self, building_index):
        """Get the information about a building such as perimeter,
            position, number of floors.

            Parameters
            ----------
            id : int
                Building ID

            Returns
            -------
            dict
                A dictionary containing all the information about the building.
            """
        return self.buildings.loc[self.buildings['id'] == building_index]

    def get_lat_lon_spawn_points(self, n_points=5):
        """Get the latitude and longitude spawn points

        Parameters
        ----------
        n_points : int, optional
            Number of points to random latitude and longitude points, by default 5

        Returns
        -------
        array
            An array of cartesian spawn points
        """
        # TODO: Verify if the projection is correct and the warning is not
        # affecting the values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf = ox.utils_geo.sample_points(self.node_graph, n_points)
            points = np.vstack([gdf.centroid.y, gdf.centroid.x]).T
        return points

    def get_cartesian_node_position(self, node_index):
        """Get the cartesian co-ordinates given the node index

        Parameters
        ----------
        node_index : int
            The node index in the map

        Returns
        -------
        array
            The cartesian co-ordinates
        """
        node_info = self.get_node_info(node_index=node_index)
        lat = node_info['y']
        lon = node_info['x']
        cartesian_pos = np.dot([lat, lon, 1], self.A)
        return cartesian_pos

    def get_cartesian_spawn_points(self, n_points=5):
        """Get the cartesian spawn points

        Parameters
        ----------
        n_points : int, optional
            Number of points to random cartesian co-ordinates, by default 5

        Returns
        -------
        array
            An array of cartesian spawn points
        """
        lat_lon_points = self.get_lat_lon_spawn_points(n_points)

        cartesian_spawn_points = []
        for point in lat_lon_points:
            cartesian_spawn_points.append(
                np.dot([point[0], point[1], 1], self.A))
        return cartesian_spawn_points[0]

    def get_all_buildings(self):
        """Get all the buildings

        Returns
        -------
        dataframe
            A dataframe with all the building information
        """
        return self.buildings

    def get_transformation_matrix(self):
        """Get the transformation matrix to convert lat lon to cartesian

        Returns
        -------
        array
            The transformation matrix
        """
        return self.A
