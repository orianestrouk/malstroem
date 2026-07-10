# coding=utf-8
# -------------------------------------------------------------------------------------------------
# Copyright (c) 2016
# Developed by Septima.dk and Thomas Balstrøm (University of Copenhagen) for the Danish Agency for
# Data Supply and Efficiency. This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 2 of the License, or (at you option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PORPOSE. See the GNU Gene-
# ral Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not,
# see http://www.gnu.org/licenses/.
# -------------------------------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function) #, unicode_literals)
from builtins import *
import logging
import numpy as np

from collections import defaultdict, deque

class FinalStateCalculator(object):
    """Calculate final state on a network.

    This class takes the initial water volumes and calculates spill over from each blue spot to the 
    next downstream bluespot

    Parameters
    ----------
    input_nodes : vectorreader
        Nodes including a precalculated initial water volume for each node
    output_finalstatedata : vectorwriter
        Writes the output final state data
    simulation_duration_s : float, optional
        Duration of the rain event in seconds. Used to calculate drained volume from drainage capacity.
    allow_initial_spillover : bool, optional
        If True, water above bspot_capacity spills over instantly and cannot be drained.
        If False, water above bspot_capacity can be drained at peak drainage capacity.
    """

    def __init__(self, input_nodes, input_volume_attribute, output_finalstatedata,
                 simulation_duration_s=3600, allow_initial_spillover=True):
        self.input_nodes = input_nodes
        self.input_volume_attribute = input_volume_attribute
        self.output_finalstatedata = output_finalstatedata
        self.simulation_duration_s = simulation_duration_s
        self.allow_initial_spillover = allow_initial_spillover
        self.logger = logging.getLogger(__name__)

    def process(self):
        """Process

        Returns
        -------
        None
        """

        self.logger.info("Reading input nodes")
        geojsonnodes_index = {gjn['properties']['nodeid']: gjn for gjn in self.input_nodes.read_geojson_features()}

        # Only use 'properties'
        nodes = [gjn['properties'] for gjn in geojsonnodes_index.values()]

        self.logger.info("Creating stream network")
        network = Network(simulation_duration_s=self.simulation_duration_s, allow_initial_spillover=self.allow_initial_spillover)
        network.add_nodes(nodes)

        # event properties to copy to geojson output
        copy_props = ['upstreamv','spillv', 'v', 'drain_capacity', 'drainv', 'pctv']

        self.logger.info("Calculating final state")
        initial_volume_attribute = self.input_volume_attribute
        self.logger.info(f"Reading initial water volumes from '{initial_volume_attribute}'")
        eventvalues = network.rain_event(initial_volume_attribute)
        for e in eventvalues:
            node_id = e['nodeid']
            gjn = geojsonnodes_index[node_id]
            for prop in copy_props:
                gjn['properties'][prop] = e[prop]

        self.logger.info("Writing output")
        self.output_finalstatedata.write_geojson_features(geojsonnodes_index.values())

        self.logger.info("Done")

class Network(object):
    """Stream network

    Attributes
    ----------
    nodes : list
        Nodes in the network
    root_nodes : list
        Root nodes. Nodes at the root of the stream tree. Ie nodes that do not have any downstream node
    upstream_tree : dict
        Maps a node id to the ids of nodes one step upstream        
    """

    def __init__(self, simulation_duration_s=3600, allow_initial_spillover=True):
        self.nodes = []
        self.nodes_index = {}
        self.root_nodes = []
        self.upstream_tree = defaultdict(list)
        self._node_rain_values = {}
        self.simulation_duration_s = simulation_duration_s
        self.allow_initial_spillover = allow_initial_spillover
        self.logger = logging.getLogger(__name__)

    def add_nodes(self, nodes):
        """Add a sequence of nodes to the stream network

        Parameters
        ----------
        nodes : sequence
            Nodes to add

        Returns
        -------
        None
        """
        for n in nodes:
            self.add_node(n)

    def add_node(self, node):
        """Add a single node to the stream network

        Parameters
        ----------
        node : dict
            Node to add

        Returns
        -------
        None
        """
        self.nodes.append(node)
        node_id = node['nodeid']
        downstream_id = node['dstrnodeid']
        self.nodes_index[node_id] = node
        self.upstream_tree[downstream_id].append(node_id)
        if downstream_id is None:
            self.root_nodes.append(node_id)

    def _calc_node(self, node_id, volume_attribute):
        node = self.nodes_index[node_id]
        area = float(node['wshed_area'])
        wshed_water_vol = float(node[volume_attribute])
        bspot_capacity = float(node['bspot_vol'])

        # How much is coming from upstream
        upstream_node_ids = self.upstream_tree[node_id]
        upstream_volume = 0.0
        if upstream_node_ids:
            upstream_event_values = [self._node_rain_values[nid] for nid in upstream_node_ids]
            upstream_volume = sum([un['spillv'] for un in upstream_event_values])

        total_water_vol = wshed_water_vol + upstream_volume

        # Bluespot fill before drainage (used to determine initial capacity index)
        bspot_filled_vol_initial = min(total_water_vol, bspot_capacity)

        # Dynamic drainage over time
        drain_vol = 0.0

        # Lists to store step-by-step history
        capacity_history = []
        drained_volume_history = []
        
        # If drainage properties are defined, calculate drainage over the simulation duration
        if node.get('drain_volumes') and node.get('drain_capacity_curve'):
            drain_volumes = np.array([float(x) for x in node['drain_volumes'].split('|')])
            drain_curve   = np.array([float(x) for x in node['drain_capacity_curve'].split('|')])
            
            # Remaining simulation time (in seconds)
            time_remaining = float(self.simulation_duration_s)
            
            # USER CHOICE: 
            # If True, water above bspot_capacity spills instantly and cannot be drained. Conservative flash-flood assumption.
            # If False, the entire water volume can be drained at maximum drainage capacity first. Continuous mass-balance assumption.
            if getattr(self, 'allow_initial_spillover', True):
                current_water_to_drain = bspot_filled_vol_initial
            else:
                current_water_to_drain = total_water_vol
            
            # Find the starting index in drain_volumes based on the chosen initial volume
            search_vol = min(current_water_to_drain, bspot_filled_vol_initial)
            idx = np.searchsorted(drain_volumes, search_vol, side='right') - 1
            idx = max(0, min(idx, len(drain_curve) - 1))
            
            # Dynamic drainage loop (stepping down through the volume tiers)
            while time_remaining > 0 and current_water_to_drain > 0 and idx >= 0:
                c_current = float(drain_curve[idx])
                
                if c_current <= 0:
                    # If drainage capacity is zero for this tier, drainage stops
                    break
                    
                # Determine the floor volume for this step
                # If current water exceeds the bluespot capacity (and allow_initial_spillover=False), 
                # the "floor" is the top of the bluespot capacity where the max curve rate applies.
                if current_water_to_drain > bspot_capacity:
                    v_floor = bspot_capacity
                else:
                    v_floor = float(drain_volumes[idx]) if idx > 0 else 0.0
                
                # Water volume available to drain within this specific tier/segment
                v_available_in_tier = current_water_to_drain - v_floor
                
                # Time required to fully drain this tier
                t_needed = v_available_in_tier / c_current
                
                if time_remaining >= t_needed:
                    # Sufficient time is available to completely drain this tier
                    drain_vol += v_available_in_tier
                    current_water_to_drain -= v_available_in_tier  # = v_floor
                    time_remaining -= t_needed

                    # Record the history for this step
                    capacity_history.append(c_current)
                    drained_volume_history.append(v_available_in_tier)

                    if current_water_to_drain <= 0:
                        break

                    # Move down the curve index
                    idx -= 1

                else:
                    # Time runs out before the tier is fully drained
                    actual_drained = c_current * time_remaining
                    drain_vol += actual_drained
                    current_water_to_drain -= c_current * time_remaining
                    time_remaining = 0.0

                    # Record the partial step history
                    capacity_history.append(c_current)
                    drained_volume_history.append(actual_drained)

        # Update volumes after drainage
        if getattr(self, 'allow_initial_spillover', True):
            spillover = max(0.0, total_water_vol - bspot_capacity)
            bspot_filled_vol = max(0.0, total_water_vol - spillover - drain_vol)
        else:
            effective_water_vol = max(0.0, total_water_vol - drain_vol)
            bspot_filled_vol = min(effective_water_vol, bspot_capacity)
            spillover = max(0.0, effective_water_vol - bspot_capacity)

        # Convert lists to pipe-separated strings for storage (or leave as '0' if empty)
        drain_capacity_str = "|".join(f"{x:.4f}" for x in capacity_history) if capacity_history else "0.0"
        drainv_str = "|".join(f"{x:.2f}" for x in drained_volume_history) if drained_volume_history else "0.0"

        event = dict(nodeid=node['nodeid'])
        event[volume_attribute] = wshed_water_vol
        event['upstreamv'] = upstream_volume
        event['spillv'] = spillover
        event['v'] = bspot_filled_vol
        event['drain_capacity'] = drain_capacity_str  # m3/s at each step
        event['drainv'] = drainv_str            # m3 drained over simulation duration at each step         
        event['pctv'] = None if not bspot_capacity else 100.0 * bspot_filled_vol / bspot_capacity
        self._node_rain_values[node_id] = event

    def _calc_stream_tree(self, root_node_id, volume_attribute):
        tree = deque()
        nodes = deque([root_node_id])
        while nodes:
            n = nodes.pop()
            tree.append(n)
            nodes.extend(self.upstream_tree[n])
        # Now tree has root at the beginning and leaves at the end
        # Process from leaves to root
        while tree:
            n = tree.pop()
            self._calc_node(n, volume_attribute)

    def rain_event(self, volume_attribute):
        """Calculate rain event

        Parameters
        ----------
        mmrain : float
            Amount of rain in mm

        Returns
        -------
        nodes : list
            All nodes in the network with event specific information added
        """
        self._node_rain_values = {}
        for rn in self.root_nodes:
            self._calc_stream_tree(rn, volume_attribute)
        return list(self._node_rain_values.values())
