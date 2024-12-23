import time, random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Network():
    def __init__(self, topology: int = 0, service_rate: int = 3000, system_capacity: int = 10000):
        """
        topology= 0, 1, 2
        """
        self.topology = topology
        self.t = 0  # Tracks t

        #match self.topology:
        
        #    case 0:
        if self.topology == 0:
                self.num_of_switches = 25  # N = number of switches
                self.M = 150  # Number of flows for a given time ; 
                self.Graph = nx.grid_2d_graph(5, 5)
                self.Graph = nx.convert_node_labels_to_integers(self.Graph)  # Convert Nodes into integers
        #    case 1:
        elif self.topology == 1:
                self.num_of_switches = 40  # N = number of switches
                self.M = 150  # Number of flows for a given time ; 
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 61)

                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(40, 60)
        #    case 2:
        elif self.topology == 2: 
                self.num_of_switches = 19  # N = number of switches
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)
                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)
                self.M = 100  # Number of flows for a given time ;
        
                # self.Graph = nx.Graph()

                # for i in range(self.num_of_switches):
                #     self.Graph.add_node(i)

                # backbone_nodes = list(range(5)) # 5 Backbone Nodes
                # regional_nodes = list(set(range(self.num_of_switches)) - set(backbone_nodes)) # Regional Nodes

                # for i in range(len(backbone_nodes)):
                #     for j in range(i + 1, len(backbone_nodes)):
                #         self.Graph.add_edge(backbone_nodes[i], backbone_nodes[j])

                # for node in regional_nodes:
                #     self.Graph.add_edge(node, backbone_nodes[node % len(backbone_nodes)])
         #   case 3:
        elif self.topology == 3: 
                self.num_of_switches = 20
                self.M = 150  # Number of flows for a given time ;
                self.Graph = nx.Graph()

                core_switches = range(4)
                aggregation_switches = range(4, 12)
                edge_switches = range(12, 20)

                core_to_aggregation = {
                    0: [4, 6, 8, 10],
                    1: [4, 6, 8, 10],
                    2: [5, 7, 9, 11],
                    3: [5, 7, 9, 11]
                }
                for core, aggs in core_to_aggregation.items():
                    for agg in aggs:
                        self.Graph.add_edge(core, agg)

                aggregation_to_edge = {
                    4: [12, 13],
                    5: [12, 13],
                    6: [14, 15],
                    7: [14, 15],
                    8: [16, 17],
                    9: [16, 17],
                    10: [18, 19],
                    11: [18, 19]
                }
                for agg, edges in aggregation_to_edge.items():
                    for edge in edges:
                        self.Graph.add_edge(agg, edge)

        for edge in self.Graph.edges:
            self.Graph[edge[0]][edge[1]]['weight'] = np.random.uniform(1, 5)

        #system_capacity = global_system_capacity // self.num_of_switches
        #service_rate = global_service_rate // self.num_of_switches

        # TODO: Initialize for t=0, and populate at runtime.
        
        if self.topology == 3:
            
            edge_switches = list(range(12, 20))
            core_switches = list(range(4))
            self.f = np.array([(random.choice(edge_switches), random.choice(core_switches)) for _ in range(self.M)],
                            dtype='i,i')  # List of (src, dst) tuples of size self.M
        else:
            nodes = list(self.Graph.nodes)
            self.f = np.array([(random.choice(nodes), random.choice(nodes)) for _ in range(self.M)],
                            dtype='i,i')  # List of (src, dst) tuples of size self.M
        
        self.lam = np.array(
            [np.random.poisson(np.random.uniform(10, 300)) for i in range(self.M)])  # Poisson distributed arrival rates
        
        self.u = np.full(self.num_of_switches, service_rate)  # Exponentially distributed service rates
        self.K = np.full(self.num_of_switches,
                         system_capacity)  # Total system capacity of each switch. ; Constant system capacity, for later work we can vary rates.

        # Aggregate arrival rate per switch
        self.agg_lam = np.zeros(self.num_of_switches)  # Need to keep track of this for all switches.
        for i in range(len(self.f)):
            src, dst = self.f[i]
            path = self.shortest_path(src, dst)
            for node in path:
                self.agg_lam[node] += self.lam[i]
            #self.agg_lam[dst] += self.lam[i]  # Aggregate flow into destination switch

        self.rho = self.agg_lam / self.u  # Traffic Intensity for each switch.

        self.e_n = self.get_queue_occupation(self.rho,
                                             self.K)  # Returns an array of expected queue occupation of each switch.
        self.P = self.get_P(self.rho, 
                self.K)  # Returns a probability array(of packets getting lost) of size self.num_of_switches.
        self.e_d = self.get_expected_delays(self.e_n, self.agg_lam,
                                            self.P)  # Returns an expected delay for each switch.

        self.d_k_e2e = self.get_end_end_delay(self.f,
                                              self.e_d)  # Returns an array of size self.M(for each active flow at time t) ; self.d_k_e2e.mean() would be the e2e mean of the network.

        self.e_l = self.get_expected_loss(self.agg_lam,
                                          self.P)  # Returns an array of expected loss of size self.num_of_switches.

        self.fig, self.ax = None, None
    
    def show_network(self):
        #match self.topology:
         #   case 0:
        if self.topology == 0:
                self.pos = self.get_custom_grid_coordinates(5, 5)
        #    case 1:
        elif self.topology == 1:
                self.pos = nx.spring_layout(self.Graph, seed=42)
        #    case 2:
        elif self.topology == 2:
                self.pos = nx.spring_layout(self.Graph, seed=42)
        #    case 3:
        elif self.topology == 3: 
                self.pos = nx.spring_layout(self.Graph, seed=42)
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.plot_graph()
    
    def get_custom_grid_coordinates(self, rows, cols):
        positions = {}
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                positions[node_id] = (j, -i)  # (x, y), flip y for top-to-bottom ordering
        return positions

    def shortest_path(self, src, dest):
        """
        Gets shortest path given src, dest in the graph.
        """
        return nx.shortest_path(self.Graph, source=src, target=dest, weight=lambda s, d, a: self.Graph[s][d]['weight'],
                                method='dijkstra')

    def get_queue_occupation(self, rho, K):
        """
        Gets queue occupation of each switch.
        """
        N = np.zeros(len(rho))
        for i in range(len(rho)):
            if rho[i] < 1 and rho[i] != 0:
                N[i] = (rho[i] / (1 - rho[i])) - (((K[i] + 1) * (rho[i] ** (K[i] + 1))) / (1 - (rho[i] ** (K[i] + 1))))
                #N[i] = (rho[i] / (1 - rho[i])) - (((K[i] + 1) * (rho[i] ** (K[i] + 1))) / ((1 - rho[i]) ** (K[i] + 1)))
                if np.isnan(N[i]):  # Handles invalid value encountered in scalar divide warning.
                    N[i] = rho[i] / (1 - rho[i])
                if N[i] > K[i] / 2: # If queue occupation exceeds queue size, cap to queue size
                    N[i] = K[i] / 2
            if rho[i] >= 1:
                N[i] = K[i] / 2
        return N

    def get_P(self, rho, K):
        """
        Gets probability array of packets getting lost of size self.num_of_switches.
        """
        rho = rho.astype(np.float128)
        K = K.astype(np.float128)
        temp_p = ((1 - rho) * (rho ** K)) / (1 - (rho ** (K + 1)))
        for i in range(len(temp_p)):
            if np.isnan(temp_p[i]):  # Handles invalid value encountered in scalar divide warning.
                temp_p[i] = 1 / (K[i] + 1)
        rho = rho.astype(np.float32)
        K = K.astype(np.float32)
        temp_p = temp_p.astype(np.float32)
        return temp_p

    def get_expected_delays(self, e_n, lam, P):
        """
        Gets expected delay for each switch.
        """
        denominator = (lam * (1 - P))
        mask = np.where(denominator == 0, 1, denominator)
        return e_n / mask

    def get_end_end_delay(self, f, e_d):
        """
        Gets end to end delay for each flow in time self.t.
        """
        D = np.zeros(len(f))
        for i in range(len(f)):
            flow = f[i]
            p_star = self.shortest_path(flow[0], flow[1])
            for n in p_star:
                D[i] += e_d[n]
        return D

    def get_expected_loss(self, lam, P):
        """
        Gets expected loss for each switch.
        """
        print("EXPECTED LOSS")
        print(lam)
        print(P)
        print(lam * P)
        return lam * P

    def update_weights(self, weights, alpha=0.9):
        """
        Updates the weights of existing edges using the provided array of weights and
        selects new arrival rates (lambdas) for random flows.

        Args:
            weights (list or np.array): An array of weights with a length equal to the number of edges.
        """
        if len(weights) != len(self.Graph.edges):
            print(len(weights))
            print(len(self.Graph.edges))
            raise ValueError("Length of weights array must match the number of edges in the graph.")

        # Update Weights
        for (edge, weight) in zip(self.Graph.edges, weights):
            self.Graph[edge[0]][edge[1]]['weight'] = weight #* alpha

        # Update lambdas for random flows
        for i in range(self.M):
            if self.topology == 3:
                if self.t % 2 == 0: # Alternate core to edge and vice versa.
                    src, dst = random.choice(range(12, 20)), random.choice(range(4))
                else:
                    src, dst = random.choice(range(4)), random.choice(range(12, 20))
            else:
                src, dst = random.choice(list(self.Graph.nodes)), random.choice(list(self.Graph.nodes))
            while src == dst:
                dst = random.choice(list(self.Graph.nodes))

            self.f[i] = (src, dst)
            self.lam[i] = np.random.poisson(np.random.uniform(50, 300))  # new arrival rate

        # Recalculate values
        self.agg_lam = np.zeros(self.num_of_switches)  # aggregated arrival rates
        for i in range(len(self.f)):
            src, dst = self.f[i]
            path = self.shortest_path(src, dst)
            for node in path:
                self.agg_lam[node] += self.lam[i]
            #self.agg_lam[dst] += self.lam[i]

        self.rho = self.agg_lam / self.u  # traffic intensity
        self.e_n = self.get_queue_occupation(self.rho, self.K)  # queue occupation
        self.P = self.get_P(self.rho, self.K)  # loss probabilities
        self.e_d = self.get_expected_delays(self.e_n, self.agg_lam, self.P)  # delays
        self.d_k_e2e = self.get_end_end_delay(self.f, self.e_d)  # end-to-end delays
        self.e_l = self.get_expected_loss(self.agg_lam, self.P)  # expected losses
        self.t += 1 # time step

        if self.fig:
            self.plot_graph()

    def plot_graph(self):
        """
        Plot the graph with current weights on the given axis
        Plot the graph with current weights on the given axis
        """
        self.ax.clear()
        nx.draw_networkx_nodes(self.Graph, self.pos, ax=self.ax, node_size=500, node_color='skyblue')
        self.ax.clear()
        nx.draw_networkx_nodes(self.Graph, self.pos, ax=self.ax, node_size=500, node_color='skyblue')

        nx.draw_networkx_edges(self.Graph, self.pos, ax=self.ax, width=1.5, alpha=0.7)
        edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
        nx.draw_networkx_edge_labels(self.Graph, self.pos, edge_labels=edge_labels, ax=self.ax, label_pos=0.5, font_size=8)
        nx.draw_networkx_edges(self.Graph, self.pos, ax=self.ax, width=1.5, alpha=0.7)
        edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
        nx.draw_networkx_edge_labels(self.Graph, self.pos, edge_labels=edge_labels, ax=self.ax, label_pos=0.5, font_size=8)

        nx.draw_networkx_labels(self.Graph, self.pos, ax=self.ax, font_size=10, font_family="sans-serif")
        self.ax.set_title(f"Network Topology {self.topology}")
        self.ax.axis('off')

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        nx.draw_networkx_labels(self.Graph, self.pos, ax=self.ax, font_size=10, font_family="sans-serif")
        self.ax.set_title(f"Network Topology {self.topology}")
        self.ax.axis('off')

if __name__ == '__main__':
    network = Network(topology=3)
    network.show_network()
    num_of_edges = len(network.Graph.edges)
    for i in range(1000):
        temp_W = np.random.uniform(low=1.0, high=5.0, size=num_of_edges)
        network.update_weights(temp_W)
        print(f"Iteration Number: {i}")
