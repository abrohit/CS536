import time, random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Network():
    def __init__(self, topology:int = 0, service_rate:int = 3000, system_capacity:int = 10000):
        """
        topology= 0, 1, 2
        """
        self.topology = topology
        self.t = 0 # Tracks t

        match self.topology:
            case 0:
                self.num_of_switches = 25 # N = number of switches
                self.Graph = nx.grid_2d_graph(5, 5)
                self.Graph = nx.convert_node_labels_to_integers(self.Graph) # Convert Nodes into integers
            case 1:
                self.num_of_switches = 40 # N = number of switches
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 61)
                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(40, 60)

                pos = nx.spring_layout(self.Graph, seed=42)
                nx.draw(self.Graph, pos, with_labels=True, node_color='orange', node_size=500, font_size=8, font_color='black')
                plt.title("InternetMCI Topology with 19 Switches")
                plt.show()
            case 2:
                self.num_of_switches = 19 # N = number of switches
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)
                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)

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

        for edge in self.Graph.edges:
            self.Graph[edge[0]][edge[1]]['weight'] = np.random.uniform(1, 5)

        # TODO: Initialize for t=0, and populate at runtime.
        self.M = 5 # Number of flows for a given time ; TODO: Needs a proper value and changes for every t.
        nodes = list(self.Graph.nodes)
        self.f = np.array([(random.choice(nodes), random.choice(nodes)) for _ in range(self.M)], dtype='i,i') # List of (src, dst) tuples of size self.M
        
        self.lam = np.array([np.random.poisson(np.random.uniform(10, 300)) for i in range(self.M)]) #Poisson distributed arrival rates
        self.u = np.full(self.num_of_switches, service_rate) # Exponentially distributed service rates
        self.K = np.full(self.num_of_switches, system_capacity) # Total system capacity of each switch. ; Constant system capacity, for later work we can vary rates.

        # Aggregate arrival rate per switch
        self.agg_lam = np.zeros(self.num_of_switches) # Need to keep track of this for all switches.
        for i in range(len(self.f)):
            src, dst = self.f[i]
            self.agg_lam[dst] += self.lam[i]  # Aggregate flow into destination switch

        self.rho = self.agg_lam / self.u # Traffic Intensity for each switch.
        
        self.e_n = self.get_queue_occupation(self.rho, self.K) # Returns an array of expected queue occupation of each switch.  
        self.P = self.get_P(self.rho, self.K) # Returns a probability array(of packets getting lost) of size self.num_of_switches.
        self.e_d = self.get_expected_delays(self.e_n, self.agg_lam, self.P) # Returns an expected delay for each switch.

        self.d_k_e2e = self.get_end_end_delay(self.f, self.e_d) # Returns an array of size self.M(for each active flow at time t) ; self.d_k_e2e.mean() would be the e2e mean of the network.

        self.e_l = self.get_expected_loss(self.agg_lam, self.P) # Returns an array of expected loss of size self.num_of_switches.
    
    def shortest_path(self, src, dest):
        """
        Gets shortest path given src, dest in the graph.
        """
        return nx.shortest_path(self.Graph, source=src, target=dest, weight= lambda s,d,a: self.Graph[s][d]['weight'], method='dijkstra')
    
    def get_queue_occupation(self, rho, K):
        """
        Gets queue occupation of each switch.
        """
        N = np.zeros(len(rho))
        for i in range(len(rho)):
            if rho[i] < 1 and rho[i] != 0:
                N[i] = (rho[i] / (1 - rho[i])) - (((K[i] + 1) * (rho[i] ** (K[i] + 1))) / ((1 - rho[i]) ** (K[i] + 1)))
                if np.isnan(N[i]): # Handles invalid value encountered in scalar divide warning.
                    N[i] = rho[i] / (1 - rho[i])
            if rho[i] == 1:
                N[i] = K[i] / 2
        return N

    def get_P(self, rho, K):
        """
        Gets probability array of packets getting lost of size self.num_of_switches.
        """
        temp_p = ( (1 - rho) * (rho ** K) ) / ( 1 - (rho ** (K + 1)) )
        for i in range(len(temp_p)):
            if np.isnan(temp_p[i]): # Handles invalid value encountered in scalar divide warning.
                    temp_p[i] = 0
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
        return lam * P
    
    def update_weights(self, weights):
        """
        Updates the weights of existing edges using the provided array of weights.

        Args:
            weights (list or np.array): An array of weights with a length equal to the number of edges.
        """
        # Check if the length of weights matches the number of edges
        if len(weights) != len(self.Graph.edges):
            raise ValueError("Length of weights array must match the number of edges in the graph.")

        # Iterate through the edges and update their weights
        for (edge, weight) in zip(self.Graph.edges, weights):
            self.Graph[edge[0]][edge[1]]['weight'] = weight

    def plot_graph(self):
        """
        Plot the graph with current weights
        """
        pos = nx.spring_layout(self.Graph)
        nx.draw_networkx_nodes(self.Graph, pos, node_size=500, node_color='skyblue')
        
        nx.draw_networkx_edges(self.Graph, pos, width=1.5, alpha=0.7) # Draw edges
        edge_labels = nx.get_edge_attributes(self.Graph, 'weight')  # Get edge weights
        nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8) # Draw edge weights

        nx.draw_networkx_labels(self.Graph, pos, font_size=10, font_family="sans-serif") # Draw node labels

        plt.title(f"Network Topology {self.topology}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    network = Network()
    num_of_edges = len(network.Graph.edges)
    for i in range(5):
        temp_W = np.random.uniform(low=1.0, high=5.0, size=num_of_edges)
        network.update_weights(temp_W)
    # network.run(interval=1, is_plot=True)
    # network.plot_graph()
